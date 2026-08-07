"""RMCT agent loop for verl (``@register("rmct")``).

One dataset row = one prompt variant of one datapoint. The variant choice
(neutral vs cued prompt) happens at dataset-build time in
``scripts/make_rmct_dataset.py``, so this loop never sees the pair — it
tokenizes whatever ``raw_prompt`` it was handed, samples once, classifies the
answer, and forwards the RMCT bookkeeping (``variant``, ``group_id``,
``parse_ok``, ``trait``) to the trainer through ``extra_fields``.

Why ``extra_fields`` and not the dataset columns: ``AgentLoopWorker._postprocess``
only re-attaches the input non-tensor columns when the async reward loop is
disabled (agent_loop.py:1097), while ``extra_fields`` is propagated
unconditionally (agent_loop.py:1122-1128). Echoing the columns back through
``extra_fields`` makes the port correct either way.

``reward_score`` is set to 0.0 rather than left None on purpose. verl's
``extract_reward`` requires ``rm_scores`` in the batch, and a non-None
``reward_score`` both creates that tensor and short-circuits
``AgentLoopWorker._compute_score``, so RMCT needs no reward manager, no reward
worker, and no scoring model. The actual RMCT signal is not a reward at all —
it is a per-group advantage computed in ``rmct_trainer``.
"""

from __future__ import annotations

import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

from .rmct_core import REFERENCE_VARIANT, TRAINING_VARIANT

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_CLASSIFIER: list[Any] = []


def _classifier():
    """``slime_port.rmct_rollout._classify`` — the exact function the Miles/slime
    port classifies with (mcq-bias ``parse_answer`` + vendored ``matches_bias``).

    Imported lazily: ``slime_port.rmct_rollout`` pulls in ``rollout_writer``,
    which needs ``zstandard``. Install it in the verl image (see README).
    """
    if not _CLASSIFIER:
        try:
            from slime_port.rmct_rollout import _classify
        except ImportError as exc:  # pragma: no cover - environment problem, not logic
            raise ImportError(
                "RMCT answer classification needs the slime port importable "
                "(RMCT_SLIME_PORT_DIR) and its deps installed: pip install zstandard mcq-bias"
            ) from exc
        _CLASSIFIER.append(_classify)
    return _CLASSIFIER[0]


@register("rmct")
class RMCTAgentLoop(AgentLoopBase):
    """Single-turn sampling plus RMCT trait classification."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], priority: int = 0, **kwargs) -> AgentLoopOutput:
        priority = int(priority)
        messages = list(kwargs["raw_prompt"])
        variant = str(kwargs["variant"])
        group_id = str(kwargs["group_id"])
        biased_option = str(kwargs["biased_option"])
        if variant not in (REFERENCE_VARIANT, TRAINING_VARIANT):
            raise ValueError(f"unknown RMCT variant {variant!r}")

        prompt_ids = await self.apply_chat_template(messages)

        metrics = {}
        with simple_timer("generate_sequences", metrics):
            output: TokenOutput = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                priority=priority,
            )
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1

        # A response that hit the length cap is a parse failure by construction —
        # counting it would distort p_hat (slime_port/rmct_rollout.py:_to_rollout).
        # verl's TokenOutput carries no finish_reason, so length is the signal.
        truncated = len(output.token_ids) >= self.response_length
        response_ids = output.token_ids[: self.response_length]
        response_logprobs = output.log_probs[: self.response_length] if output.log_probs else None
        text = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        trait, answer_parsed = _classifier()(text, {"biased_option": biased_option})
        parse_ok = bool(answer_parsed) and not truncated

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=[1] * len(response_ids),
            response_logprobs=response_logprobs,
            num_turns=2,
            metrics=metrics,
            reward_score=0.0,
            extra_fields={
                "variant": variant,
                "group_id": group_id,
                "parse_ok": parse_ok,
                "trait": float(trait),
                "parsed_answer_truncated": truncated,
                "turn_scores": [],
                "tool_rewards": [],
            },
        )
