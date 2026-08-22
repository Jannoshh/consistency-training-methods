"""VDCT agent loop for verl (``@register("vdct")``).

One dataset row = one (variant, kind) cell of one datapoint; the routing
happens at dataset-build time in ``scripts/make_vdct_dataset.py``. The loop
dispatches on the row's ``kind``:

- ``distribution`` rows carry the elicitation instruction in their prompt;
  the response's stated distribution is parsed strictly
  (``vdct_elicitation.parse_option_distribution``) and forwarded dense in
  ``option_labels`` order.
- ``answer`` rows carry the unmodified paired prompt; the response is
  classified with the SAME parser the RMCT/slime port uses
  (``mcq_bias.parsers.parse_answer``, loaded via ``slime_port``), and the
  parsed answer's option index is forwarded — the log-score term needs the
  actual option, which the RMCT loop's one-bit ``trait`` cannot supply. This
  is why answer rows route here rather than to the ``rmct`` loop (README
  deviation V1); prompt and sampling are otherwise identical. The trait bit
  itself is not forwarded: it is derivable offline from the dumped
  ``answer_option`` and ``biased_option``.

All bookkeeping goes through ``extra_fields`` (propagated unconditionally by
``AgentLoopWorker._postprocess``, unlike dataset columns — same rationale as
the RMCT loop). ``reward_score=0.0`` creates verl's ``rm_scores`` tensor and
short-circuits reward computation; the real signal is the per-group advantage
computed in ``vdct_trainer``.

A response that hits the length cap is a parse failure by construction for
both kinds (verl's ``TokenOutput`` carries no finish_reason, so length is the
signal — same as the RMCT loop).
"""

from __future__ import annotations

import functools
import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

from .vdct_core import ANSWER_KIND, DISTRIBUTION_KIND, REFERENCE_VARIANT, TRAINING_VARIANT
from .vdct_elicitation import SUM_TOLERANCE, parse_option_distribution

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@functools.cache
def _parse_answer():
    """``mcq_bias.parsers.parse_answer`` — the exact parser the RMCT/slime
    port classifies with. Imported lazily: ``slime_port.rmct_rollout`` pulls
    in ``rollout_writer``, which needs ``zstandard`` (install it in the verl
    image, see README)."""
    try:
        from slime_port.rmct_rollout import _load_parse_answer
    except ImportError as exc:  # pragma: no cover - environment problem, not logic
        raise ImportError(
            "VDCT answer classification needs the slime port importable "
            "(bootstrapped via vdct_core / RMCT_SLIME_PORT_DIR) and its deps "
            "installed: pip install zstandard mcq-bias"
        ) from exc
    return _load_parse_answer()


@register("vdct")
class VDCTAgentLoop(AgentLoopBase):
    """Single-turn sampling plus distribution parsing / answer classification."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        # Parse strictness is reward policy (format compliance is trained), so
        # it lives in the run's resolved config, not in library code.
        self.parse_sum_tolerance = float(self.config.get("vdct", {}).get("parse_sum_tolerance", SUM_TOLERANCE))

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], priority: int = 0, **kwargs) -> AgentLoopOutput:
        priority = int(priority)
        messages = list(kwargs["raw_prompt"])
        variant = str(kwargs["variant"])
        kind = str(kwargs["kind"])
        group_id = str(kwargs["group_id"])
        biased_option = str(kwargs["biased_option"])
        option_labels = [str(label) for label in kwargs["option_labels"]]
        if variant not in (REFERENCE_VARIANT, TRAINING_VARIANT):
            raise ValueError(f"unknown VDCT variant {variant!r}")
        if kind not in (DISTRIBUTION_KIND, ANSWER_KIND):
            raise ValueError(f"unknown VDCT kind {kind!r}")

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

        truncated = len(output.token_ids) >= self.response_length
        response_ids = output.token_ids[: self.response_length]
        response_logprobs = output.log_probs[: self.response_length] if output.log_probs else None
        # Decoding a multi-thousand-token response is blocking work; keep it
        # off the shared event loop like verl's own tokenizer calls.
        text = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(response_ids, skip_special_tokens=True)
        )

        option_distribution: list[float] | None = None
        answer_option: str | None = None
        answer_index: int | None = None
        if kind == DISTRIBUTION_KIND:
            parsed = parse_option_distribution(text, option_labels, sum_tolerance=self.parse_sum_tolerance)
            parse_ok = parsed is not None and not truncated
            option_distribution = parsed if parse_ok else None
        else:
            answer = _parse_answer()(text)
            index = option_labels.index(answer) if answer in option_labels else None
            parse_ok = index is not None and not truncated
            answer_option = answer if parse_ok else None
            answer_index = index if parse_ok else None

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
                "kind": kind,
                "group_id": group_id,
                "parse_ok": parse_ok,
                "option_distribution": option_distribution,
                "answer_option": answer_option,
                "answer_index": answer_index,
                "option_labels": option_labels,
                "biased_option": biased_option,
                "response_truncated": truncated,
                "turn_scores": [],
                "tool_rewards": [],
            },
        )
