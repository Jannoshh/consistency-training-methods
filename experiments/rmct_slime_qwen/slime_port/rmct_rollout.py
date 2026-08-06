"""RMCT custom rollout function for slime.

Replaces slime's default GRPO rollout (``--rollout-function-path
slime_port.rmct_rollout.generate_rollout``). Per training step (= one slime
rollout) it:

1. selects ``batch_size`` datapoints from the frozen mcq-bias JSONL
   (deterministic per-epoch shuffle seeded from the config seed),
2. samples ``n_ref_rollouts`` completions on the neutral prompt (perturbation
   0) and ``n_train_rollouts`` on the biased prompt (perturbation 1) via the
   SGLang router,
3. classifies each response with the mcq-bias answer parser (trait =
   ``matches_bias``; a rollout is usable only if the answer parses AND
   logprobs are present — ``unparsed_handling='discard'``),
4. computes rewards/advantages with the parity-tested ported math
   (``pipeline.compute_batch_advantages``),
5. returns only gradient-bearing samples to slime, with the per-sample RMCT
   advantage stored in ``sample.reward`` (the DP-split-safe channel;
   ``rmct_advantage.compute_advantages`` broadcasts it per token and applies
   the KL term), and
6. persists every sampled response as ctm-schema rollout records.

Ported behaviours that differ from ctm's loop, recorded as deviations:
- zero-signal batches return their samples with ``remove_sample=True``
  (loss-masked) instead of skipping the optimizer step outright;
- the per-epoch datapoint order uses this module's seeded shuffle, not ctm's
  global-`random` stream (batching order only — not rate-affecting);
- anchor_weight > 0 is not yet wired (needs the one-time base-rate
  measurement); the config loader rejects it until Phase 4 lands it.
"""

import asyncio
import json
import logging
import random
from argparse import Namespace
from pathlib import Path
from typing import Any

from .pipeline import build_batch_item, compute_batch_advantages
from .rmct_config import RMCTConfig
from .rollout_writer import RolloutWriter
from .types import Rollout

logger = logging.getLogger(__name__)

_STATE: dict[str, Any] = {}


def _load_datapoints(config: RMCTConfig) -> list[dict]:
    rows: list[dict] = []
    required = {"question_id", "unbiased_messages", "biased_messages", "biased_option"}
    for path in config.data_paths:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                missing = required - row.keys()
                if missing:
                    raise ValueError(f"{path}: row missing {sorted(missing)}")
                rows.append(row)
    if len(rows) < config.n_datapoints:
        raise ValueError(f"need {config.n_datapoints} datapoints, have {len(rows)}")
    return rows[: config.n_datapoints]


def _state(args: Namespace) -> dict[str, Any]:
    if _STATE:
        return _STATE
    from .framework import load_tokenizer

    config = RMCTConfig.load()
    if config.advantage.anchor_weight > 0:
        raise NotImplementedError("anchor_weight > 0 needs the base-rate measurement port (Phase 4)")
    _STATE.update(
        config=config,
        datapoints=_load_datapoints(config),
        tokenizer=load_tokenizer(args.hf_checkpoint, trust_remote_code=True),
        writer=RolloutWriter(config.rollout_dir),
        semaphore=asyncio.Semaphore(args.sglang_server_concurrency),
    )
    return _STATE


def _batch_for_step(config: RMCTConfig, datapoints: list[dict], rollout_id: int) -> list[tuple[int, dict]]:
    """Deterministic (seed, epoch)-shuffled batch for this step."""
    steps_per_epoch = (len(datapoints) + config.batch_size - 1) // config.batch_size
    epoch, step_in_epoch = divmod(rollout_id, steps_per_epoch)
    order = list(range(len(datapoints)))
    random.Random(config.seed * 1_000_003 + epoch).shuffle(order)
    chosen = order[step_in_epoch * config.batch_size : (step_in_epoch + 1) * config.batch_size]
    return [(idx, datapoints[idx]) for idx in chosen]


def _perturbation_messages(config: RMCTConfig, datapoint: dict, perturbation_idx: int) -> list[dict]:
    if perturbation_idx == 0 or config.control:
        return datapoint["unbiased_messages"]
    return datapoint["biased_messages"]


def _load_parse_answer():
    """``mcq_bias.parsers.parse_answer`` from the pinned install, loaded by
    file path: the package ``__init__`` imports inspect_ai (absent from the
    slime image), but parsers.py itself is stdlib-only."""
    import importlib.util

    spec = importlib.util.find_spec("mcq_bias")
    parsers_path = Path(spec.origin).parent / "parsers.py"
    mod_spec = importlib.util.spec_from_file_location("_mcq_bias_parsers", parsers_path)
    module = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(module)
    return module.parse_answer


def _matches_bias(answer: str, biased_option: str, negation_prefix: str = "NOT ") -> float | None:
    """Vendored verbatim from mcq_bias.scorers.matches_bias (pin 8c0b46b) —
    scorers.py needs inspect_ai at import time. Parity with the real package is
    asserted by tests/test_rollout_module.py::test_classifier_matches_ctm_adapter."""
    if not biased_option:
        return None
    if biased_option.startswith(negation_prefix):
        return 1.0 if answer != biased_option.removeprefix(negation_prefix) else 0.0
    return 1.0 if answer == biased_option else 0.0


def _classify(text: str, datapoint: dict) -> tuple[float, bool]:
    """(trait_value, answer_parsed) — identical to
    ``ctm_data.adapters.mcq_bias.setting.trait_classifier``."""
    if "parse_answer" not in _STATE:
        _STATE["parse_answer"] = _load_parse_answer()
    answer = _STATE["parse_answer"](text)
    if answer is None:
        return 0.0, False
    if not datapoint["biased_option"]:
        raise ValueError("datapoint lacks biased_option")
    return float(_matches_bias(answer, datapoint["biased_option"])), True


async def _sample_once(
    args: Namespace,
    state: dict[str, Any],
    prompt_ids: list[int],
    sampling_params: dict[str, Any],
) -> dict[str, Any]:
    from .framework import post

    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    payload = {"sampling_params": sampling_params, "return_logprob": True, "input_ids": prompt_ids}
    # LoRA runs (Miles): the trained adapter lives in a named SGLang slot and
    # is only applied when the request selects it — without this, sampling
    # silently uses the frozen base (off-policy drift each optimizer step).
    try:
        from miles.utils.lora import LORA_ADAPTER_NAME, lora_rollout_enabled

        if lora_rollout_enabled(args):
            payload["lora_path"] = LORA_ADAPTER_NAME
    except ImportError:
        pass  # slime: full-param weight replacement, no adapter routing
    async with state["semaphore"]:
        output = await post(url, payload)
    meta = output["meta_info"]
    pairs = meta.get("output_token_logprobs") or []
    return {
        "text": output["text"],
        "tokens": [p[1] for p in pairs],
        "logprobs": [p[0] for p in pairs],
        "finish_reason": (meta.get("finish_reason") or {}).get("type"),
        "cached_tokens": meta.get("cached_tokens", 0),
        "prompt_tokens": meta.get("prompt_tokens", 0),
    }


async def _collect_datapoint(
    args: Namespace, state: dict[str, Any], dp_idx: int, datapoint: dict
) -> tuple[int, dict, dict[int, list[dict]], dict[int, list[int]]]:
    """Sample all perturbation populations for one datapoint."""
    config: RMCTConfig = state["config"]
    tokenizer = state["tokenizer"]
    sampling_params = {
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
        "max_new_tokens": config.max_new_tokens,
        "no_stop_trim": True,
        "spaces_between_special_tokens": False,
    }
    counts = {0: config.n_ref_rollouts, 1: config.n_train_rollouts}
    prompt_ids: dict[int, list[int]] = {}
    tasks = []
    for pert, n in counts.items():
        messages = _perturbation_messages(config, datapoint, pert)
        ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        if not isinstance(ids, list):  # BatchEncoding on some tokenizer versions
            ids = ids["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        prompt_ids[pert] = [int(t) for t in ids]
        for _ in range(n):
            tasks.append((pert, asyncio.create_task(_sample_once(args, state, ids, dict(sampling_params)))))
    raw: dict[int, list[dict]] = {pert: [] for pert in counts}
    for pert, task in tasks:
        raw[pert].append(await task)
    return dp_idx, datapoint, raw, prompt_ids


def _to_rollout(sample_output: dict, datapoint: dict, perturbation_idx: int) -> Rollout:
    trait, answer_parsed = _classify(sample_output["text"], datapoint)
    truncated = sample_output["finish_reason"] == "length"
    has_logprobs = bool(sample_output["logprobs"])
    return Rollout(
        tokens=sample_output["tokens"],
        logprobs=sample_output["logprobs"],
        text=sample_output["text"],
        trait_value=trait,
        perturbation_idx=perturbation_idx,
        # usable = answer parsed AND logprobs present AND not truncated (a
        # truncated response is a parse failure by construction; counting it
        # would distort p_hat).
        parsed_successfully=answer_parsed and has_logprobs and not truncated,
        answer_parsed=answer_parsed and not truncated,
        has_logprobs=has_logprobs,
    )


def generate_rollout(args: Namespace, rollout_id: int, data_source: Any, evaluation: bool = False):
    """slime entry point (``--rollout-function-path``)."""
    from .framework import run

    assert not evaluation, "RMCT evaluation goes through the Inspect runner, not slime eval"
    return run(_generate_rollout_async(args, rollout_id))


async def _generate_rollout_async(args: Namespace, rollout_id: int) -> "Any":
    from .framework import RolloutFnTrainOutput
    from .framework import Sample

    state = _state(args)
    config: RMCTConfig = state["config"]
    step = rollout_id + 1  # ctm steps are 1-based

    batch = _batch_for_step(config, state["datapoints"], rollout_id)
    collected = await asyncio.gather(
        *[_collect_datapoint(args, state, dp_idx, datapoint) for dp_idx, datapoint in batch]
    )

    # Rates + advantages via the parity-tested port.
    port_items = []
    rollouts_by_dp: dict[int, dict[int, list[Rollout]]] = {}
    prompt_ids_by_dp: dict[int, dict[int, list[int]]] = {}
    cached_tokens = prompt_tokens = 0
    for dp_idx, datapoint, raw, prompt_ids in collected:
        prompt_ids_by_dp[dp_idx] = prompt_ids
        rollouts: dict[int, list[Rollout]] = {}
        for pert, outputs in raw.items():
            for output in outputs:
                rollout = _to_rollout(output, datapoint, pert)
                rollouts.setdefault(pert, []).append(rollout)
                cached_tokens += output["cached_tokens"]
                prompt_tokens += output["prompt_tokens"]
        rollouts_by_dp[dp_idx] = rollouts
        item = build_batch_item(datapoint_idx=dp_idx, rollouts=rollouts, reference_indices=[0], training_indices=[1])
        if item is not None:
            port_items.append(item)
    result = compute_batch_advantages(port_items, config.advantage)
    advantage_by_id = {
        id(rollout): (reward, advantage)
        for rollout, reward, advantage in zip(
            result.grad_rollouts, result.consistency_rewards + result.anchor_rewards, result.advantages
        )
    }
    item_by_dp = {item.datapoint_idx: item for item in port_items}

    # Build slime Samples for gradient-bearing rollouts; persist everything.
    groups: list[list[Sample]] = []
    records: list[dict] = []
    sample_index = 0
    completion_lengths: list[int] = []
    for dp_idx, datapoint, raw, prompt_ids in collected:
        item = item_by_dp.get(dp_idx)
        group: list[Sample] = []
        for pert, rollout_list in rollouts_by_dp[dp_idx].items():
            for rollout in rollout_list:
                completion_lengths.append(len(rollout.tokens))
                candidate = advantage_by_id.get(id(rollout))
                if candidate is not None:
                    reward, advantage = candidate
                    skipped = not result.has_signal
                    skip_reason = "zero_advantage_batch" if skipped else None
                else:
                    reward = advantage = None
                    skipped = True
                    if not rollout.answer_parsed:
                        skip_reason = "answer_parse_failure"
                    elif not rollout.has_logprobs:
                        skip_reason = "missing_logprobs"
                    else:
                        skip_reason = "rate_only"
                # Every TRAINING-perturbation rollout becomes a Sample, so each
                # generation yields exactly batch_size * n_train_rollouts
                # samples and --global-batch-size equals that count (one
                # optimizer step per generation, divisible by any DP size).
                # Skipped rollouts (parse failure / zero-signal batch) carry an
                # all-zero loss mask: zero policy gradient AND zero KL — exact
                # ctm skip semantics without shrinking the batch. Reference
                # (perturbation 0) rollouts never enter the trainer, as in ctm.
                if pert != 0:
                    trainable = candidate is not None and not skipped
                    group.append(
                        Sample(
                            group_index=dp_idx,
                            index=sample_index,
                            rollout_id=rollout_id,
                            prompt=state["tokenizer"].decode(prompt_ids[pert]),
                            tokens=prompt_ids[pert] + rollout.tokens,
                            response=rollout.text,
                            response_length=len(rollout.tokens),
                            rollout_log_probs=rollout.logprobs,
                            loss_mask=[1 if trainable else 0] * len(rollout.tokens),
                            reward=advantage if trainable else 0.0,
                            remove_sample=False,
                            status=Sample.Status.COMPLETED,
                            metadata={"datapoint_idx": dp_idx, "perturbation_idx": pert, "step": step},
                        )
                    )
                    if trainable:
                        sample_index += 1
                records.append(
                    {
                        "step": step,
                        "epoch": rollout_id * config.batch_size // max(len(state["datapoints"]), 1),
                        "datapoint_idx": dp_idx,
                        "perturbation_idx": pert,
                        "role": "train" if candidate is not None else "rate",
                        "sample_source": "policy",
                        "prompt_text": state["tokenizer"].decode(prompt_ids[pert]),
                        "prompt_context": {},
                        "completion_text": rollout.text,
                        "trait_value": rollout.trait_value,
                        "parsed_successfully": rollout.parsed_successfully,
                        "grader_failed": False,
                        "reward": reward,
                        "advantage": advantage,
                        "skipped_from_training": skipped,
                        "skip_reason": skip_reason,
                        "p_hat": (item.p_hat.get(pert) if item and candidate is not None else None),
                        "p_ref": item.p_ref if item else None,
                        "p_ref_init": None,
                    }
                )
        if group:
            groups.append(group)

    state["writer"].write_step(step, records)

    n_total = sum(len(outs) for _, _, raw, _ in collected for outs in raw.values())
    parse_ok = sum(
        1 for rollouts in rollouts_by_dp.values() for rs in rollouts.values() for r in rs if r.parsed_successfully
    )
    completion_lengths.sort()
    p50 = completion_lengths[len(completion_lengths) // 2] if completion_lengths else 0
    p99 = completion_lengths[int(len(completion_lengths) * 0.99)] if completion_lengths else 0
    metrics = {
        "rmct/parse_rate": parse_ok / max(n_total, 1),
        "rmct/p_ref_mean": (sum(i.p_ref for i in port_items) / len(port_items) if port_items else 0.0),
        "rmct/p_hat_mean": (
            sum(rate for i in port_items for rate in i.p_hat.values()) / max(sum(len(i.p_hat) for i in port_items), 1)
        ),
        "rmct/gap_mean": (
            sum(rate - i.p_ref for i in port_items for rate in i.p_hat.values())
            / max(sum(len(i.p_hat) for i in port_items), 1)
        ),
        "rmct/has_signal": float(result.has_signal),
        "rmct/n_grad_samples": float(sample_index),
        "rmct/completion_len_p50": float(p50),
        "rmct/completion_len_p99": float(p99),
        "rmct/truncated_frac": (
            sum(
                1
                for rollouts in rollouts_by_dp.values()
                for rs in rollouts.values()
                for r in rs
                if not r.answer_parsed and r.has_logprobs
            )
            / max(n_total, 1)
        ),
        "rmct/prefix_cache_hit_rate": cached_tokens / max(prompt_tokens, 1),
    }
    logger.info(
        "RMCT step %d: %d grad samples, parse_rate=%.3f, gap=%.4f, cache_hit=%.2f",
        step,
        sample_index,
        metrics["rmct/parse_rate"],
        metrics["rmct/gap_mean"],
        metrics["rmct/prefix_cache_hit_rate"],
    )
    return RolloutFnTrainOutput(samples=groups, metrics=metrics)
