"""Pure VDCT math for the verl port — no Ray, no torch.distributed, no verl imports.

Everything here is CPU-testable in-process (``tests/test_vdct_math.py``),
following the RMCT recipe's pattern. Responsibilities:

1. distribution arithmetic — :func:`js_divergence`, :func:`log_score`,
   :func:`entropy`, :func:`total_variation`, :func:`mean_distribution`;
2. :func:`compute_row_advantages` — turn a flat list of verl batch rows into
   per-row rewards, advantages, and trainability flags. Per-group
   standardization is NOT reimplemented: it delegates to the parity-tested
   ``slime_port.advantages.normalize_grouped`` (per-item mode standardizes
   each 8-rollout distribution population on its own);
3. :func:`build_rows` / :func:`merge_dump_fields` — the trainer's row view and
   rollout-dump threading, kept verl-free so they stay testable.

The batch-centered KL term is REUSED from the RMCT recipe
(``recipe.rmct.rmct_core.centered_kl_penalty``), re-exported here for the
trainer; importing it also bootstraps the slime_port ``sys.path`` entry.

Reward model (see the plan and README)
--------------------------------------
Distribution rollouts carry gradient on both sides; answer rollouts never do
(they exist only as outcome samples for the proper-scoring term — the
structural block against self-fulfilling distributions).

    training side:   r = -w_c * JS(q, q_ref_target) + lambda * mean_m log(max(q[a_m], eps))
    reference side:  r =                              lambda * mean_m log(max(q[a_m], eps))

``w_c`` is ``consistency_weight`` (default 1). The plan's ablation arms map
onto the two weights: ``lambda = 0`` is the no-proper-scoring arm and
``w_c = 0`` is the proper-scoring-only arm.

``q_ref_target`` is this step's mean parsed reference-side distribution for
the group. Answers ``a_m`` always come from the SAME side's answer rollouts.
An unparseable distribution is excluded from ``q_ref_target`` and receives
the worst-case reward ``-w_c * JS_MAX + lambda * log(eps)`` — at or below any
parseable reward on either side — while still carrying gradient, so format
compliance is itself trained.

There is NO anchor term (2026-08-22 decision): the plan's frozen
``q_ref_initial`` reference target is dropped for now, so the reference side
trains on calibration alone and nothing structurally blocks the pair from
co-drifting to a matched-but-shifted distribution — the KL term against the
frozen base, the entropy/calibration diagnostics, and the control arm are the
monitors. Same status as the RMCT port's unsupported ``anchor_weight``.

Documented edge policies (all visible in metrics / skip reasons):

- every reference-side distribution unparsed → the group has no consistency
  target, so its training-side distribution rows are skipped whole
  (``no_reference_target``, the analog of RMCT's ``no_reference_rate``);
- no parsed same-side answer rollouts → the log-score term is omitted for
  that side's rewards this step.

Locating the RMCT recipe
------------------------
``recipe`` is an implicit namespace package; the RMCT portion lives in the
sibling ``experiments/rmct_verl``. Override the location with the
``VDCT_RMCT_VERL_DIR`` environment variable when only this experiment
directory was synced.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path


def _rmct_verl_root() -> Path:
    override = os.environ.get("VDCT_RMCT_VERL_DIR")
    if override:
        return Path(override).resolve()
    # .../experiments/vdct_verl/recipe/vdct/vdct_core.py -> .../experiments
    return Path(__file__).resolve().parents[3] / "rmct_verl"


_RMCT_ROOT = _rmct_verl_root()
if not (_RMCT_ROOT / "recipe" / "rmct" / "rmct_core.py").exists():
    raise ImportError(
        f"RMCT recipe not found under {_RMCT_ROOT}. Set VDCT_RMCT_VERL_DIR to the "
        "directory containing recipe/rmct (experiments/rmct_verl)."
    )
if str(_RMCT_ROOT) not in sys.path:
    sys.path.insert(0, str(_RMCT_ROOT))

from recipe.rmct.rmct_core import centered_kl_penalty  # re-export; bootstraps slime_port
from slime_port.advantages import normalize_grouped

from .vdct_schema import (
    ANSWER_KIND,
    DISTRIBUTION_KIND,
    REFERENCE_VARIANT,
    TRAINING_VARIANT,
    VDCTConfig,
)

__all__ = [
    "ANSWER_KIND",
    "DISTRIBUTION_KIND",
    "JS_MAX",
    "REFERENCE_VARIANT",
    "TRAINING_VARIANT",
    "VDCTBatchResult",
    "VDCTConfig",
    "VDCTRow",
    "build_rows",
    "centered_kl_penalty",
    "compute_row_advantages",
    "dump_columns",
    "entropy",
    "js_divergence",
    "log_score",
    "mean_distribution",
    "mean_side_distributions",
    "merge_dump_fields",
    "rows_from_records",
    "total_variation",
    "worst_case_reward",
]

# JS divergence in nats is bounded by ln 2; the log-score term uses natural
# logs too, so the two reward terms share one scale.
JS_MAX = math.log(2.0)

_SIGNAL_EPS = 1e-8  # same has-signal threshold as slime_port.pipeline


@dataclass(frozen=True)
class VDCTRow:
    """One verl batch row, reduced to what the VDCT math reads.

    ``option_distribution`` is a dense vector in the dataset's frozen
    ``option_labels`` order; ``answer_index`` indexes the same order. Fields
    are only meaningful for the row's ``kind`` (and, for
    ``option_distribution``, when ``parse_ok``).
    """

    group_id: str
    variant: str
    kind: str
    parse_ok: bool
    option_distribution: tuple[float, ...] | None = None
    answer_index: int | None = None


@dataclass
class VDCTBatchResult:
    """Per-row outputs, aligned index-for-index with the input row list."""

    rewards: list[float]
    advantages: list[float]
    trainable: list[bool]
    skip_reasons: list[str | None]
    has_signal: bool
    metrics: dict[str, float] = field(default_factory=dict)
    # Per-group consistency target used this step (None = no parsed
    # reference-side distributions, group's training rows skipped).
    q_ref_target: dict[str, list[float] | None] = field(default_factory=dict)


# ── Distribution arithmetic ──────────────────────────────────────────────────


def _check_distribution(p: list[float] | tuple[float, ...], name: str) -> None:
    if not p:
        raise ValueError(f"{name} is empty")
    if any(v < 0.0 for v in p):
        raise ValueError(f"{name} has negative entries: {list(p)}")


def entropy(p: list[float] | tuple[float, ...]) -> float:
    """Shannon entropy in nats, with the 0·log 0 = 0 convention."""
    _check_distribution(p, "distribution")
    return -sum(v * math.log(v) for v in p if v > 0.0)


def _kl(p, q) -> float:
    total = 0.0
    for pi, qi in zip(p, q):
        if pi > 0.0:
            if qi <= 0.0:
                return math.inf
            total += pi * math.log(pi / qi)
    return total


def js_divergence(p: list[float] | tuple[float, ...], q: list[float] | tuple[float, ...]) -> float:
    """Jensen–Shannon divergence in nats: 0.5·KL(p‖m) + 0.5·KL(q‖m), m=(p+q)/2.

    Bounded by ``JS_MAX`` = ln 2; the midpoint mixture makes it finite for any
    pair of distributions on the same support.
    """
    _check_distribution(p, "p")
    _check_distribution(q, "q")
    if len(p) != len(q):
        raise ValueError(f"length mismatch: {len(p)} vs {len(q)}")
    m = [(pi + qi) / 2.0 for pi, qi in zip(p, q)]
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def total_variation(p: list[float] | tuple[float, ...], q: list[float] | tuple[float, ...]) -> float:
    if len(p) != len(q):
        raise ValueError(f"length mismatch: {len(p)} vs {len(q)}")
    return 0.5 * sum(abs(pi - qi) for pi, qi in zip(p, q))


def mean_distribution(dists: list[tuple[float, ...]]) -> list[float]:
    if not dists:
        raise ValueError("no distributions to average")
    length = len(dists[0])
    if any(len(d) != length for d in dists):
        raise ValueError("distribution length mismatch")
    return [sum(d[i] for d in dists) / len(dists) for i in range(length)]


def mean_side_distributions(rows: list[VDCTRow]) -> dict[str, dict[str, list[float] | None]]:
    """Mean parsed stated distribution per (group, variant) side, or None.

    The one definition of side-level aggregation, shared by the training-time
    ``vdct/tv_cue_mean`` metric and the offline diagnostics' cue-invariance
    report so the two cannot drift apart.
    """
    per_side: dict[str, dict[str, list[tuple[float, ...]]]] = {}
    for row in rows:
        if row.kind != DISTRIBUTION_KIND:
            continue
        group = per_side.setdefault(row.group_id, {REFERENCE_VARIANT: [], TRAINING_VARIANT: []})
        if row.parse_ok:
            group[row.variant].append(row.option_distribution)
    return {
        group_id: {variant: mean_distribution(dists) if dists else None for variant, dists in sides.items()}
        for group_id, sides in per_side.items()
    }


def log_score(
    q: list[float] | tuple[float, ...],
    answer_indices: list[int],
    epsilon: float,
) -> float:
    """Proper-scoring term: mean over answers of log(max(q[a], eps)).

    Expected value is uniquely maximized when ``q`` equals the true answer
    distribution of the policy on that prompt (log score is strictly proper);
    the epsilon floor only bounds the penalty for zero-mass answers.
    """
    if not answer_indices:
        raise ValueError("log_score needs at least one answer sample")
    for a in answer_indices:
        if not 0 <= a < len(q):
            raise ValueError(f"answer index {a} outside distribution of length {len(q)}")
    return sum(math.log(max(q[a], epsilon)) for a in answer_indices) / len(answer_indices)


def worst_case_reward(config: VDCTConfig) -> float:
    """Reward for an unparseable distribution: maximal (weighted) JS penalty
    plus the floored log score — at or below any parseable distribution's
    reward on either side (reference-side parseable rewards are bounded below
    by ``lambda * log(eps)``; see the README's deviations table for the
    per-side asymmetry this deliberately keeps)."""
    return -config.consistency_weight * JS_MAX + config.lambda_log_score * math.log(config.epsilon)


# ── Row → advantage assembly ─────────────────────────────────────────────────


def _validate_row(row_idx: int, row: VDCTRow) -> None:
    if row.variant not in (REFERENCE_VARIANT, TRAINING_VARIANT):
        raise ValueError(f"row {row_idx}: unknown variant {row.variant!r}")
    if row.kind not in (DISTRIBUTION_KIND, ANSWER_KIND):
        raise ValueError(f"row {row_idx}: unknown kind {row.kind!r}")
    if row.kind == DISTRIBUTION_KIND and row.parse_ok and row.option_distribution is None:
        raise ValueError(f"row {row_idx}: parse_ok distribution row without option_distribution")
    if row.kind == ANSWER_KIND and row.parse_ok and row.answer_index is None:
        raise ValueError(f"row {row_idx}: parse_ok answer row without answer_index")


def compute_row_advantages(rows: list[VDCTRow], config: VDCTConfig | None = None) -> VDCTBatchResult:
    """VDCT reward and advantage per row; see the module docstring for the
    reward model and edge policies.

    Standardization slices are each (group, variant) distribution population
    — the plan's "per-group standardization over each 8-rollout distribution
    group" — handed to ``slime_port.advantages.normalize_grouped`` in
    first-appearance group order, reference side then training side.
    """
    config = config or VDCTConfig()
    n = len(rows)
    rewards = [0.0] * n
    advantages = [0.0] * n
    trainable = [False] * n
    skip_reasons: list[str | None] = [None] * n

    group_order: list[str] = []
    by_group: dict[str, list[int]] = {}
    for row_idx, row in enumerate(rows):
        _validate_row(row_idx, row)
        if row.group_id not in by_group:
            group_order.append(row.group_id)
            by_group[row.group_id] = []
        by_group[row.group_id].append(row_idx)
        if row.kind == ANSWER_KIND:
            skip_reasons[row_idx] = "answer_row"

    flat_rewards: list[float] = []
    flat_row_indices: list[int] = []
    slices: list[tuple[int, int]] = []
    # The reward target and the tv_cue_mean metric read the SAME aggregation,
    # so they cannot drift (mean_side_distributions is also the diagnostics'
    # definition of a side mean).
    side_means = mean_side_distributions(rows)
    q_ref_target: dict[str, list[float] | None] = {}

    n_sides_missing_answers = 0
    n_unparsed_dist = 0
    js_train = [0.0, 0]
    log_score_sums = {REFERENCE_VARIANT: [0.0, 0], TRAINING_VARIANT: [0.0, 0]}
    entropy_sums = {REFERENCE_VARIANT: [0.0, 0], TRAINING_VARIANT: [0.0, 0]}

    for group_id in group_order:
        indices = by_group[group_id]
        dist_rows = {
            variant: [i for i in indices if rows[i].kind == DISTRIBUTION_KIND and rows[i].variant == variant]
            for variant in (REFERENCE_VARIANT, TRAINING_VARIANT)
        }
        answers = {
            variant: [
                rows[i].answer_index
                for i in indices
                if rows[i].kind == ANSWER_KIND and rows[i].variant == variant and rows[i].parse_ok
            ]
            for variant in (REFERENCE_VARIANT, TRAINING_VARIANT)
        }
        target = side_means.get(group_id, {}).get(REFERENCE_VARIANT)
        q_ref_target[group_id] = target

        for variant in (REFERENCE_VARIANT, TRAINING_VARIANT):
            side_rows = dist_rows[variant]
            if not side_rows:
                continue
            if variant == TRAINING_VARIANT and target is None:
                # No parsed reference-side distributions: no consistency
                # target, so the training side is skipped whole — the analog
                # of RMCT's no_reference_rate drop.
                for i in side_rows:
                    skip_reasons[i] = "no_reference_target"
                continue
            side_answers = answers[variant]
            if not side_answers and any(rows[i].parse_ok for i in side_rows):
                n_sides_missing_answers += 1
            slice_start = len(flat_rewards)
            for i in side_rows:
                row = rows[i]
                if not row.parse_ok:
                    reward = worst_case_reward(config)
                    n_unparsed_dist += 1
                else:
                    q = row.option_distribution
                    reward = 0.0
                    # Only the training side has a consistency term — the
                    # reference side has no anchor (see module docstring).
                    if variant == TRAINING_VARIANT:
                        js = js_divergence(q, target)
                        reward -= config.consistency_weight * js
                        js_train[0] += js
                        js_train[1] += 1
                    entropy_sums[variant][0] += entropy(q)
                    entropy_sums[variant][1] += 1
                    if side_answers:
                        score = log_score(q, side_answers, config.epsilon)
                        reward += config.lambda_log_score * score
                        log_score_sums[variant][0] += score
                        log_score_sums[variant][1] += 1
                rewards[i] = reward
                flat_rewards.append(reward)
                flat_row_indices.append(i)
            slices.append((slice_start, len(flat_rewards)))

    flat_advantages = normalize_grouped(flat_rewards, slices, config.normalization)
    has_signal = bool(flat_advantages) and any(abs(a) >= _SIGNAL_EPS for a in flat_advantages)

    for flat_idx, row_idx in enumerate(flat_row_indices):
        if has_signal:
            advantages[row_idx] = float(flat_advantages[flat_idx])
            trainable[row_idx] = True
        else:
            advantages[row_idx] = 0.0
            skip_reasons[row_idx] = "zero_advantage_batch"

    n_dist = sum(1 for row in rows if row.kind == DISTRIBUTION_KIND)
    n_answer = n - n_dist
    n_dist_parsed = sum(1 for row in rows if row.kind == DISTRIBUTION_KIND and row.parse_ok)
    n_answer_parsed = sum(1 for row in rows if row.kind == ANSWER_KIND and row.parse_ok)

    def _side_parse_rate(variant: str) -> float:
        side = [row for row in rows if row.kind == DISTRIBUTION_KIND and row.variant == variant]
        return sum(row.parse_ok for row in side) / max(len(side), 1)

    def _mean(pair: list[float]) -> float:
        return pair[0] / pair[1] if pair[1] else 0.0

    tv_values = [
        total_variation(sides[TRAINING_VARIANT], sides[REFERENCE_VARIANT])
        for sides in side_means.values()
        if sides[REFERENCE_VARIANT] is not None and sides[TRAINING_VARIANT] is not None
    ]

    metrics = {
        "vdct/dist_parse_rate": n_dist_parsed / max(n_dist, 1),
        "vdct/dist_parse_rate_ref": _side_parse_rate(REFERENCE_VARIANT),
        "vdct/dist_parse_rate_train": _side_parse_rate(TRAINING_VARIANT),
        "vdct/answer_parse_rate": n_answer_parsed / max(n_answer, 1),
        "vdct/js_train_mean": _mean(js_train),
        "vdct/log_score_train_mean": _mean(log_score_sums[TRAINING_VARIANT]),
        "vdct/log_score_ref_mean": _mean(log_score_sums[REFERENCE_VARIANT]),
        "vdct/entropy_train_mean": _mean(entropy_sums[TRAINING_VARIANT]),
        "vdct/entropy_ref_mean": _mean(entropy_sums[REFERENCE_VARIANT]),
        "vdct/tv_cue_mean": sum(tv_values) / len(tv_values) if tv_values else 0.0,
        "vdct/has_signal": float(has_signal),
        "vdct/n_groups": float(len(group_order)),
        "vdct/n_grad_rows": float(sum(trainable)),
        "vdct/grad_row_frac": sum(trainable) / max(n_dist, 1),
        "vdct/n_unparsed_dist_rows": float(n_unparsed_dist),
        "vdct/sides_missing_answer_samples": float(n_sides_missing_answers),
        "vdct/lambda_log_score": config.lambda_log_score,
        "vdct/consistency_weight": config.consistency_weight,
    }
    for reason in skip_reasons:
        if reason is not None:
            key = f"vdct/skip/{reason}"
            metrics[key] = metrics.get(key, 0.0) + 1.0

    return VDCTBatchResult(
        rewards=rewards,
        advantages=advantages,
        trainable=trainable,
        skip_reasons=skip_reasons,
        has_signal=has_signal,
        metrics=metrics,
        q_ref_target=q_ref_target,
    )


# ── verl-batch adapters (kept here so they stay CPU-testable) ────────────────

REQUIRED_ROW_FIELDS = ("group_id", "variant", "kind", "parse_ok")


def dump_columns(rows: list[VDCTRow], result: VDCTBatchResult) -> dict[str, list]:
    """Per-row computed columns for the rollout dump, aligned with ``rows``.

    The trainer writes these into the batch's non-tensor columns right after
    computing them, so the dump stage picks them up generically alongside the
    agent loop's raw fields — no trainer-held state, no hand-curated list.
    """
    return {
        "vdct_reward": list(result.rewards),
        "advantage": list(result.advantages),
        "trainable": list(result.trainable),
        "skip_reason": list(result.skip_reasons),
        "q_ref_target": [result.q_ref_target.get(row.group_id) for row in rows],
    }


def build_rows(non_tensor_batch: dict) -> list[VDCTRow]:
    """Extract the VDCT row view from a verl batch's non-tensor columns.

    ``option_distribution`` / ``answer_index`` arrive as per-row object
    entries (lists or None) forwarded by the agent loop through
    ``extra_fields``.
    """
    missing = [key for key in REQUIRED_ROW_FIELDS if key not in non_tensor_batch]
    if missing:
        raise KeyError(
            f"batch is missing VDCT fields {missing}; the rows must come from the 'vdct' agent loop "
            f"(present keys: {sorted(non_tensor_batch)})"
        )
    n = len(non_tensor_batch["group_id"])

    def column(name):
        values = non_tensor_batch.get(name)
        return [None] * n if values is None else values

    dists = column("option_distribution")
    answer_indices = column("answer_index")
    return rows_from_records(
        [
            {
                "group_id": non_tensor_batch["group_id"][i],
                "variant": non_tensor_batch["variant"][i],
                "kind": non_tensor_batch["kind"][i],
                "parse_ok": non_tensor_batch["parse_ok"][i],
                "option_distribution": dists[i],
                "answer_index": answer_indices[i],
            }
            for i in range(n)
        ]
    )


def rows_from_records(records: list[dict]) -> list[VDCTRow]:
    """The VDCT row view of per-rollout dict records (rollout dumps, audit
    generations) — the record-shaped sibling of :func:`build_rows`."""
    rows = []
    for record in records:
        dist = record.get("option_distribution")
        answer_index = record.get("answer_index")
        rows.append(
            VDCTRow(
                group_id=str(record["group_id"]),
                variant=str(record["variant"]),
                kind=str(record["kind"]),
                parse_ok=bool(record["parse_ok"]),
                option_distribution=tuple(float(v) for v in dist) if dist is not None else None,
                answer_index=int(answer_index) if answer_index is not None else None,
            )
        )
    return rows


def merge_dump_fields(reward_extra_infos_dict: dict, vdct_fields: dict[str, list], n_rows: int) -> dict:
    """Merge per-row VDCT fields into the rollout-dump dict.

    verl's ``_write_generations`` silently drops any column whose length
    disagrees with the batch, so mismatching fields are rejected here instead
    of disappearing from the dump. Existing keys are never overwritten.
    """
    merged = dict(reward_extra_infos_dict)
    for key, values in vdct_fields.items():
        if len(values) != n_rows:
            raise ValueError(f"dump field {key!r} has {len(values)} entries for {n_rows} rows")
        if key not in merged:
            merged[key] = list(values)
    return merged
