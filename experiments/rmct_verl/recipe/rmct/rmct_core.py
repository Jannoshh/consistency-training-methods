"""Pure RMCT math for the verl port — no Ray, no torch.distributed, no verl imports.

Everything here is testable on CPU in-process, which is what makes Gate A
(``tests/test_rmct_verl_math.py``) possible for this port.

Two responsibilities:

1. :func:`compute_row_advantages` — turn a flat list of verl batch rows (each
   row = one rollout, tagged with ``group_id`` / ``variant`` / ``parse_ok`` /
   ``trait``) into a per-row advantage plus a trainability flag. The rate and
   advantage arithmetic itself is NOT reimplemented here: rows are packed into
   the slime port's ``Rollout`` objects and handed to
   ``slime_port.pipeline.build_batch_item`` / ``compute_batch_advantages``,
   the Gate-A-tested code. This module only owns the row→rollout→row mapping.
2. :func:`centered_kl_penalty` — the tinker-semantics batch-centered KL term,
   ported from ``slime_port/rmct_advantage.py`` (which in turn ports
   ``ctm/backends/local/engine.py::incorporate_kl_penalty``). verl's own
   ``kl_penalty`` clamps the log-ratio to ±20 and is deliberately NOT used.

Locating the slime port
-----------------------
The parity rule says reuse, never reimplement, so this module imports the
sibling experiment directory ``experiments/rmct_slime_qwen`` by path. Two
knobs, in order: the ``RMCT_SLIME_PORT_DIR`` environment variable (set this on
a pod where only ``experiments/rmct_verl`` was synced), else the repo-relative
sibling path. A file-relative ``sys.path`` insert was chosen over vendoring
because a vendored copy could silently drift from the parity-tested original,
and over packaging because neither experiment directory is installable.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path


def _slime_port_parent() -> Path:
    override = os.environ.get("RMCT_SLIME_PORT_DIR")
    if override:
        parent = Path(override).resolve()
        if parent.name == "slime_port":
            parent = parent.parent
        return parent
    # .../experiments/rmct_verl/recipe/rmct/rmct_core.py -> .../experiments
    return Path(__file__).resolve().parents[3] / "rmct_slime_qwen"


_PARENT = _slime_port_parent()
if not (_PARENT / "slime_port" / "pipeline.py").exists():
    raise ImportError(
        f"slime_port not found under {_PARENT}. Set RMCT_SLIME_PORT_DIR to the directory "
        "containing the slime_port package (experiments/rmct_slime_qwen)."
    )
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from slime_port.pipeline import (
    AdvantageConfig,
    build_batch_item,
    compute_batch_advantages,
)
from slime_port.types import Rollout

__all__ = [
    "REFERENCE_VARIANT",
    "TRAINING_VARIANT",
    "AdvantageConfig",
    "RMCTBatchResult",
    "RMCTRow",
    "centered_kl_penalty",
    "compute_row_advantages",
]

REFERENCE_VARIANT = "reference"
TRAINING_VARIANT = "training"

# Perturbation indices used inside the slime port's math. RMCT on a paired
# prompt has exactly two: 0 = the neutral/reference prompt whose trait rate is
# p_ref, 1 = the cued/training prompt whose trait rate is p_hat.
_REFERENCE_PERTURBATION = 0
_TRAINING_PERTURBATION = 1


@dataclass(frozen=True)
class RMCTRow:
    """One verl batch row, reduced to what the RMCT math reads.

    ``trait`` is the trait indicator (mcq-bias ``matches_bias``) and is only
    meaningful when ``parse_ok``; unparsed rows carry 0.0 and are excluded from
    the rate denominators, exactly as in ``slime_port.rmct_rollout._to_rollout``
    (``parsed_successfully = answer_parsed and not truncated``).
    """

    group_id: str
    variant: str
    parse_ok: bool
    trait: float = 0.0


@dataclass
class RMCTBatchResult:
    """Per-row outputs, aligned index-for-index with the input row list."""

    advantages: list[float]
    trainable: list[bool]
    skip_reasons: list[str | None]
    has_signal: bool
    metrics: dict[str, float] = field(default_factory=dict)
    # Per-group diagnostics, keyed by group_id (None when the group was dropped).
    p_ref: dict[str, float | None] = field(default_factory=dict)
    p_hat: dict[str, float | None] = field(default_factory=dict)


def compute_row_advantages(
    rows: list[RMCTRow],
    config: AdvantageConfig | None = None,
) -> RMCTBatchResult:
    """RMCT advantage per row, via the slime port's parity-tested pipeline.

    Row semantics (identical to the slime rollout function's Sample semantics):

    - reference rows never carry gradient — they only supply ``p_ref``;
    - a training row carries gradient only when it parsed AND the batch has
      signal (``has_signal``);
    - a group whose reference rows all failed to parse yields no ``p_ref`` and
      is dropped whole (``build_batch_item`` returns None).

    ``anchor_weight > 0`` is rejected: the anchor term needs the one-time
    initial-reference-rate measurement, which this port does not carry (same
    restriction as ``slime_port.rmct_rollout``).
    """
    config = config or AdvantageConfig()
    if config.anchor_weight > 0:
        raise NotImplementedError("anchor_weight > 0 needs the base-rate measurement port")

    n = len(rows)
    advantages = [0.0] * n
    trainable = [False] * n
    skip_reasons: list[str | None] = [None] * n

    # Group rows in first-appearance order so the batch-item order (and hence
    # any pooled normalization) is deterministic and independent of dict order.
    group_order: list[str] = []
    rollouts_by_group: dict[str, dict[int, list[Rollout]]] = {}
    row_of_rollout: dict[int, int] = {}

    for row_idx, row in enumerate(rows):
        if row.variant == REFERENCE_VARIANT:
            pert = _REFERENCE_PERTURBATION
        elif row.variant == TRAINING_VARIANT:
            pert = _TRAINING_PERTURBATION
        else:
            raise ValueError(f"row {row_idx}: unknown variant {row.variant!r}")
        if row.group_id not in rollouts_by_group:
            group_order.append(row.group_id)
            rollouts_by_group[row.group_id] = {}
        rollout = Rollout(
            tokens=[],
            logprobs=[],
            text="",
            trait_value=float(row.trait),
            perturbation_idx=pert,
            parsed_successfully=bool(row.parse_ok),
            answer_parsed=bool(row.parse_ok),
            has_logprobs=True,
        )
        rollouts_by_group[row.group_id].setdefault(pert, []).append(rollout)
        row_of_rollout[id(rollout)] = row_idx
        if row.variant == REFERENCE_VARIANT:
            skip_reasons[row_idx] = "reference_row"
        elif not row.parse_ok:
            skip_reasons[row_idx] = "answer_parse_failure"

    batch_items = []
    p_ref: dict[str, float | None] = {}
    p_hat: dict[str, float | None] = {}
    for group_idx, group_id in enumerate(group_order):
        # n_gradient=None keeps every parsed rollout, in row order — no random
        # subsampling, so the row mapping below is exact.
        item = build_batch_item(
            datapoint_idx=group_idx,
            rollouts=rollouts_by_group[group_id],
            reference_indices=[_REFERENCE_PERTURBATION],
            training_indices=[_TRAINING_PERTURBATION],
            n_gradient=None,
        )
        p_ref[group_id] = item.p_ref if item is not None else None
        p_hat[group_id] = item.p_hat.get(_TRAINING_PERTURBATION) if item is not None else None
        if item is None:
            for pert_rollouts in rollouts_by_group[group_id].values():
                for rollout in pert_rollouts:
                    row_idx = row_of_rollout[id(rollout)]
                    if rows[row_idx].variant == TRAINING_VARIANT:
                        skip_reasons[row_idx] = "no_reference_rate"
        else:
            batch_items.append(item)

    result = compute_batch_advantages(batch_items, config)

    for rollout, advantage in zip(result.grad_rollouts, result.advantages):
        row_idx = row_of_rollout[id(rollout)]
        advantages[row_idx] = float(advantage)
        if result.has_signal:
            trainable[row_idx] = True
            skip_reasons[row_idx] = None
        else:
            advantages[row_idx] = 0.0
            skip_reasons[row_idx] = "zero_advantage_batch"

    n_parsed = sum(1 for row in rows if row.parse_ok)
    n_train_rows = sum(1 for row in rows if row.variant == TRAINING_VARIANT)
    parsed_p_ref = [v for v in p_ref.values() if v is not None]
    parsed_p_hat = [v for v in p_hat.values() if v is not None]
    metrics = {
        "rmct/parse_rate": n_parsed / max(n, 1),
        "rmct/p_ref_mean": sum(parsed_p_ref) / max(len(parsed_p_ref), 1),
        "rmct/p_hat_mean": sum(parsed_p_hat) / max(len(parsed_p_hat), 1),
        "rmct/gap_mean": (
            sum(p_hat[g] - p_ref[g] for g in group_order if p_hat[g] is not None and p_ref[g] is not None)
            / max(sum(1 for g in group_order if p_hat[g] is not None and p_ref[g] is not None), 1)
        ),
        "rmct/has_signal": float(result.has_signal),
        "rmct/n_groups": float(len(group_order)),
        "rmct/n_groups_dropped": float(len(group_order) - len(batch_items)),
        "rmct/n_grad_rows": float(sum(trainable)),
        "rmct/grad_row_frac": sum(trainable) / max(n_train_rows, 1),
    }
    metrics.update(result.metrics)

    return RMCTBatchResult(
        advantages=advantages,
        trainable=trainable,
        skip_reasons=skip_reasons,
        has_signal=result.has_signal,
        metrics=metrics,
        p_ref=p_ref,
        p_hat=p_hat,
    )


def centered_kl_penalty(policy_log_probs, ref_log_probs, response_mask, kl_coef: float):
    """Batch-centered KL term, tinker semantics. Returns ``(penalty, avg_diff)``.

        diff_i   = (logp_policy_i - logp_ref_i) * mask_i
        avg_diff = sum_i(diff_i) / sum_i(mask_i)
        penalty_i = kl_coef * mask_i * (avg_diff - diff_i)

    The caller ADDS ``penalty`` to the advantages. The term is mean-zero over
    the masked population by construction, so it reweights within the batch
    rather than shifting it.

    Deliberately not ``verl.trainer.ppo.core_algos.kl_penalty``: that clamps
    the log-ratio to ±20 and offers only the k1/k2/k3/low_var estimators, none
    of which is this centered form.

    torch tensors in, torch tensor out; no distributed collectives. In the
    single-controller trainer the batch handed to ``_update_actor`` is the
    WHOLE optimizer-step batch before DP dispatch, so a plain sum over it is
    already the global population — the ``_dp_all_reduce_pair`` dance the Miles
    port needed (it ran inside each DP rank) has no counterpart here.
    """
    mask = response_mask.to(policy_log_probs.dtype)
    diff = (policy_log_probs - ref_log_probs) * mask
    total_mask = mask.sum()
    avg_diff = diff.sum() / total_mask.clamp(min=1e-8)
    return kl_coef * mask * (avg_diff - diff), avg_diff
