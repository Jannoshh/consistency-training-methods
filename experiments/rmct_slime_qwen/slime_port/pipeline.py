"""Batch-level RMCT advantage construction, ported from
``ctm.training.rl.RLTrainer._build_training_batch`` (ctm @ 58c889c).

Pure math: rollout populations in, per-rollout rewards and advantages out. No
tinker datums, no logging, no backend. slime's custom rollout / reward
post-process path calls :func:`build_batch_item` per (datapoint, rollouts) and
:func:`compute_batch_advantages` per training batch, then attaches the
resulting per-sample advantage to each slime ``Sample``.

The KL penalty is NOT applied here — in the original loop
``backend.incorporate_kl_penalty`` mutates advantages after this stage, and the
logged reward/advantage (used by Gate A) are pre-KL. On the slime side the KL
term is configured through slime's own loss options and recorded as such.
"""

from dataclasses import dataclass, field
from typing import Literal

from . import advantages as adv_math
from .rewards import ConsistencyReward
from .types import Rollout


@dataclass
class BatchItem:
    """One datapoint's rollout populations, ready for reward computation.

    Mirrors ``ctm.core.types.BatchItem`` minus the logging-only fields.
    """

    datapoint_idx: int
    train_rollouts: list[Rollout]
    anchor_rollouts: list[Rollout]
    p_hat: dict[int, float]
    p_hat_counts: dict[int, int]
    p_ref: float | None
    reference_rates: dict[int, float]
    reference_rate_counts: dict[int, int]
    initial_reference_rates: dict[int, float]
    initial_reference_rate_counts: dict[int, int]
    n_ref_parsed: int


@dataclass
class AdvantageConfig:
    """The subset of ``RLConfig`` consumed by the advantage math."""

    advantage_estimator: Literal["grpo_normalized", "snr_scaling", "matched_pair"] = "grpo_normalized"
    normalization: Literal["pooled", "per_item"] = "per_item"
    anchor_weight: float = 0.0
    snr_mode: Literal["soft", "hard"] = "soft"
    snr_z: float = 2.0
    snr_normalizer: Literal["trait_std", "none"] = "trait_std"
    var_floor: float = 0.01
    pseudocount: float = 1.0
    reference_rate_n_rollouts: int = 128  # fallback initial-rate count (anchor SE)


@dataclass
class BatchAdvantages:
    """Rewards and advantages, ordered consistency-then-anchor like the original."""

    consistency_rewards: list[float]
    anchor_rewards: list[float]
    advantages: list[float]  # consistency advantages followed by anchor advantages
    grad_rollouts: list[Rollout]  # same order as ``advantages``
    has_signal: bool
    metrics: dict[str, float] = field(default_factory=dict)


def build_batch_item(
    datapoint_idx: int,
    rollouts: dict[int, list[Rollout]],
    reference_indices: list[int],
    training_indices: list[int],
    initial_reference_rates: dict[int, float] | None = None,
    initial_reference_rate_counts: dict[int, int] | None = None,
    n_gradient: int | None = None,
    reference_aggregation: Literal["mean", "min", "max"] = "mean",
    p_ref_init_fallback: float | None = None,
) -> BatchItem | None:
    """Group raw rollouts into a :class:`BatchItem`, following
    ``RLTrainer.collect_for_datapoint``: rates from parsed rollouts only,
    ``p_ref`` as the configured aggregate of reference rates, falling back to
    the initial reference aggregate when every reference rollout is unusable.
    Returns None when no reference rate is available at all (item dropped).
    """
    reference_rates_opt, reference_rate_counts = adv_math.compute_rates(rollouts, reference_indices)
    p_hat_opt, p_hat_counts = adv_math.compute_rates(rollouts, training_indices)
    p_ref = adv_math.aggregate_rates(reference_rates_opt, reference_indices, reference_aggregation)
    if p_ref is None:
        p_ref = p_ref_init_fallback
    if p_ref is None:
        return None
    p_hat = {idx: rate for idx, rate in p_hat_opt.items() if rate is not None}
    reference_rates = {idx: rate for idx, rate in reference_rates_opt.items() if rate is not None}
    return BatchItem(
        datapoint_idx=datapoint_idx,
        train_rollouts=adv_math.select_rollouts(
            {idx: rollouts.get(idx, []) for idx in training_indices if idx in p_hat},
            [idx for idx in training_indices if idx in p_hat],
            n_gradient,
        ),
        anchor_rollouts=adv_math.select_rollouts(rollouts, reference_indices, n_gradient),
        p_hat=p_hat,
        p_hat_counts=p_hat_counts,
        p_ref=p_ref,
        reference_rates=reference_rates,
        reference_rate_counts=reference_rate_counts,
        initial_reference_rates=dict(initial_reference_rates or {}),
        initial_reference_rate_counts=dict(initial_reference_rate_counts or {}),
        n_ref_parsed=sum(reference_rate_counts.values()),
    )


def compute_batch_advantages(batch_items: list[BatchItem], config: AdvantageConfig) -> BatchAdvantages:
    """Port of ``_build_training_batch``'s reward/advantage math.

    Ordering contract (needed for Gate A replay): consistency rollouts in
    ``batch_items`` order, each item's ``train_rollouts`` in order; then (when
    ``anchor_weight > 0``) anchor rollouts likewise.
    """
    reward_fn = ConsistencyReward()
    use_snr = config.advantage_estimator == "snr_scaling"
    use_matched = config.advantage_estimator == "matched_pair"

    consistency_rewards: list[float] = []
    consistency_div: list[float] = []
    consistency_snr: list[float] = []
    consistency_slices: list[tuple[int, int]] = []
    grad_rollouts: list[Rollout] = []
    raw_gap_abs, shrunk_gap_abs, gap_n = 0.0, 0.0, 0

    for item in batch_items:
        gaps = None
        baseline = None
        div_p = None
        snr_f: dict[int, float] = {}
        if use_snr:
            gaps = {}
            for pert, rate in item.p_hat.items():
                raw = rate - item.p_ref
                se = adv_math.gap_se(
                    rate, item.p_hat_counts.get(pert, 0), item.p_ref, item.n_ref_parsed, config.pseudocount
                )
                snr_f[pert] = adv_math.snr_shrink_factor(raw, se, config.snr_mode, config.snr_z)
                gaps[pert] = raw
                raw_gap_abs += abs(raw)
                shrunk_gap_abs += abs(raw * snr_f[pert])
                gap_n += 1
        elif use_matched and item.p_hat:
            n_pool = sum(item.p_hat_counts.get(p, 0) for p in item.p_hat)
            p_pool = (
                sum(rate * item.p_hat_counts.get(p, 0) for p, rate in item.p_hat.items()) / n_pool
                if n_pool > 0
                else sum(item.p_hat.values()) / len(item.p_hat)
            )
            raw = p_pool - item.p_ref
            se = adv_math.matched_pair_gap_se(
                item.p_hat, item.p_hat_counts, item.p_ref, item.n_ref_parsed, config.pseudocount
            )
            g = adv_math.snr_scale_gap(raw, se, config.snr_mode, config.snr_z)
            gaps = {pert: g for pert in item.p_hat}
            baseline = item.p_ref
            div_p = p_pool
            raw_gap_abs += abs(raw)
            shrunk_gap_abs += abs(g)
            gap_n += 1
        rewards = reward_fn.compute_rewards(item.train_rollouts, item.p_hat, item.p_ref, gaps=gaps, baseline=baseline)
        slice_start = len(consistency_rewards)
        consistency_rewards.extend(rewards)
        consistency_slices.append((slice_start, len(consistency_rewards)))
        for rollout in item.train_rollouts:
            grad_rollouts.append(rollout)
            p_for_div = div_p if div_p is not None else item.p_hat.get(rollout.perturbation_idx, 0.0)
            consistency_div.append(adv_math.trait_std(p_for_div, config.var_floor))
            consistency_snr.append(snr_f.get(rollout.perturbation_idx, 1.0))

    anchor_rewards: list[float] = []
    anchor_div: list[float] = []
    anchor_snr: list[float] = []
    anchor_slices: list[tuple[int, int]] = []
    anchor_grad_rollouts: list[Rollout] = []
    if config.anchor_weight > 0:
        for item in batch_items:
            valid_indices = set(item.reference_rates) & set(item.initial_reference_rates)
            valid_rollouts = [r for r in item.anchor_rollouts if r.perturbation_idx in valid_indices]
            anchor_gaps: dict[int, float] | None = None
            anchor_factors: dict[int, float] = {}
            if use_snr or use_matched:
                anchor_gaps = {}
                for idx in valid_indices:
                    current = item.reference_rates[idx]
                    initial = item.initial_reference_rates[idx]
                    raw = current - initial
                    n_current = item.reference_rate_counts.get(idx, 0)
                    n_initial = item.initial_reference_rate_counts.get(idx, config.reference_rate_n_rollouts)
                    se = adv_math.gap_se(current, n_current, initial, n_initial, config.pseudocount)
                    if use_snr:
                        anchor_gaps[idx] = raw
                        anchor_factors[idx] = adv_math.snr_shrink_factor(raw, se, config.snr_mode, config.snr_z)
                    else:
                        anchor_gaps[idx] = adv_math.snr_scale_gap(raw, se, config.snr_mode, config.snr_z)
            rewards = reward_fn.compute_anchor_rewards(
                valid_rollouts, item.reference_rates, item.initial_reference_rates, gaps=anchor_gaps
            )
            a_start = len(anchor_rewards)
            anchor_rewards.extend(rewards)
            anchor_slices.append((a_start, len(anchor_rewards)))
            for rollout in valid_rollouts:
                anchor_grad_rollouts.append(rollout)
                anchor_div.append(adv_math.trait_std(item.reference_rates[rollout.perturbation_idx], config.var_floor))
                anchor_snr.append(anchor_factors.get(rollout.perturbation_idx, 1.0))

    if use_matched:
        if config.snr_normalizer == "none":
            consistency_adv = list(consistency_rewards)
            anchor_adv = list(anchor_rewards)
        else:
            consistency_adv = [r / d for r, d in zip(consistency_rewards, consistency_div)]
            anchor_adv = [r / d for r, d in zip(anchor_rewards, anchor_div)]
    else:
        consistency_adv = adv_math.normalize_grouped(consistency_rewards, consistency_slices, config.normalization)
        anchor_adv = adv_math.normalize_grouped(anchor_rewards, anchor_slices, config.normalization)
        if use_snr:
            consistency_adv = [f * a for f, a in zip(consistency_snr, consistency_adv)]
            anchor_adv = [f * a for f, a in zip(anchor_snr, anchor_adv)]
    consistency_adv = [a * (1 - config.anchor_weight) for a in consistency_adv]
    anchor_adv = [a * config.anchor_weight for a in anchor_adv]

    advantages = consistency_adv + anchor_adv
    has_signal = bool(advantages) and any(abs(a) >= 1e-8 for a in advantages)
    metrics = (
        {
            "train/gap_raw_abs_mean": raw_gap_abs / gap_n,
            "train/gap_snr_scaled_abs_mean": shrunk_gap_abs / gap_n,
            "train/gap_snr_scale_factor": (shrunk_gap_abs / raw_gap_abs) if raw_gap_abs > 1e-9 else 0.0,
        }
        if (use_snr or use_matched) and gap_n > 0
        else {}
    )
    return BatchAdvantages(
        consistency_rewards=consistency_rewards,
        anchor_rewards=anchor_rewards,
        advantages=advantages,
        grad_rollouts=grad_rollouts + anchor_grad_rollouts,
        has_signal=has_signal,
        metrics=metrics,
    )
