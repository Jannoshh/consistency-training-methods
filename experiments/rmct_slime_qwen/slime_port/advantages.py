"""Advantage construction and rollout statistics — pure math, no backend coupling.

Extracted verbatim from ``RLTrainer``'s (static) methods in
``cot_transparency.apis.tinker.rl_training``; the trainer now delegates here.
Config-dependent behaviour (SNR mode/z, pooled vs per-item normalization,
reference aggregation) is passed explicitly so every function is pure.
"""

import random
from typing import Literal

from .types import Rollout

# ── Index / rollout selection ────────────────────────────────────────────────


def resolve_indices(indices: list[int] | str, n_total: int) -> list[int]:
    if indices == "all":
        return list(range(n_total))
    return list(indices)


def select_rollouts(rollouts: dict[int, list[Rollout]], indices: list[int], n_gradient: int | None) -> list[Rollout]:
    """Select up to n_gradient parsed rollouts per perturbation index."""
    result = []
    for idx in indices:
        parsed = [r for r in rollouts.get(idx, []) if r.parsed_successfully]
        if n_gradient is None or n_gradient >= len(parsed):
            result.extend(parsed)
        else:
            result.extend(random.sample(parsed, n_gradient))
    return result


# ── Rates ────────────────────────────────────────────────────────────────────


def compute_rates(
    rollouts: dict[int, list[Rollout]], indices: list[int]
) -> tuple[dict[int, float | None], dict[int, int]]:
    """Compute trait rates from parsed rollouts only."""
    rates: dict[int, float | None] = {}
    counts = {}
    for idx in indices:
        parsed = [r for r in rollouts.get(idx, []) if r.parsed_successfully and r.trait_value is not None]
        counts[idx] = len(parsed)
        rates[idx] = sum(float(r.trait_value) for r in parsed) / len(parsed) if parsed else None
    return rates, counts


def aggregate_rates(
    rates: dict[int, float | None], indices: list[int], aggregation: Literal["mean", "min", "max"] = "mean"
) -> float | None:
    """Aggregate reference perturbation rates into a single scalar."""
    valid: list[float] = [rates[i] for i in indices if i in rates and rates[i] is not None]  # type: ignore[misc]
    if not valid:
        return None
    if aggregation == "mean":
        return sum(valid) / len(valid)
    elif aggregation == "min":
        return min(valid)
    elif aggregation == "max":
        return max(valid)
    raise ValueError(f"Unknown aggregation: {aggregation}")


# ── Sampling-noise statistics ────────────────────────────────────────────────


def trait_std(p: float, var_floor: float = 0.01) -> float:
    """Per-rollout Bernoulli std, used to standardize trait noise (NOT the gap).

    Dividing the SNR-scaling reward by this — rather than by the full reward std —
    normalizes only the per-rollout noise and leaves the gap magnitude intact, so
    |advantage| ~ |SNR-scaled gap| (comparable scale to grpo's unit advantages).
    """
    return (p * (1.0 - p) + var_floor) ** 0.5


def binom_var(p: float, n: int, pseudocount: float = 1.0) -> float:
    """Laplace/Beta(pseudocount, pseudocount)-smoothed binomial variance p(1-p)/n.

    Smoothing keeps the boundary (p=0 or p=1) from looking like zero uncertainty: the
    raw normal-approx variance collapses to 0 at an observed rate of exactly 0, which would make
    a tiny empirical gap look infinitely significant. The rule of three says 0/n carries
    uncertainty ~3/n, not 0. Interior proportions are ~unchanged.
    """
    n_eff = max(n, 1)
    n_smooth = n_eff + 2.0 * pseudocount
    p_smooth = (p * n_eff + pseudocount) / n_smooth  # shrink toward 0.5
    return p_smooth * (1.0 - p_smooth) / n_smooth


def gap_se(p1: float, n1: int, p2: float, n2: int, pseudocount: float = 1.0) -> float:
    """Standard error of the gap (p1 - p2) under independent binomial sampling."""
    return (binom_var(p1, n1, pseudocount) + binom_var(p2, n2, pseudocount)) ** 0.5


def matched_pair_gap_se(
    p_hat: dict[int, float], p_hat_counts: dict[int, int], p_ref: float, n_ref: int, pseudocount: float = 1.0
) -> float:
    """SE of (p_pool - p_ref), where p_pool is the count-weighted mean of the per-cue
    rates across the cue family.

    The cue family is a FIXED set of heterogeneous paraphrases, so the cued side's
    sampling variance is the STRATIFIED (cluster) variance ``sum_c (n_c/n_pool)^2 ·
    Var(p_c)`` — NOT a single pooled binomial ``p_pool(1-p_pool)/n_pool``. The pooled
    binomial counts the genuine between-cue spread as if it were sampling noise (mixture
    variance ≥ mean within-cue variance), inflating the SE and over-shrinking real gaps
    — directly contradicting the "cue diversity supplies the low-variance gap" design.
    """
    n_pool = sum(p_hat_counts.get(c, 0) for c in p_hat)
    if n_pool > 0:
        cued_var = sum(
            (p_hat_counts.get(c, 0) / n_pool) ** 2 * binom_var(rate, p_hat_counts.get(c, 0), pseudocount)
            for c, rate in p_hat.items()
        )
    else:
        # No parsed counts: treat each cue as one observation of the cue-mean.
        k = max(len(p_hat), 1)
        cued_var = sum(binom_var(rate, 1, pseudocount) for rate in p_hat.values()) / (k * k)
    return (cued_var + binom_var(p_ref, n_ref, pseudocount)) ** 0.5


# ── SNR gating ───────────────────────────────────────────────────────────────


def snr_scale_gap(d: float, se: float, mode: Literal["soft", "hard"] = "soft", z: float = 2.0) -> float:
    """Scale an empirical gap toward 0 by its sampling SNR = (d/se)^2.

    soft: d * snr / (snr + z^2)     — smooth, half-weight at |d| = z·SE
    hard: d * max(0, 1 - z^2/snr)   — positive-part gate, zero below |d| = z·SE

    Larger sampling budget → smaller SE → less SNR scaling (you trust smaller gaps).
    """
    if d == 0.0:
        return 0.0
    if se <= 0.0:
        return d
    snr = (d * d) / (se * se)
    z2 = z**2
    if mode == "hard":
        factor = max(0.0, 1.0 - z2 / snr)
    else:
        factor = snr / (snr + z2)
    return d * factor


def snr_shrink_factor(d: float, se: float, mode: Literal["soft", "hard"] = "soft", z: float = 2.0) -> float:
    """The SNR gate in [0,1] such that snr_scale_gap(d, se) == d * factor.

    snr_scaling multiplies the unit-variance GRPO advantage by this per-item factor:
    ~1 when the gap clears sampling noise (full GRPO step), tapering to 0 within it
    (anti-overshoot). z=0 => factor 1 (faithful GRPO).
    """
    if d == 0.0:
        return 0.0
    if se <= 0.0:
        return 1.0
    snr = (d * d) / (se * se)
    z2 = z**2
    if mode == "hard":
        return max(0.0, 1.0 - z2 / snr)
    return snr / (snr + z2)


# ── Advantage normalization ──────────────────────────────────────────────────


def normalize_advantages(rewards: list[float]) -> list[float]:
    if not rewards:
        return rewards
    mean_r = sum(rewards) / len(rewards)
    var = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
    std_r = var**0.5
    if std_r < 1e-8:
        return [0.0] * len(rewards)
    return [(r - mean_r) / std_r for r in rewards]


def normalize_grouped(
    rewards: list[float], slices: list[tuple[int, int]], mode: Literal["pooled", "per_item"] = "per_item"
) -> list[float]:
    """Unit-variance standardization, pooled over the whole batch or per-item (per group).

    per_item: each item's rollouts are standardized on their own, so the per-item gap (a
    constant within the group) cancels — leaving sign(gap)·standardized(trait−p_hat); the SNR
    gate is then the sole magnitude signal and no single big-gap item dominates the batch.
    An item with zero within-group variance (e.g. all rollouts refuse) correctly gets 0.
    pooled: standardize the whole list together → gap magnitude becomes cross-item weighting.
    """
    if mode != "per_item":
        return normalize_advantages(rewards)
    adv = list(rewards)
    for s, e in slices:
        adv[s:e] = normalize_advantages(rewards[s:e])
    return adv
