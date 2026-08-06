"""Consistency reward — pure math, no backend coupling.

Canonical home of ``ConsistencyReward`` (previously defined inside
``cot_transparency.apis.tinker.rl_training``, which now re-exports from here).
"""

from .types import Rollout


class ConsistencyReward:
    """Consistency reward for training perturbation rollouts.

    Pushes p_hat toward p_ref (privileged) using variance-optimal baseline p_hat.
    r = -(p_hat - p_ref) * (trait - p_hat)

    Anchor reward is computed separately on reference perturbation rollouts.
    """

    def compute_rewards(
        self,
        rollouts: list[Rollout],
        p_hat: dict[int, float],
        p_ref: float,
        gaps: dict[int, float] | None = None,
        baseline: float | None = None,
    ) -> list[float]:
        """Consistency-only rewards for training perturbation rollouts.

        r = -gap[pert] * (trait - baseline)

        ``gaps`` lets the caller substitute a transformed gap (e.g. SNR-scaled)
        per perturbation. When None, the raw gap (p_hat[pert] - p_ref) is used.
        ``baseline`` overrides the per-perturbation mean p_hat[pert] used to centre the
        score term; the matched-pair estimator passes p_ref (the neutral control) so every
        cued rollout is judged against the reference, not its own per-cue mean. Any constant
        baseline leaves E[grad] = -gap * ∇p_hat unchanged (it only affects variance).
        """
        if gaps is None:
            gaps = {pert: rate - p_ref for pert, rate in p_hat.items()}
        if any(r.trait_value is None for r in rollouts):
            raise ValueError("consistency rewards require non-abstained rollouts")
        return [
            -gaps[r.perturbation_idx]
            * (float(r.trait_value) - (p_hat[r.perturbation_idx] if baseline is None else baseline))
            for r in rollouts
        ]

    def compute_anchor_rewards(
        self,
        ref_rollouts: list[Rollout],
        reference_rates: dict[int, float],
        initial_reference_rates: dict[int, float],
        gaps: dict[int, float] | None = None,
    ) -> list[float]:
        """Per-reference anchor rewards for reference perturbation rollouts.

        ``r[x,i] = -(p_x - p_x_initial) * (trait[x,i] - p_x)``.

        ``gaps`` lets the caller substitute a transformed gap per reference index.
        The caller must exclude rollouts whose current or initial reference rate is
        unavailable; mixing an aggregate reference rate into this term is incorrect.
        """
        if gaps is None:
            gaps = {
                idx: rate - initial_reference_rates[idx]
                for idx, rate in reference_rates.items()
                if idx in initial_reference_rates
            }
        if any(r.trait_value is None for r in ref_rollouts):
            raise ValueError("anchor rewards require non-abstained rollouts")
        return [
            -gaps[r.perturbation_idx] * (float(r.trait_value) - reference_rates[r.perturbation_idx])
            for r in ref_rollouts
        ]
