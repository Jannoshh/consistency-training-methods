"""RMCT custom advantage function for slime (``--custom-advantage-function-path``).

Two jobs, matching ctm's loop exactly:

1. Broadcast the per-sample RMCT advantage (computed by the parity-tested
   pipeline in the rollout function and shipped through ``sample.reward`` —
   the channel that survives slime's DP split) across the response tokens.
2. Apply the KL penalty with **tinker semantics**, ported from
   ``ctm/backends/local/engine.py::incorporate_kl_penalty`` (ctm @ 58c889c):

       diff_i     = (logp_sampled_i - logp_base_i) * mask_i          (per token)
       avg_diff   = sum_i(diff_i) / sum_i(mask_i)                    (batch scalar)
       A_i       += kl_coef * mask_i * (avg_diff - diff_i)

   The penalty is batch-centered (mean-zero), and uses the ROLLOUT sampling
   logprobs vs the frozen base model — so slime must run with ``--ref-load``
   pointing at the base checkpoint, ``--kl-coef 0`` and ``--use-kl-loss``
   UNSET (slime's own KL paths stay off; this function owns the term).
   ``kl_discount_factor`` (paper: 0.0) is not ported; nonzero values raise.

Runs on the last pipeline stage per DP rank, after slime computed
``log_probs``/``ref_log_probs``. The centering mean is all-reduced across
the data-parallel group, so the population is the whole optimizer-step
batch at any DP size — matching ctm's single-process batch mean exactly.

Configured through the same RMCT_CONFIG JSON as the rollout function.
"""

import logging

import torch

from .rmct_config import RMCTConfig

logger = logging.getLogger(__name__)

_CONFIG: list[RMCTConfig] = []


def _config() -> RMCTConfig:
    if not _CONFIG:
        _CONFIG.append(RMCTConfig.load())
    return _CONFIG[0]


def _dp_all_reduce_pair(total_diff, total_mask):
    """Sum (Σdiff, Σmask) across data-parallel ranks so the KL centering
    population is the whole optimizer-step batch (exact tinker semantics at
    any DP size). No-op outside distributed / at DP 1."""
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return total_diff, total_mask
    try:
        from megatron.core import parallel_state as mpu

        group = mpu.get_data_parallel_group(with_context_parallel=True)
    except Exception:
        group = None  # default group: fine at TP/PP 1
    if dist.get_world_size(group=group) == 1:
        return total_diff, total_mask
    pair = torch.stack([torch.as_tensor(total_diff).float(), torch.as_tensor(total_mask).float()]).cuda()
    dist.all_reduce(pair, group=group)
    return pair[0], pair[1]


def compute_advantages(args, rollout_data: dict) -> None:
    """Set ``rollout_data['advantages']`` / ``['returns']`` in place."""
    config = _config()
    rewards = rollout_data["rewards"]
    loss_masks = rollout_data["loss_masks"]

    # 1. Broadcast per-sample scalar advantage across response tokens.
    advantages = [
        torch.full_like(mask, float(reward), dtype=torch.float32) * mask for reward, mask in zip(rewards, loss_masks)
    ]

    # 2. Centered KL penalty vs the frozen base (tinker semantics).
    if config.kl_coef != 0.0:
        rollout_log_probs = rollout_data.get("rollout_log_probs")
        ref_log_probs = rollout_data.get("ref_log_probs")
        missing = [
            name
            for name, value in [("rollout_log_probs", rollout_log_probs), ("ref_log_probs", ref_log_probs)]
            if value is None
        ]
        if missing:
            raise RuntimeError(
                f"kl_coef != 0 but rollout_data lacks {missing}. ref_log_probs needs "
                "--ref-load AND (--use-kl-loss or slime kl_coef != 0) so the ref model loads; "
                "rollout_log_probs needs the rollout fn to set sample.rollout_log_probs."
            )
        diffs = [(sampled - ref) * mask for sampled, ref, mask in zip(rollout_log_probs, ref_log_probs, loss_masks)]
        total_mask = sum(mask.sum() for mask in loss_masks)
        total_diff = sum(diff.sum() for diff in diffs)
        # Center over the FULL step batch, not this DP rank's shard — ctm's
        # loop was single-process, so its mean spanned the whole batch.
        # All-reduce the (sum, count) pair across the DP group when sharded.
        total_diff, total_mask = _dp_all_reduce_pair(total_diff, total_mask)
        avg_diff = total_diff / torch.clamp(total_mask, min=1e-8)
        advantages = [
            adv + config.kl_coef * mask * (avg_diff - diff) for adv, mask, diff in zip(advantages, loss_masks, diffs)
        ]
        logger.info("rmct/kl_policy_base=%.6f (kl_coef=%.3f)", float(avg_diff), config.kl_coef)

    rollout_data["advantages"] = advantages
    rollout_data["returns"] = [adv.clone() for adv in advantages]
