"""RMCT trainer for verl: a ``RayPPOTrainer`` with the advantage stage replaced.

Everything upstream of ``_update_actor`` is stock verl — rollout, old_log_prob,
ref_log_prob, and even verl's own (ignored) GRPO advantage pass. This subclass
does two things:

1. forces ``use_reference_policy = True``. RMCT keeps verl's KL machinery off
   (``algorithm.use_kl_in_reward=False``, ``actor.use_kl_loss=False``) because
   it owns the KL term itself, but ``need_reference_policy`` derives the ref
   worker's existence from exactly those two flags — without the override the
   ref forward never runs and ``ref_log_prob`` is absent. Under LoRA
   (``lora_rank > 0``) verl sets ``ref_in_actor``, so the "reference" is the
   same worker with the adapter disabled: the frozen base model, which is
   precisely RMCT's KL target.

2. overrides ``_update_actor`` to write RMCT's ``advantages`` and
   ``response_mask`` before delegating to super().

Grouping note: verl assigns one ``uid`` per DATASET row and then repeats each
row ``rollout.n`` times, so a uid identifies (datapoint, variant) — the two
variants of one datapoint get different uids. RMCT needs the pair, so grouping
is by the dataset's ``group_id`` column, with ``variant`` selecting the
subgroup. ``uid`` is not used.

DP awareness: this method runs in the single-controller driver, and ``batch``
here is the entire optimizer-step batch before any dispatch to workers. The
KL centering mean and the advantage normalization therefore already span the
global population; the ``_dp_all_reduce_pair`` the Miles port needed (it ran
inside each DP rank's loss function) has no counterpart here. ``balance_batch``
may reorder rows before this point, which is harmless: every computation here
is either per-row or a group/batch reduction.
"""

from __future__ import annotations

import logging

import torch
from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

from .rmct_core import AdvantageConfig, RMCTRow, centered_kl_penalty, compute_row_advantages

logger = logging.getLogger(__name__)

_REQUIRED_ROW_FIELDS = ("group_id", "variant", "parse_ok", "trait")


def _advantage_config(rmct_cfg) -> AdvantageConfig:
    """Build the slime port's ``AdvantageConfig`` from the hydra ``rmct`` block."""
    return AdvantageConfig(
        advantage_estimator=rmct_cfg.get("advantage_estimator", "grpo_normalized"),
        normalization=rmct_cfg.get("normalization", "per_item"),
        anchor_weight=rmct_cfg.get("anchor_weight", 0.0),
        snr_mode=rmct_cfg.get("snr_mode", "soft"),
        snr_z=rmct_cfg.get("snr_z", 2.0),
        snr_normalizer=rmct_cfg.get("snr_normalizer", "trait_std"),
        var_floor=rmct_cfg.get("var_floor", 0.01),
        pseudocount=rmct_cfg.get("pseudocount", 1.0),
    )


def build_rows(non_tensor_batch: dict) -> list[RMCTRow]:
    """Extract the RMCT row view from a verl batch's non-tensor columns."""
    missing = [key for key in _REQUIRED_ROW_FIELDS if key not in non_tensor_batch]
    if missing:
        raise KeyError(
            f"batch is missing RMCT fields {missing}; the rows must come from the 'rmct' agent loop "
            f"(present keys: {sorted(non_tensor_batch)})"
        )
    group_ids = non_tensor_batch["group_id"]
    variants = non_tensor_batch["variant"]
    parse_oks = non_tensor_batch["parse_ok"]
    traits = non_tensor_batch["trait"]
    return [
        RMCTRow(
            group_id=str(group_ids[i]),
            variant=str(variants[i]),
            parse_ok=bool(parse_oks[i]),
            trait=float(traits[i]),
        )
        for i in range(len(group_ids))
    ]


class RayRMCTTrainer(RayPPOTrainer):
    """RayPPOTrainer with the RMCT advantage/KL/masking stage."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Must be set before init_workers(); see the module docstring.
        self.use_reference_policy = True
        self.rmct_config = self.config.rmct
        self.rmct_advantage_config = _advantage_config(self.rmct_config)
        kl_source = self.rmct_config.get("kl_logprob_source", "old_log_probs")
        if kl_source not in ("old_log_probs", "rollout_log_probs"):
            raise ValueError(f"rmct.kl_logprob_source must be old_log_probs or rollout_log_probs, got {kl_source!r}")
        self.rmct_kl_source = kl_source
        if self.config.algorithm.get("use_kl_in_reward", False) or self.config.actor_rollout_ref.actor.use_kl_loss:
            raise ValueError(
                "RMCT owns the KL term; set algorithm.use_kl_in_reward=False and "
                "actor_rollout_ref.actor.use_kl_loss=False (rmct.kl_coef controls the RMCT KL)."
            )

    def _update_actor(self, batch: DataProto) -> DataProto:
        rows = build_rows(batch.non_tensor_batch)
        result = compute_row_advantages(rows, self.rmct_advantage_config)

        device = batch.batch["response_mask"].device
        response_mask = batch.batch["response_mask"]
        trainable = torch.tensor(result.trainable, dtype=response_mask.dtype, device=device).unsqueeze(-1)
        # Zero the mask for every reference row, every unparsed training row and
        # (when the batch has no signal) everything: those rows must contribute
        # neither policy gradient nor KL, and a zeroed row also drops out of the
        # DP-reduced token-count denominator.
        response_mask = response_mask * trainable
        batch.batch["response_mask"] = response_mask

        n_masked_tokens = int(response_mask.sum().item())
        metrics = {k: float(v) for k, v in result.metrics.items()}
        metrics["rmct/masked_token_count"] = float(n_masked_tokens)
        for reason in result.skip_reasons:
            if reason is not None:
                key = f"rmct/skip/{reason}"
                metrics[key] = metrics.get(key, 0.0) + 1.0

        if n_masked_tokens == 0:
            # token-mean loss aggregation divides by the masked token count, so an
            # entirely masked batch is not merely a no-op — it is a division by
            # zero. Skip the actor update for this step, as ctm's loop does when
            # a batch carries no signal.
            logger.warning(
                "RMCT: no trainable tokens at step %s (has_signal=%s); skipping the actor update",
                getattr(self, "global_steps", "?"),
                result.has_signal,
            )
            metrics["rmct/skipped_update"] = 1.0
            return DataProto.from_single_dict(data={}, meta_info={"metrics": metrics})
        metrics["rmct/skipped_update"] = 0.0

        row_advantages = torch.tensor(result.advantages, dtype=torch.float32, device=device).unsqueeze(-1)
        advantages = row_advantages * response_mask.to(torch.float32)

        kl_coef = float(self.rmct_config.get("kl_coef", 0.0))
        if kl_coef != 0.0:
            if self.rmct_kl_source not in batch.batch:
                raise RuntimeError(
                    f"rmct.kl_coef != 0 but '{self.rmct_kl_source}' is not in the batch "
                    "(rollout_log_probs needs actor_rollout_ref.rollout.calculate_log_probs=True)"
                )
            if "ref_log_prob" not in batch.batch:
                raise RuntimeError(
                    "rmct.kl_coef != 0 but 'ref_log_prob' is not in the batch; the reference "
                    "forward did not run (use_reference_policy)."
                )
            penalty, avg_diff = centered_kl_penalty(
                batch.batch[self.rmct_kl_source].to(torch.float32),
                batch.batch["ref_log_prob"].to(torch.float32),
                response_mask,
                kl_coef,
            )
            advantages = advantages + penalty
            metrics["rmct/kl_policy_base"] = float(avg_diff.item())
        metrics["rmct/kl_coef"] = kl_coef

        batch.batch["advantages"] = advantages
        metrics["rmct/advantage_abs_mean"] = float((advantages.abs().sum() / max(n_masked_tokens, 1)).item())

        actor_output = super()._update_actor(batch)
        actor_output.meta_info.setdefault("metrics", {}).update(metrics)
        return actor_output
