"""VDCT trainer for verl: a ``RayPPOTrainer`` with the advantage stage replaced.

Follows ``recipe.rmct.rmct_trainer`` exactly in structure; see that module's
docstring for the verl mechanics (why ``use_reference_policy`` is forced, DP
awareness, why grouping uses the dataset ``group_id`` rather than verl's per
-row uid). What differs:

- rewards/advantages come from ``vdct_core.compute_row_advantages`` (graded
  JS + log-score rewards over distribution rollouts, per-group
  standardization) instead of the RMCT rate-gap pipeline;
- answer-kind rows are zero-masked alongside anything untrainable — they are
  outcome samples only;
- ``_log_rollout_data`` is overridden to thread the VDCT per-row fields
  (parsed distributions, answers, rewards, advantages, targets) into
  ``reward_extra_infos_dict``, fixing for this recipe the inherited RMCT gap
  where rollout dumps carried only verl's standard fields. In verl 2b0fe51's
  ``fit()`` the dump runs after ``_update_actor`` on the same batch object,
  so the computed per-row results stashed there are aligned and current.

The batch-centered KL term (tinker semantics) is reused from the RMCT recipe
via ``vdct_core.centered_kl_penalty``.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

from .vdct_core import VDCTConfig, build_rows, centered_kl_penalty, compute_row_advantages, merge_dump_fields

logger = logging.getLogger(__name__)

# Agent-loop extra_fields copied verbatim from the batch into the rollout dump.
_DUMP_BATCH_FIELDS = (
    "group_id",
    "variant",
    "kind",
    "parse_ok",
    "option_distribution",
    "answer_option",
    "answer_index",
    "trait",
    "option_labels",
    "biased_option",
    "response_truncated",
)


def _vdct_config(vdct_cfg) -> VDCTConfig:
    """Build the math config from the hydra ``vdct`` block."""
    return VDCTConfig(
        lambda_log_score=float(vdct_cfg.get("lambda_log_score", 1.0)),
        epsilon=float(vdct_cfg.get("epsilon", 1e-3)),
        normalization=vdct_cfg.get("normalization", "per_item"),
    )


class RayVDCTTrainer(RayPPOTrainer):
    """RayPPOTrainer with the VDCT advantage/KL/masking stage."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Must be set before init_workers(); the KL term targets the frozen
        # base model even though both of verl's own KL switches are off.
        self.use_reference_policy = True
        self.vdct_config = self.config.vdct
        self.vdct_math_config = _vdct_config(self.vdct_config)
        kl_source = self.vdct_config.get("kl_logprob_source", "old_log_probs")
        if kl_source not in ("old_log_probs", "rollout_log_probs"):
            raise ValueError(f"vdct.kl_logprob_source must be old_log_probs or rollout_log_probs, got {kl_source!r}")
        self.vdct_kl_source = kl_source
        if self.config.algorithm.get("use_kl_in_reward", False) or self.config.actor_rollout_ref.actor.use_kl_loss:
            raise ValueError(
                "VDCT owns the KL term; set algorithm.use_kl_in_reward=False and "
                "actor_rollout_ref.actor.use_kl_loss=False (vdct.kl_coef controls the VDCT KL)."
            )
        # Per-row results of the latest _update_actor, consumed by the
        # rollout-dump override.
        self._vdct_dump_fields: dict[str, list] | None = None

    def _update_actor(self, batch: DataProto) -> DataProto:
        rows = build_rows(batch.non_tensor_batch)
        result = compute_row_advantages(rows, self.vdct_math_config)

        self._vdct_dump_fields = {
            "vdct_reward": [float(r) for r in result.rewards],
            "advantage": [float(a) for a in result.advantages],
            "trainable": [bool(t) for t in result.trainable],
            "skip_reason": list(result.skip_reasons),
            "q_ref_target": [result.q_ref_target.get(row.group_id) for row in rows],
        }

        device = batch.batch["response_mask"].device
        response_mask = batch.batch["response_mask"]
        trainable = torch.tensor(result.trainable, dtype=response_mask.dtype, device=device).unsqueeze(-1)
        # Zero the mask for every answer row, and (when the batch has no
        # signal) everything: those rows must contribute neither policy
        # gradient nor KL, and a zeroed row also drops out of the DP-reduced
        # token-count denominator. Unparsed DISTRIBUTION rows stay trainable —
        # they carry the worst-case reward, so format compliance is trained.
        response_mask = response_mask * trainable
        batch.batch["response_mask"] = response_mask

        n_masked_tokens = int(response_mask.sum().item())
        metrics = {k: float(v) for k, v in result.metrics.items()}
        metrics["vdct/masked_token_count"] = float(n_masked_tokens)
        for reason in result.skip_reasons:
            if reason is not None:
                key = f"vdct/skip/{reason}"
                metrics[key] = metrics.get(key, 0.0) + 1.0

        if n_masked_tokens == 0:
            # token-mean loss aggregation divides by the masked token count;
            # an entirely masked batch would divide by zero. Skip the actor
            # update for this step (same guard as the RMCT trainer).
            logger.warning(
                "VDCT: no trainable tokens at step %s (has_signal=%s); skipping the actor update",
                getattr(self, "global_steps", "?"),
                result.has_signal,
            )
            metrics["vdct/skipped_update"] = 1.0
            return DataProto.from_single_dict(data={}, meta_info={"metrics": metrics})
        metrics["vdct/skipped_update"] = 0.0

        row_advantages = torch.tensor(result.advantages, dtype=torch.float32, device=device).unsqueeze(-1)
        advantages = row_advantages * response_mask.to(torch.float32)

        kl_coef = float(self.vdct_config.get("kl_coef", 0.0))
        if kl_coef != 0.0:
            if self.vdct_kl_source not in batch.batch:
                raise RuntimeError(
                    f"vdct.kl_coef != 0 but '{self.vdct_kl_source}' is not in the batch "
                    "(rollout_log_probs needs actor_rollout_ref.rollout.calculate_log_probs=True)"
                )
            if "ref_log_prob" not in batch.batch:
                raise RuntimeError(
                    "vdct.kl_coef != 0 but 'ref_log_prob' is not in the batch; the reference "
                    "forward did not run (use_reference_policy)."
                )
            penalty, avg_diff = centered_kl_penalty(
                batch.batch[self.vdct_kl_source].to(torch.float32),
                batch.batch["ref_log_prob"].to(torch.float32),
                response_mask,
                kl_coef,
            )
            advantages = advantages + penalty
            metrics["vdct/kl_policy_base"] = float(avg_diff.item())
        metrics["vdct/kl_coef"] = kl_coef

        batch.batch["advantages"] = advantages
        metrics["vdct/advantage_abs_mean"] = float((advantages.abs().sum() / max(n_masked_tokens, 1)).item())

        actor_output = super()._update_actor(batch)
        actor_output.meta_info.setdefault("metrics", {}).update(metrics)
        return actor_output

    def _log_rollout_data(self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir):
        """Thread VDCT per-row fields into the rollout dump.

        verl's dump writes only ``reward_extra_infos_dict`` columns (plus its
        standard fields); the agent loop's ``extra_fields`` land in
        ``non_tensor_batch`` and would otherwise be lost — the gap inherited
        from the RMCT port. Merge both the raw loop fields and the computed
        per-row results before delegating.
        """
        n = len(batch)
        fields: dict[str, list] = {}
        for key in _DUMP_BATCH_FIELDS:
            values = batch.non_tensor_batch.get(key)
            if values is not None:
                as_list = values.tolist() if isinstance(values, np.ndarray) else list(values)
                fields[key] = as_list
        if self._vdct_dump_fields is not None:
            fields.update(self._vdct_dump_fields)
        merged = merge_dump_fields(reward_extra_infos_dict, fields, n)
        return super()._log_rollout_data(batch, merged, timing_raw, rollout_data_dir)
