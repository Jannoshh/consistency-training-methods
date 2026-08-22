"""VDCT trainer for verl: a ``RayPPOTrainer`` with the advantage stage replaced.

Follows ``recipe.rmct.rmct_trainer`` exactly in structure; see that module's
docstring for the verl mechanics (DP awareness, why grouping uses the dataset
``group_id`` rather than verl's per-row uid). What differs:

- rewards/advantages come from ``vdct_core.compute_row_advantages`` (graded
  JS + log-score rewards over distribution rollouts, per-group
  standardization) instead of the RMCT rate-gap pipeline;
- answer-kind rows are zero-masked alongside anything untrainable — they are
  outcome samples only;
- ``use_reference_policy`` is forced on only when ``vdct.kl_coef`` is
  nonzero: the reference forward exists solely for the KL term, and with the
  term disabled it would be a wasted full forward per step (``main_vdct``
  threads the same condition through worker registration);
- ``_log_rollout_data`` is overridden to thread the VDCT per-row fields into
  ``reward_extra_infos_dict``, fixing for this recipe the inherited RMCT gap
  where rollout dumps carried only verl's standard fields. The dump is
  generic: ``_update_actor`` writes its computed columns
  (``vdct_core.dump_columns``) into ``batch.non_tensor_batch``, and the
  override then dumps every per-row non-tensor column not on a small
  exclusion list — a new agent-loop ``extra_field`` or computed column
  appears in dumps automatically. In verl 2b0fe51's ``fit()`` the dump runs
  after ``_update_actor`` on the same batch object, so the computed columns
  are aligned and current.

The batch-centered KL term (tinker semantics) is reused from the RMCT recipe
via ``vdct_core.centered_kl_penalty``.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

from .vdct_core import (
    VDCTConfig,
    build_rows,
    centered_kl_penalty,
    compute_row_advantages,
    dump_columns,
    merge_dump_fields,
)

logger = logging.getLogger(__name__)

# Per-row non-tensor columns NOT copied into the rollout dump: bulky prompt
# payloads already covered by the dump's own "input" column, and verl's
# structured plumbing fields.
_DUMP_EXCLUDED_COLUMNS = {
    "raw_prompt",
    "raw_prompt_ids",
    "reward_model",
    "extra_info",
    "tools_kwargs",
    "interaction_kwargs",
    "multi_modal_data",
    "multi_modal_inputs",
    "turn_scores",
    "tool_rewards",
}


def _vdct_config(vdct_cfg) -> VDCTConfig:
    """Build the math config from the hydra ``vdct`` block."""
    return VDCTConfig(
        lambda_log_score=float(vdct_cfg.get("lambda_log_score", 1.0)),
        consistency_weight=float(vdct_cfg.get("consistency_weight", 1.0)),
        epsilon=float(vdct_cfg.get("epsilon", 1e-3)),
        normalization=vdct_cfg.get("normalization", "per_item"),
    )


class RayVDCTTrainer(RayPPOTrainer):
    """RayPPOTrainer with the VDCT advantage/KL/masking stage."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vdct_config = self.config.vdct
        self.vdct_math_config = _vdct_config(self.vdct_config)
        self.vdct_kl_coef = float(self.vdct_config.get("kl_coef", 0.0))
        # Must be set before init_workers(). The KL term targets the frozen
        # base even though both of verl's own KL switches are off — but with
        # the term disabled there is nothing to compute against, so skip the
        # per-step reference forward entirely.
        self.use_reference_policy = self.vdct_kl_coef != 0.0
        kl_source = self.vdct_config.get("kl_logprob_source", "old_log_probs")
        if kl_source not in ("old_log_probs", "rollout_log_probs"):
            raise ValueError(f"vdct.kl_logprob_source must be old_log_probs or rollout_log_probs, got {kl_source!r}")
        self.vdct_kl_source = kl_source
        if self.config.algorithm.get("use_kl_in_reward", False) or self.config.actor_rollout_ref.actor.use_kl_loss:
            raise ValueError(
                "VDCT owns the KL term; set algorithm.use_kl_in_reward=False and "
                "actor_rollout_ref.actor.use_kl_loss=False (vdct.kl_coef controls the VDCT KL)."
            )

    def _update_actor(self, batch: DataProto) -> DataProto:
        rows = build_rows(batch.non_tensor_batch)
        result = compute_row_advantages(rows, self.vdct_math_config)

        # Attach the computed per-row columns to the batch so the generic
        # rollout dump carries them (np.empty keeps list-valued entries
        # per-row instead of collapsing equal-length lists into a 2-D array).
        for key, values in dump_columns(rows, result).items():
            column = np.empty(len(values), dtype=object)
            column[:] = values
            batch.non_tensor_batch[key] = column

        response_mask = batch.batch["response_mask"]
        device = response_mask.device
        trainable = torch.tensor(result.trainable, dtype=response_mask.dtype, device=device).unsqueeze(-1)
        # Zero the mask for every answer row, and (when the batch has no
        # signal) everything: those rows must contribute neither policy
        # gradient nor KL, and a zeroed row also drops out of the DP-reduced
        # token-count denominator. Unparsed DISTRIBUTION rows stay trainable —
        # they carry the worst-case reward, so format compliance is trained.
        response_mask = response_mask * trainable
        batch.batch["response_mask"] = response_mask

        n_masked_tokens = int(response_mask.sum().item())
        metrics = dict(result.metrics)
        metrics["vdct/masked_token_count"] = float(n_masked_tokens)

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

        if self.vdct_kl_coef != 0.0:
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
                self.vdct_kl_coef,
            )
            advantages = advantages + penalty
            metrics["vdct/kl_policy_base"] = float(avg_diff.item())
        metrics["vdct/kl_coef"] = self.vdct_kl_coef

        batch.batch["advantages"] = advantages
        metrics["vdct/advantage_abs_mean"] = float((advantages.abs().sum() / n_masked_tokens).item())

        actor_output = super()._update_actor(batch)
        actor_output.meta_info.setdefault("metrics", {}).update(metrics)
        return actor_output

    def _log_rollout_data(self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir):
        """Thread VDCT per-row fields into the rollout dump.

        verl's dump writes only ``reward_extra_infos_dict`` columns (plus its
        standard fields); the agent loop's ``extra_fields`` land in
        ``non_tensor_batch`` and would otherwise be lost — the gap inherited
        from the RMCT port. Copy every per-row non-tensor column (raw loop
        fields and the computed columns ``_update_actor`` attached), minus
        the exclusion list, before delegating.
        """
        n = len(batch)
        fields = {
            key: values.tolist() if isinstance(values, np.ndarray) else list(values)
            for key, values in batch.non_tensor_batch.items()
            if key not in _DUMP_EXCLUDED_COLUMNS and len(values) == n
        }
        merged = merge_dump_fields(reward_extra_infos_dict, fields, n)
        return super()._log_rollout_data(batch, merged, timing_raw, rollout_data_dir)
