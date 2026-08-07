# RMCT on verl

RMCT (reference-matched consistency training) as an **external verl recipe** —
verl itself is not forked or patched. This is the port sketched in
`../rmct_slime_qwen/verl_derisk.md` ("RMCT port shape"), written against a
pinned verl and reusing the Gate-A-tested math from the slime/Miles port
verbatim.

**Status: written and unit-tested on CPU only. Never executed on a GPU.** The
verl LoRA smoke test passed (commit `1e78f8c`), which clears the stack; it does
not exercise a single line of this recipe, and see the `use_v1` caveat below.

```
Pinned verl:  2b0fe51   ("[rocm] feat: enable DeepSeek-V4-Flash GRPO on AMD GPUs (#7050)")
Math source:  experiments/rmct_slime_qwen/slime_port/{pipeline,rewards,advantages,rmct_advantage}.py
Environment:  experiments/rmct_slime_qwen/verl_smoke_notes.md  (image, pip layer, LoRA regex, VRAM)
              experiments/rmct_slime_qwen/scripts/verl_run_smoke.sh  (verified stages + override arrays)
```

The smoke script is the single source of truth for the image, pip layer, LoRA
regex, and FSDP/vLLM override arrays. This README does not restate them; it
lists only what RMCT changes on top.

## Layout

| Path | What it is |
| --- | --- |
| `recipe/rmct/rmct_core.py` | Pure functions: row→group→advantage mapping and the centered KL term. No Ray, no verl, no distributed. Everything else imports from here. |
| `recipe/rmct/rmct_agent_loop.py` | `@register("rmct")` agent loop: tokenize the row's prompt, sample once, classify the answer, emit `variant`/`group_id`/`parse_ok`/`trait` via `extra_fields`. |
| `recipe/rmct/rmct_trainer.py` | `RayRMCTTrainer(RayPPOTrainer)`: forces the reference policy on, overrides `_update_actor` to write RMCT `advantages` + `response_mask`. |
| `recipe/rmct/main_rmct.py` | Entry point (`python -m recipe.rmct.main_rmct`). |
| `recipe/rmct/config/rmct_trainer.yaml` | Hydra overlay on verl's `ppo_trainer`, plus the `rmct.*` block. |
| `recipe/rmct/config/agent_loop.yaml` | Registers the `rmct` agent loop in each `AgentLoopWorker` actor. |
| `scripts/make_rmct_dataset.py` | Paired-prompt JSONL → verl parquet (2 rows per datapoint). |
| `tests/test_rmct_verl_math.py` | Gate A for this port: bitwise parity against `slime_port.pipeline`, plus masking and KL semantics. |

The RMCT math is **imported**, never copied: `rmct_core` puts
`experiments/rmct_slime_qwen` on `sys.path` (override with
`RMCT_SLIME_PORT_DIR`) and calls `slime_port.pipeline.build_batch_item` /
`compute_batch_advantages` directly. A vendored copy could drift from the
parity-tested original; packaging was not an option because neither experiment
directory is installable.

## How it maps onto verl

* **Dataset**: two rows per datapoint sharing a `group_id` —
  `variant="reference"` carries `unbiased_messages`, `variant="training"`
  carries `biased_messages`. verl repeats each row `rollout.n` times, so one
  dataset row becomes one rollout population.
* **Rollout**: the `rmct` agent loop samples the row's prompt once and
  classifies the answer with the *same* function the slime port uses
  (`slime_port.rmct_rollout._classify` → mcq-bias `parse_answer` + vendored
  `matches_bias`). A response that hits the length cap counts as a parse
  failure, matching `_to_rollout`.
* **Advantage**: `_update_actor` groups rows by `group_id`, splits by
  `variant`, computes `p_ref`/`p_hat` and the RMCT advantage through the slime
  pipeline, broadcasts the per-row scalar over `response_mask`, and adds the
  batch-centered KL.
* **Skipping**: zeroing `response_mask` is verl's first-class skip mechanism —
  a zeroed row contributes no gradient, no KL, and drops out of the
  DP-reduced token-count denominator. Reference rows, unparsed training rows,
  and (when the batch has no signal) everything get zeroed.
* **Reference model**: with `lora_rank > 0` verl sets `ref_in_actor`, so
  `ref_log_prob` comes from the *same* worker with the adapter disabled — the
  frozen base model, which is exactly RMCT's KL target. No second model is
  materialized.

## Launch

### 0. Environment

Bring the pod up exactly as the smoke test did — image, pip layer, model
download, and the CPU-only preflight (which is what actually proves the LoRA
regex hits attention+MLP and misses the GatedDeltaNet):

```bash
experiments/rmct_slime_qwen/scripts/verl_run_smoke.sh env
experiments/rmct_slime_qwen/scripts/verl_run_smoke.sh model
experiments/rmct_slime_qwen/scripts/verl_run_smoke.sh preflight
```

Then add what RMCT needs on top. The recipe lives outside the verl clone; Ray
workers **do** inherit the driver's `PYTHONPATH`
(`verl/trainer/constants_ppo.py:121`), so exposing it that way is enough, but
`RMCT_SLIME_PORT_DIR` must be forwarded explicitly through the Ray runtime env
(see the launch command).

```bash
export VERL_DIR=/workspace/verl                       # clone at 2b0fe51, pip install --no-deps -e .
export CTM_DIR=/workspace/consistency-training-methods
export PYTHONPATH="${CTM_DIR}/experiments/rmct_verl:${PYTHONPATH:-}"
export RMCT_SLIME_PORT_DIR="${CTM_DIR}/experiments/rmct_slime_qwen"
pip install zstandard                                  # slime_port.rollout_writer imports it
cd "${VERL_DIR}"                                       # the hydra searchpath is CWD-relative
```

`recipe/` is an implicit namespace package on both sides, so
`${CTM_DIR}/experiments/rmct_verl/recipe` and `${VERL_DIR}/recipe` merge:
`recipe.rmct` and `recipe.dapo` both import.

### 1. Dataset

```bash
uv run --no-sync python "${CTM_DIR}/experiments/rmct_verl/scripts/make_rmct_dataset.py" \
    --input  "${CTM_DIR}/experiments/rmct_slime_qwen/data/run_9b/wrong-argument-pairs-64.jsonl" \
    --output /workspace/rmct/data/run_9b/rmct_pairs.parquet \
    --n-datapoints 64 --n-ref-rollouts 128 --n-train-rollouts 128
```

Batch arithmetic, using the `configs/run_9b.json` values:

| run_9b config | verl override |
| --- | --- |
| `n_ref_rollouts = n_train_rollouts = 128` | `actor_rollout_ref.rollout.n=128` |
| `batch_size = 4` datapoints/step | `data.train_batch_size=8` (= 2 × datapoints, in ROWS) |
| `n_datapoints = 64` | 128 dataset rows = one epoch = 16 steps |

`rollout.n` is necessarily shared by both variants, so this stack cannot
express `n_ref != n_train`; `make_rmct_dataset.py` rejects that combination up
front.

### 2. Train

The LoRA/FSDP/vLLM block is the one validated by the smoke script
(`scratchpad/verl_smoke/run_smoke.sh`) — same regex, same `lora.merge=True`
full-weight sync, same `save_lora_only` checkpointing.

```bash
LORA_TARGET_MODULES='.*language_model\.layers\.[0-9]+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)'

python3 -m recipe.rmct.main_rmct \
    data.train_files=/workspace/rmct/data/run_9b/rmct_pairs.parquet \
    data.val_files=/workspace/rmct/data/run_9b/rmct_pairs.parquet \
    data.train_batch_size=8 \
    data.max_prompt_length=4096 \
    data.max_response_length=20480 \
    data.shuffle=False \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    \
    rmct.kl_coef=0.05 \
    rmct.advantage_estimator=grpo_normalized \
    rmct.normalization=per_item \
    rmct.anchor_weight=0.0 \
    \
    actor_rollout_ref.model.path=/workspace/models/Qwen3.5-9B \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=64 \
    actor_rollout_ref.model.target_modules="'${LORA_TARGET_MODULES}'" \
    ++actor_rollout_ref.model.lora.merge=True \
    \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=1.0e-05 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=1 \
    ++actor_rollout_ref.actor.checkpoint.save_lora_only=True \
    \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.ref.use_torch_compile=False \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=128 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
    \
    algorithm.use_kl_in_reward=False \
    \
    trainer.n_gpus_per_node=1 trainer.nnodes=1 \
    trainer.total_epochs=1 \
    trainer.logger='["console"]' \
    trainer.experiment_name=rmct_9b_lora \
    trainer.default_local_dir=/workspace/ckpts/rmct_9b \
    trainer.rollout_data_dir=/workspace/runs/rmct_9b/rollouts \
    +ray_kwargs.ray_init.runtime_env.env_vars.RMCT_SLIME_PORT_DIR="${RMCT_SLIME_PORT_DIR}"
```

`trainer.use_v1=False`, `actor_rollout_ref.rollout.agent.agent_loop_config_path`
and the `rmct.*` defaults are pinned in `config/rmct_trainer.yaml`; the
overrides above only cover what is run-specific.

## Deviations from the Miles/slime port

| # | Deviation | Why / impact |
| --- | --- | --- |
| V1 | **Reference rows pay a wasted train forward** with an all-zero mask. Miles never sent perturbation-0 rollouts to the trainer at all. | verl's single-controller `fit()` builds one batch for the whole step; slicing reference rows out needs a `fit()` override. Correctness is unaffected (zero mask ⇒ zero gradient, zero KL, excluded from the loss denominator); cost is roughly a 2× train-forward on the reference half. Fixable later. |
| V2 | **KL uses `old_log_probs` by default**, not the sampler's `rollout_log_probs`. Miles/slime used the rollout log-probs (ctm's "sampled" log-probs). | `old_log_probs` is verl's recomputed π_old under the training engine — a cleaner estimate of log π_θ than vLLM's sampler numerics. Switch with `rmct.kl_logprob_source=rollout_log_probs` (needs `actor_rollout_ref.rollout.calculate_log_probs=True`) if a run must match Miles exactly. |
| V3 | **No DP all-reduce for the KL centering mean.** | `_update_actor` runs in the driver on the *whole* pre-dispatch batch, so the mean is already global. The Miles port's `_dp_all_reduce_pair` existed only because its advantage hook ran inside each DP rank. This is strictly simpler, not different. |
| V4 | **Truncation is detected by length**, not by a finish reason. | verl's `TokenOutput` carries no `finish_reason` (only `stop_reason ∈ {completed, aborted}`), so `len(token_ids) >= response_length` stands in. A response that legitimately ends exactly at the cap would be misclassified as truncated — same direction of error as the slime port, negligible at 20k tokens. |
| V5 | **verl computes GRPO advantages that are then overwritten.** | `compute_advantage` runs in `fit()` before `_update_actor`. Rewards are all zero (the agent loop reports `reward_score=0.0`), so the pass is cheap and its output is fully replaced. Removing it needs a `fit()` override. |
| V6 | **Zero-signal batches skip the optimizer step** rather than running it with zero advantages. | Miles zero-masked the samples and still stepped. Under verl, an all-zero mask makes `token-mean` aggregation divide by zero, so `_update_actor` returns early with `rmct/skipped_update=1`. This is closer to ctm's original loop, which skipped the step outright. |
| V7 | **`n_ref_rollouts` must equal `n_train_rollouts`.** | verl samples `rollout.n` completions per dataset row, and both variants are rows in one dataset. run_9b already uses 128/128. |
| V8 | **`anchor_weight > 0` is rejected.** | Same restriction as the slime port — the anchor term needs the one-time initial-reference-rate measurement, which neither port carries. |
| V9 | **Per-step rollout persistence is `trainer.rollout_data_dir`**, not ctm-schema `step_*.jsonl.zst` records. | The RMCT record fields (`p_ref`, `p_hat`, `parse_ok`, skip reasons) ride along as reward-extra/metric fields. Converting the dump to the ctm schema is a separate task if the analysis tooling needs it. |

## API mismatches found against verl 2b0fe51

1. **`verl.trainer.main_ppo` no longer exports `TaskRunner` / `create_rl_dataset` /
   `create_rl_sampler`.** The DAPO recipe in this clone still imports them from
   there and would fail; it pins older SHAs in `recipe/dapo/REQUIRED_VERL.txt`.
   This recipe imports `BaseTaskRunner` from `verl.trainer.main_ppo_v0` and the
   dataset helpers from `verl.trainer.ppo.utils`.
2. **`trainer.use_v1` defaults to `true`.** The default path is `TaskRunnerV1`
   plus the TransferQueue trainer in `verl/trainer/ppo/v1/`, whose
   `PPOTrainer._update_actor(batch: KVBatchMeta, metrics)` is a different
   signature over a different data model. The `RayPPOTrainer` this recipe
   extends is the V0 trainer, marked
   `@deprecated("will be removed in v0.9.0")`. The recipe pins
   `trainer.use_v1=False` and `main_rmct.main` refuses to start otherwise.
   **The verl LoRA smoke test ran the V1 default**, so a green smoke result
   does not by itself validate the code path this recipe uses.
3. **`main_ppo_v0.TaskRunner` is `@ray.remote`-decorated** and therefore not
   subclassable; `RMCTTaskRunner` extends `BaseTaskRunner` and replicates
   `run()`.
4. **Dataset columns can be dropped when the async reward loop is enabled.**
   `AgentLoopWorker._postprocess` only re-attaches the input non-tensor columns
   when `reward_loop_worker_handles is None` (agent_loop.py:1097). The agent
   loop therefore echoes `variant`/`group_id`/`parse_ok`/`trait` back through
   `extra_fields`, which is propagated unconditionally.
5. **`extract_reward` requires `rm_scores` in the batch.** RMCT has no reward
   model and no reward function; the agent loop sets `reward_score=0.0`, which
   both creates the tensor and short-circuits `AgentLoopWorker._compute_score`.

## Tests

```bash
uv run --no-sync python -m pytest experiments/rmct_verl/tests -q     # 61 passed
```

Gate A for this port asserts the row-level advantages equal an independently
constructed direct call to `slime_port.pipeline` on the same populations
(exact float equality, across all six estimator/normalization combinations and
eight seeds), plus row-order invariance, the three masking rules, and the KL
term against the `rmct_advantage.py` formula. The tests import `rmct_core`
directly and need neither verl nor a GPU.
