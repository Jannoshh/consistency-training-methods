# RMCT on verl

RMCT (reference-matched consistency training) as an **external verl recipe** —
verl itself is not forked or patched. This is the port sketched in
`../rmct_slime_qwen/verl_derisk.md` ("RMCT port shape"), written against a
pinned verl and reusing the Gate-A-tested math from the slime/Miles port
verbatim.

**Status: EXECUTED END-TO-END ON GPU (2026-08-07, 1× H200, Qwen3.5-2B tiny
shape).** One full RMCT step on the real wrong-argument pairs ran to exit 0
through the v0 trainer: agent-loop rollouts (16 rows × n=8 = 128 samples,
correct count in the rollout dump), ref + policy logprobs, the RMCT
`_update_actor` override, and the GDN backward. `rollout_probs_diff_mean`
0.0047, sampling↔training pearson 0.9997, 134s/step at tiny shape. The verl
LoRA smoke test also passed separately (commit `1e78f8c`).

GPU bring-up fixes/environment additions beyond `verl_smoke_notes.md` (all
hit in sequence on a fresh pod):
- `pip install cachetools tilelang` — cachetools for verl's llm_server;
  **tilelang is required on Hopper**: fla's `chunk_bwd_dqkwg` hard-errors on
  Triton 3.4–3.7.0 (fla#640) and tilelang is the sanctioned fallback.
- vLLM must be installed WITH its dependency closure (a `--no-deps` install
  misses pybase64 etc.), and transformers must be re-pinned `>=5.5.3,<5.11`
  AFTERWARD (vLLM 0.18.1's resolver downgrades it below qwen3_5 support).
- Without flash-attn, verl's FSDP worker fails twice: transformers defaults
  to flash_attention_2 (`+actor_rollout_ref.model.override_config.attn_implementation=sdpa`
  works around it) but `verl/utils/attention_utils.py` imports
  `flash_attn.bert_padding` unconditionally — flash-attn is REQUIRED, build
  it (MAX_JOBS≤8) or reuse the cached wheel.
- `actor_rollout_ref.rollout.agent.agent_loop_config_path` must be ABSOLUTE
  (relative paths resolve against the verl clone, not this recipe).
- The non-LoRA role-mapping fix in `main_rmct.py` (commit `90a4a82`).

Known gap: the rollout dump (`trainer.rollout_data_dir`) contains verl's
standard fields only — the per-sample `variant`/`p_ref`/`p_hat`/`parse_ok`
extras are not yet threaded into `reward_extra_infos_dict`. Fix before
using dumps for Gate A replay.

## 9B LoRA science-shape step time (2026-08-07, 1× H200, MEASURED)

RMCT recipe + LoRA (r32/α64, attn+MLP regex, merge-mode sync) at the exact
paper shape — 4 datapoints × (128 ref + 128 train) rollouts per step,
max_response 20,480, packing + dynamic batching (40,960-token ceilings),
prefix caching on:

| phase | step 1 (cold) | step 2 (warm) |
|---|---:|---:|
| generation | 966s | 851s |
| policy logprobs | 338s | 267s |
| ref logprobs | 250s | 232s |
| update_actor | 964s | 829s |
| merged weight sync | 14s | 14s |
| **total step** | **2,537s (42.3 min)** | **2,197s (36.6 min)** |

~5.7–6.2M generated tokens/step (mean response ~5.1–5.6k), overall step
throughput ~2,600 tok/s on the single GPU. Health at science shape:
rollout_probs_diff_mean 0.0041–0.0044, grad_norm 0.014–0.016, nonzero
pg_loss.

**Comparison (identical config, data, and workload):**

| stack | wall/step | hardware | GPU-min/step |
|---|---:|---|---:|
| Miles Megatron LoRA (bshd, micro-batch 1) | 81 min | 2× H200 | 162 |
| **verl FSDP LoRA (packed)** | **36.6 min** | **1× H200** | **36.6** |

**4.4× cheaper per GPU, 2.2× faster wall-clock on half the hardware.** A
64-step run (1 epoch × 16 steps × 4 epochs equivalent of the RMCT-256-style
budget math) extrapolates to ~39 GPU-hours ≈ $180 on one H200 at $4.59/hr,
vs ~$650 Miles-equivalent. Obvious further levers: DP over 2–4 GPUs
(near-linear for every phase), and update_actor (38% of step) still runs
full gradient checkpointing.

Measurement caveats: rank 32 (verl-recommended) vs the paper's r8 — timing-
irrelevant, but a science run must record the deviation; driver-550 pod, so
the stack was vLLM 0.18.1 + torch 2.10/cu128 exactly as in the notes.
Log: RunPod volume `/workspace/verl_cache/rmct_lora_science_measure.log`
(volume is per-pod — copy off before terminating).

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

Reuse the smoke script's verified override arrays verbatim — `MODEL`, `ACTOR`,
`REF`, `ROLLOUT` in `experiments/rmct_slime_qwen/scripts/verl_run_smoke.sh`
(lines 192-256), whose every nonobvious entry is sourced in
`verl_smoke_notes.md`. They cover the LoRA regex, `lora.merge=True` full-weight
sync, `save_lora_only`, FSDP2, and the vLLM settings — do not re-derive them.

The script cannot simply be sourced: it dispatches on `$1` at the bottom and
would launch the smoke run. Extract the arrays together with the
user-adjustable block they interpolate, and source that instead.

Hydra also **rejects a key passed twice** ("Multiple values for ..."), so the
RMCT deltas cannot simply be appended after the arrays. Three keys collide:
`ppo_mini_batch_size` and `rollout.n` come from shell variables (set them by
export) and `use_kl_loss=True` is hardcoded (filter it out). This block does
all of it and is tested against the script at `1e78f8c`:

```bash
SMOKE="${CTM_DIR}/experiments/rmct_slime_qwen/scripts/verl_run_smoke.sh"
sed -n '24,61p;192,256p' "${SMOKE}" > /tmp/verl_blocks.sh   # defaults + MODEL/ACTOR/REF/ROLLOUT

export MODEL_PATH=/workspace/models/Qwen3.5-9B
export PPO_MINI_BATCH_SIZE=8      # ACTOR interpolates this
export ROLLOUT_N=128              # ROLLOUT interpolates this
source /tmp/verl_blocks.sh

# drop the smoke run's verl-native KL settings; RMCT owns the KL term
KEEP=(); for o in "${ACTOR[@]}"; do [[ "$o" == *kl_loss* ]] || KEEP+=("$o"); done
ACTOR=("${KEEP[@]}")
```

Re-check those line numbers against the script before trusting them.

**RMCT changes exactly these overrides on top of the smoke block:**

| Override | Why it differs from the smoke run |
| --- | --- |
| `actor_rollout_ref.actor.use_kl_loss=False` (smoke: `True`, `kl_loss_coef=0.001`) | RMCT owns the KL term; `RayRMCTTrainer.__init__` refuses to start if either verl KL path is on. Applied by filtering the `ACTOR` array above. |
| `algorithm.use_kl_in_reward=False` | Same reason. The reference forward still runs because the trainer forces `use_reference_policy=True`. |
| `ROLLOUT_N=128` (smoke: 4) | One rollout population per (datapoint, variant) row. Set by export, since `ROLLOUT` interpolates it. |
| `PPO_MINI_BATCH_SIZE=8` (smoke: 8) | Prompts per optimizer step; exported for the same reason. |
| `actor_rollout_ref.rollout.temperature=1.0 top_p=1.0 top_k=-1` | Rate-affecting sampling, pinned from `configs/run_9b.json`. |
| `data.train_batch_size=8` (smoke: 32) | 2 rows × 4 datapoints per step. |
| `data.max_response_length=20480` (smoke: 1024) | `max_new_tokens` from `configs/run_9b.json`. |
| `data.max_prompt_length=4096` (smoke: 512) | The wrong-argument prompts are long; `filter_overlong_prompts=True` will tell you if this is short. |
| `data.shuffle=False` | Row order is the frozen artifact's order. |

```bash
python3 -m recipe.rmct.main_rmct \
    "${MODEL[@]}" "${ACTOR[@]}" "${REF[@]}" "${ROLLOUT[@]}" \
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
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
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

### `trainer.use_v1` — read this before launching

verl's default is `use_v1: true`, which routes to `TaskRunnerV1` and the
TransferQueue trainer in `verl/trainer/ppo/v1/`. This recipe subclasses the V0
`RayPPOTrainer`, so **the launch must run with `trainer.use_v1=False`.** It is
pinned in `config/rmct_trainer.yaml` and `main_rmct.main` raises if it is ever
overridden back to true, so this cannot fail silently — but it does mean the
smoke run and the RMCT run exercise different trainers:
`verl_run_smoke.sh` launches `python3 -m verl.trainer.main_ppo` with no
`use_v1` override (line 278), so the passing smoke test validated the **V1**
path. Expect V0-only surprises (LoRA weight sync, checkpointing, metrics keys)
not to be covered by it; watch the same signals `verl_smoke_notes.md` §7 lists.

**Future work: port to the V1 TransferQueue trainer.** V0 carries
`@deprecated("will be removed in v0.9.0")`. The V1 equivalent is
`PPOTrainer._update_actor(batch: KVBatchMeta, metrics)` in
`verl/trainer/ppo/v1/trainer_base.py:1672` — a different data model, so the
port is a real piece of work. `rmct_core.py` is deliberately free of verl
imports and carries over unchanged; only `rmct_trainer.py` needs rewriting.

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
