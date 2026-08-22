# VDCT on verl

VDCT (verbalized-distribution consistency training) as an **external verl
recipe** — verl itself is not forked or patched. Follows the
`../rmct_verl/` recipe structurally (read that README first for the verl
mechanics, pod bring-up, and the V0/V1 trainer split; none of it is restated
here) and replaces RMCT's one-bit trait with a verbalized probability
distribution over the answer options.

**Status: Phase 0 + Phase 2 implemented and CPU-tested (2026-08-22). Not yet
executed on GPU.** Phase 1 (pod bring-up, model confirmation, audit set,
base-model diagnostics, eval-path check) and the Phase 3 smoke are pending;
per the plan, no training run starts before the Phase 1 and Phase 3 stop
points are explicitly cleared.

## Method (one paragraph)

Four dataset rows per datapoint — `variant ∈ {reference, training}` ×
`kind ∈ {distribution, answer}` — share a `group_id`, with `rollout.n = 8`
uniform. Distribution rows carry the elicitation instruction (CoT, then a
strict `<distribution>` block) and the gradient; answer rows are the
unmodified paired prompt and exist only as outcome samples (zero gradient —
the structural block against self-fulfilling distributions). Per
distribution rollout with parsed distribution `q`:

    training side:   r = -JS(q, q_ref_target) + λ · (1/M) Σ_m log(max(q[a_m], ε))
    reference side:  r =                         λ · (1/M) Σ_m log(max(q[a_m], ε))

`q_ref_target` is this step's mean parsed reference-side distribution for
the group; `a_m` are the same side's parsed answers. There is NO anchor term
(2026-08-22 decision — the plan's frozen `q_ref_initial` is dropped for
now, so nothing structurally blocks matched-but-shifted co-drift; the KL
term, the entropy/calibration diagnostics, and the control arm are the
monitors, and Phase 1's anchor precomputation step disappears). Unparseable
distributions get the worst-case reward `-ln2 + λ·log(ε)` and still train
(format compliance is trained), but are excluded from `q_ref_target`.
Advantages: per-(group, side) standardization via the parity-tested
`slime_port.advantages.normalize_grouped` (per-item mode). The
batch-centered KL vs the frozen base is reused from the RMCT recipe at its
default (`vdct.kl_coef=0.05`; setting it to 0 also skips the per-step
reference forward entirely). Defaults: `λ=1.0` (smoke sweeps
{0.3, 1.0, 3.0}), `ε=1e-3`, LR 1e-5 constant, response ceiling 8,192,
16 datapoints/step → `data.train_batch_size=64` rows → 512 rollouts/step,
256 carrying gradient. Elicitation format decision and its literature
grounding: `notes/elicitation_scheme.md`.

## Layout

| Path | What it is |
| --- | --- |
| `recipe/vdct/vdct_core.py` | Pure math: JS/log-score/entropy/TV, reward + per-group advantage assembly, trainability flags, verl-batch adapters. No Ray, no verl. Bootstraps and reuses `recipe.rmct.rmct_core` (KL term) and `slime_port` (standardization). |
| `recipe/vdct/vdct_schema.py` | Zero-dependency leaf module owning the row vocabulary (variant/kind strings) and the shared JSONL reader — importable without the rmct/slime bootstrap. |
| `recipe/vdct/vdct_elicitation.py` | Elicitation instruction text + strict distribution parser (single source of truth for the surface format). |
| `recipe/vdct/vdct_agent_loop.py` | `@register("vdct")` loop; dispatches on the row's `kind` (see deviation V1). |
| `recipe/vdct/vdct_trainer.py` | `RayVDCTTrainer(RayPPOTrainer)`: advantage/KL/masking stage + a generic `_log_rollout_data` override that dumps every per-row non-tensor column (loop fields and computed rewards/advantages/targets alike). |
| `recipe/vdct/main_vdct.py` | Entry point (`python -m recipe.vdct.main_vdct`); inherits the RMCT runner (role-mapping fix lives in one place) and gates the reference policy on `vdct.kl_coef`. |
| `recipe/vdct/config/` | Hydra overlay (`vdct_trainer.yaml`) + agent-loop registration. |
| `scripts/make_pairs_from_attct.py` | c-wei/AttCT `sycophancy_bct` assets → native paired-prompt JSONL, published as a verified `ctm.artifacts` JSONL/manifest pair (see Data). |
| `scripts/make_vdct_dataset.py` | Paired-prompt JSONL → verl parquet, 4 rows/datapoint. |
| `scripts/vdct_diagnostics.py` | ECE / entropy / cue-invariance from rollout dumps or audit generations. |
| `tests/` | 65 CPU tests: hand-computed reward cases, standardization parity vs `slime_port`, parser incl. malformed cases, builder schema/refusals, converter determinism, diagnostics. |
| `notes/elicitation_scheme.md` | Phase 0 note fixing the elicitation format. |

Run tests: `uv run --no-sync python -m pytest experiments/vdct_verl/tests -q`
(the AttCT-converter integration test skips unless a `c-wei/AttCT` checkout
is present at `$ATTCT_DIR` or `/home/user/c-wei/AttCT`).

## Data

**Primary pool (decision 2026-08-22): the AttCT repo's `sycophancy_bct`
set.** Per the project decision to source everything non-RL from
`https://github.com/c-wei/AttCT` (the *Consistency Training Along the
Transformer Stack* codebase), the training pool is its 4,000-prompt clean
train split (`datasets/sycophancy_bct/control_cot_train.jsonl`; 1,000-prompt
eval split held out) wrapped with its own 12 sycophancy templates
(`data/wrappers.py`). The converter freezes that batch-time-random
construction deterministically (seeded per question) and publishes a
verified JSONL/manifest artifact pair (`ctm.artifacts`) whose provenance
records the checkout SHA, source-file identity, seed, and skip counts:

```bash
export ATTCT_DIR=/path/to/c-wei/AttCT
uv run --no-sync python experiments/vdct_verl/scripts/make_pairs_from_attct.py \
    --style cot --split train --seed 42 \
    --output data/attct_sycophancy_pairs_cot_train.jsonl
```

This satisfies the plan's ≥1024-datapoint requirement (4,000 clean prompts,
minus the few without extractable answer choices). The same JSONL feeds
`../rmct_verl/scripts/make_rmct_dataset.py` unchanged, so the RMCT baseline
arm trains on the identical pool. Alternate inputs remain supported:
native mcq-bias rows (`--input-format native`, e.g. LogiQA+HellaSwag
wrong-argument pairs from `ctm_data.adapters.mcq_bias.materialize`) and the
shared `ctm.prompt_pairs` schema (`--input-format prompt_pairs`, e.g. the
irpan_2510_27062 artifacts — loaded manifest-verified through
`ctm.settings.pairs.load_pair_artifact`, per the repo's frozen-artifact
rule).

Then the VDCT parquet:

```bash
uv run --no-sync python experiments/vdct_verl/scripts/make_vdct_dataset.py \
    --input  data/attct_sycophancy_pairs_cot_train.jsonl \
    --output /workspace/vdct/data/vdct_rows.parquet
```

Batch arithmetic: `rollout.n=8` uniform (also fixes M=8 answer samples per
side), `data.train_batch_size = 4 × datapoints_per_step` (64 for the
standard 16), one epoch = 4 × datapoints rows.

## Launch

Environment exactly as `../rmct_verl/README.md` §Launch (same pinned verl
`2b0fe51`, same smoke-script override arrays, same PYTHONPATH mechanics),
with both recipe roots exposed and no LoRA (VDCT default is full-parameter
on a dense model — Qwen3-8B pending Phase 1 confirmation; the reference
forward runs in a separate worker, handled by `main_vdct`'s role-mapping
fix):

```bash
export PYTHONPATH="${CTM_DIR}/experiments/vdct_verl:${CTM_DIR}/experiments/rmct_verl:${PYTHONPATH:-}"
export RMCT_SLIME_PORT_DIR="${CTM_DIR}/experiments/rmct_slime_qwen"
pip install zstandard   # answer classification imports slime_port.rmct_rollout
cd "${VERL_DIR}"

python3 -m recipe.vdct.main_vdct \
    "${MODEL[@]}" "${ACTOR[@]}" "${REF[@]}" "${ROLLOUT[@]}" \
    data.train_files=/workspace/vdct/data/vdct_rows.parquet \
    data.val_files=/workspace/vdct/data/vdct_rows.parquet \
    data.train_batch_size=64 \
    data.max_prompt_length=4096 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    \
    vdct.lambda_log_score=1.0 \
    vdct.kl_coef=0.05 \
    \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    algorithm.use_kl_in_reward=False \
    \
    trainer.n_gpus_per_node=1 trainer.nnodes=1 \
    trainer.total_epochs=1 \
    trainer.logger='["console"]' \
    trainer.experiment_name=vdct_smoke \
    trainer.default_local_dir=/workspace/ckpts/vdct \
    trainer.rollout_data_dir=/workspace/runs/vdct/rollouts \
    +ray_kwargs.ray_init.runtime_env.env_vars.RMCT_SLIME_PORT_DIR="${RMCT_SLIME_PORT_DIR}"
```

`ROLLOUT_N=8` must be exported before sourcing the smoke-script blocks
(`ROLLOUT` interpolates it); `agent_loop_config_path` must be ABSOLUTE on
the pod (override
`actor_rollout_ref.rollout.agent.agent_loop_config_path`), same landmine as
RMCT. The RMCT deltas re: hydra key collisions apply verbatim — plus one
more filter for VDCT: the smoke script's `MODEL`/`ACTOR` arrays configure
LoRA (regex, merge-mode sync, `save_lora_only`); drop every `lora`-touching
entry for the full-parameter default, the same way the KL entries are
filtered.

## Evaluation

Two evaluation tracks, both untouched by this recipe:

1. **CTM frozen HLE suite** (the plan's Phase 5 primary metrics) via the
   in-tree runner, exactly as for RMCT.
2. **The AttCT repo's three-way sycophancy split** (for comparability with
   the Transformer-Stack paper, per the 2026-08-22 dataset decision), run
   through that repo's `run_evals.py` on the exported HF checkpoint:
   held-out bias BRR (Chua-suite held-out bias types), bias-on-MMLU BRR,
   and the Anthropic model-written-evals sycophancy rate (n=999, 50% = no
   sycophancy). Pre-training comparators for the shared models are that
   paper's Table 11 — for Qwen3-8B (this recipe's default): bias-on-MMLU
   BRR 0.198, held-out BRR 0.309, MWE syc. rate 0.877, MMLU acc. 0.740.

## Deviations from the plan document

| # | Deviation | Why |
| --- | --- | --- |
| V1 | **Answer rows route to the `vdct` agent loop, not the `rmct` loop.** Prompt, sampling, and classification are identical (same `parse_answer` + `matches_bias` via `slime_port`), but the loop additionally forwards the parsed option index. | The log-score term needs `q[a_m]` — the actual option each answer rollout chose. The RMCT loop forwards only the one-bit `trait`, which cannot supply it. |
| V2 | **No anchor term** (2026-08-22 user decision): reference-side rollouts train on the proper-scoring term alone, the `q_ref_initial` column and its Phase 1 precomputation are dropped, and a group whose reference-side distributions all fail to parse drops its training rows (`no_reference_target`, the analog of RMCT's `no_reference_rate`). | "No anchor for now" — same status as the RMCT port's unsupported `anchor_weight`. Cost: co-drift to a matched-but-shifted distribution is no longer structurally blocked; monitored via the KL term, entropy/calibration diagnostics, and the control arm. |
| V3 | **A side with no parsed answer rollouts omits the log-score term for that step** (counted in `vdct/sides_missing_answer_samples`). | Preferable to inventing pseudo-answers; the JS term still trains. |
| V4 | **Rollout dumps carry the VDCT per-row fields** (parsed distributions, answers, rewards, advantages, targets) via a `_log_rollout_data` override — the RMCT port's known dump gap is fixed for this recipe. | The plan requires parsed distributions in dumps for diagnostics. In verl `2b0fe51`'s `fit()`, the dump runs after `_update_actor` on the same batch, so the stashed per-row results are aligned. |
| V5 | **Dataset sourcing switched to the c-wei/AttCT assets** (2026-08-22 user decision) rather than generating LogiQA+HellaSwag wrong-argument pairs. The mcq-bias paths remain supported. | Non-RL side of the project standardizes on that repo; its 4,000-prompt sycophancy_bct pool replaces the plan's Phase 1 step 3 generation. |
