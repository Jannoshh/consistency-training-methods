# Phase 0 report — verified assumptions and contradictions

Date: 2026-08-05. Sources: ctm @ 58c889c (branch `training-profiling-rebase`),
slime @ f655e13 (2026-08-04, after v0.3.0), `Qwen/Qwen3.5-9B` / `-2B` model
cards, SGLang docs/issues. Full research transcripts live in the session; this
file records what changes the plan.

## Confirmed as assumed

- slime supports custom rollout functions (`--rollout-function-path`,
  ~25 injection points, CPU-only contract tests under `tests/plugin_contracts/`).
- **Externally computed advantages are first-class**:
  `--custom-advantage-function-path` fully bypasses built-in GRPO grouping
  (undocumented in the customization table; declared
  `slime/utils/arguments.py:958`, dispatched
  `slime/backends/megatron_utils/loss.py:715`).
- Qwen3.5 is first-class in slime on both sides: `scripts/models/qwen3.5-9B.sh`,
  HF↔Megatron converters registered for `qwen3_5`, and a ready-made
  `examples/fully_async/run-qwen3.5-9B-fully_async.sh`.
- Official Docker images exist (`slimerl/slime:latest`, nightlies; CUDA 12.9,
  B200/sm_100 supported).
- Qwen3.5-9B is dense (~9.65B; the "MoE" line on the card is family marketing),
  hybrid 3:1 GatedDeltaNet:gated-attention, 262k context. A same-architecture
  **Qwen3.5-2B exists** for the dev tier. Apache 2.0.
- MTP draft weights ship inside the official checkpoints (`mtp.*` tensors).
- SGLang prefix caching works for hybrid models (MambaRadixCache); reuse is
  exact-prefix all-or-nothing — the 128-same-prompt fan-out is its best case.
- The mcq-bias trait classifier is a local answer parser
  (`mcq_bias.parsers.parse_answer` + `matches_bias`) — no API judge in the hot
  loop, no judge endpoint requirement.
- `ctm/core/rewards.py` and `ctm/core/advantages.py` are genuinely
  backend-free (stdlib only; `Rollout` needs 3 fields).

## Contradictions / corrections to the plan document

1. **The anchor term is inert in the reference experiment.**
   `experiments/rmct_paper_vast_more_methods/experiment.yaml` sets
   `anchor_weight: 0.0`, so the base-model startup rate measurement and anchor
   rewards never run (despite `rollouts.anchor: 128` being configured). The
   Phase 4 "cache base-model anchor rates" work item only applies if the new
   experiment enables the anchor. The port implements the anchor math anyway
   (parity-tested), but the first 9B run config will carry
   `anchor_weight: 0.0` to match the paper block.
2. **The classifier never abstains in this setting.** The plan says "`None` =
   abstention"; the mcq-bias classifier returns 0.0 for unparseable responses
   and never `None`. Exclusion happens via `parsed_successfully`
   (parse + logprobs present), not classifier abstention.
3. **`p_ref` is re-measured from the current policy every step**, not frozen;
   only the (inert) anchor uses frozen initial rates. The plan's summary was
   ambiguous on this.
4. **KL term semantics differ between loop and port.** In ctm, `kl_coef=0.05`
   is applied by the backend mutating datum advantages in place
   (`incorporate_kl_penalty`), after rollout logging (logged advantages are
   pre-KL). slime instead offers reward-side KL (`--kl-coef`, inert for
   GRPO-family estimators) and loss-side KL (`--use-kl-loss --kl-loss-coef`,
   `--kl-loss-type`). These are not numerically identical to the tinker-style
   advantage shaping. **Deviation to record when the training config is
   frozen**: which slime KL formulation replaces `incorporate_kl_penalty`, with
   coefficient 0.05 carried over.
5. **No existing rollout JSONLs for Gate A replay.** The paper runs lived on
   Vast.ai/Isambard; `logs/` was never created on this machine. Gate A is
   split: (a) synthetic parity vs the original `_build_training_batch` —
   implemented and passing (see below); (b) replay of a real rollout log —
   implemented, activates via `RMCT_ROLLOUT_DIR`, needs either a retrieved log
   from a past run or the first dev-tier run's own log.
6. **Vision tower: no text-only variant exists.** vLLM can skip loading it
   (`--language-model-only`); no documented SGLang equivalent — memory cost of
   the ViT on the rollout side must be verified in Phase 3. On the training
   side, freezing/stripping the ViT is our explicit job. Deviation record:
   text-only RL with vision tower frozen (strategy fixed in Phase 3).
7. **Known correctness bugs to design around** (all verified upstream):
   - SGLang garbles Qwen3.5-dense rollouts at `--rollout-num-gpus-per-engine > 1`
     (gen_tp>1); fixed by sglang#19411 — verify the pinned image contains it,
     else keep 1 GPU/engine or patch `qwen3_5.py`.
   - Wrong outputs with attention DP=2 + NVLS on Hopper (open) — avoid dp 2.
   - Fine-tuned re-saves must preserve `mtp.*` tensors or speculative decoding
     silently drops to 0% acceptance.
   - Megatron TP does not shard the wrapped GDN/attention module (HF-wrapper
     approach) — model-level TP works; fine at 9B, watch memory.
8. **Fully-async mode restarts aborted rollouts from scratch** (no resume);
   only standard `--partial-rollout` resumes partial generations. Restart is
   acceptable under the plan's constraint (no truncation-as-completion, no
   selection against long responses — every trajectory still runs to
   completion), but it wastes tokens and raises staleness; Phase 5 will
   compare. slime has **no FSDP escape hatch** (Megatron-only training).
9. **Sampling defaults trap:** the Qwen3.5 model card recommends nonzero
   `presence_penalty` defaults; RMCT must pin temperature 1.0, no penalties,
   explicitly — inheriting card defaults would be a silent rate-affecting
   change.
10. **`train_metadata` is silently dropped** at slime's DP split (not in the
    copy whitelist despite docs). Custom per-sample data must ride the
    `rewards`/`raw_reward` (per-sample float) or `teacher_log_probs`
    (per-token) channels — or advantages must be computed step-level in
    `--custom-reward-post-process-path` before the split.
11. Fault tolerance in slime covers rollout engines only; job-level resume is
    ours (Phase 6). `--save-debug-rollout-data` / `--debug-train-only` give a
    GPU-cheap replay loop for iterating on the custom advantage path.

## Gate A status (part done)

`slime_port/` vendors `rewards.py`/`advantages.py` verbatim (import line only
changed; provenance in `slime_port/__init__.py`) and ports
`_build_training_batch`'s math as `pipeline.py`.
`tests/test_reward_parity.py`:

- Synthetic parity: 37 cases across
  {grpo_normalized, snr_scaling, matched_pair} × {per_item, pooled} ×
  anchor_weight {0, 0.5} — **exact float equality** against the original
  `RLTrainer._build_training_batch`. Passing.
- Replay parity vs a real run's `step_*.jsonl.zst`: implemented, pending real
  rollout data (`RMCT_ROLLOUT_DIR=...`).

## Deviations record (running)

| # | Deviation | Rate-affecting? | Status |
|---|---|---|---|
| D1 | Policy model gpt-oss-20b LoRA r8 → Qwen3.5-9B full-parameter | yes (declared in plan) | decided upfront |
| D2 | Backend Tinker/Local → slime (SGLang + Megatron) | potentially (logprob parity gates) | decided upfront |
| D3 | KL formulation: tinker advantage-mutation ported into slime via `--custom-advantage-function-path` (`slime_port/rmct_advantage.py`) — slime's own KL paths held at zero (`--kl-coef 0 --use-kl-loss --kl-loss-coef 0`, the latter only to force ref-model loading). Exact tinker semantics at any DP size: the centering (Σdiff, Σmask) pair is all-reduced across the data-parallel group before the mean. | yes — resolved to exact port | closed (Phase 3/4) |
| D4 | Vision tower **stripped, not frozen**: the HF→Megatron `convert_hf_to_torch_dist` conversion emits a text-only model (358 tensors, no vision keys — verified in the 2B torch_dist metadata), so the trainer never instantiates the tower and saved checkpoints are text-only. SGLang loads the full HF checkpoint but weight sync only overwrites text weights; base vision weights are inert on text-only prompts. Reusing a trained checkpoint multimodally later requires grafting the base vision tower back. | no (text-only data) | closed (Phase 3) |
| D5 | Sampling params pinned (temp 1.0, no penalties) vs card defaults | yes — pinned to paper values | decided |
| D6 | Loss-reduction weighting. Verified against source: ctm's `losses.ppo_loss` is a **global token mean** over the step's batch (`Σ mask·surrogate / Σ mask`, microbatches scaled by token fraction — engine.py `_loss_denominator`). slime's default is sum-of-sample-means ÷ `global_batch_size`, which with our `--global-batch-size 1` degenerates to an unnormalized sum over samples (weighting AND gradient scale differ). slime's `--calculate-per-token-loss` mode is exactly ctm's reduction (sum over masked tokens ÷ all-reduced step token count). Flag added to both launch scripts. | yes — was; eliminated by flag | closed (Phase 3) — `--calculate-per-token-loss` in run_dev_2b.sh and run_9b.sh |
| D7 | Learning rate: paper RMCT used constant 2.86e-4 with LoRA r8/α16 (confirmed in both repos — `experiment.yaml` grid and predecessor `paper_grpo.yaml`, which pins it). LoRA→full-param equivalence (~10–40× lower) gives ~1e-5; operator chose **1e-5 constant** (2026-08-06). | yes — LR is rate-affecting | closed — operator-approved 1e-5 in `scripts/run_9b.sh` |
| D8 | Data artifact: paper `wrong_argument` rows recovered from the predecessor repo (`dataset_dumps/test/distractor_argument_g4/{logiqa,hellaswag}_distractor_argument_g4.jsonl`, = wrong_argument with gemma-4 arguments). Converted schema-only (message lists byte-identical) by `scripts/convert_predecessor_pairs.py` with paper `load_datapoints` selection (first 32 rows/file → 64 datapoints); manifest records source SHA-256s. | no — same prompts, same selection | closed |
| D9 | LoRA arm: slime's Megatron backend has no LoRA training path (verified: only name-mapping/flops references). | n/a | superseded by D10 |
| D11 | Two upstream bugs found during Miles LoRA bring-up, with workarounds now in `run_miles.sh`: **(a)** Miles' mbridge base-weight export is broken for dense Qwen3.5 — bridge-mode `update_weights` pushes garbled base weights to SGLang (proved by full-param+bridge reproducing the identical garble; raw-mode export is correct). Workaround: `--lora-base-cpu-backup` enables Miles' `skip_base_sync` path — the frozen base is never pushed (SGLang serves the pristine HF checkpoint), only adapters flow via the maintained `export_adapter_weights`. Also a per-step throughput win. **(b)** SGLang's LoRA memory pool (`sglang-miles` tip cb05a44) allocates one buffer shape per module type across layers, breaking on Qwen3.5's heterogeneous hybrid layers (`LoRA buffer shape [10240,8] != weight [6144,8]` with a zero-init qkvo adapter). Workaround: LoRA targets **MLP-only** (`decoder.layers.*.mlp.linear_fc1/fc2`; uniform shapes, zero-adapter verified byte-identical to base) vs the paper's attention+MLP coverage — a training-capacity change. Revisit both when fixed upstream. | (a) no — pure bug workaround; (b) yes — adapter coverage change | recorded (2026-08-06); MLP-only LoRA + skip-base-sync is the working arm |
| D12 | Data selection: the predecessor source files contain rows whose gemma argument generation failed (empty `biased_question`: 324/2400 logiqa, 71/2400 hellaswag). The paper loader (`train_rl.py load_datapoints`) fed them anyway — 5 of its first-32 logiqa datapoints had NO biased prompt, contributing no gap signal. Empty prompts are not executable here; the converter now takes the first 32 rows per file with both prompts non-empty (9 logiqa rows skipped, manifest records it). Effective usable-datapoint mix therefore differs from the paper's (ours: 64 live; paper's: 59 live + 5 dead). | yes — datapoint mix shifts measured rates | recorded (2026-08-06); artifact + manifest rebuilt |
| D10 | Run tier switched slime → **Miles** (radixark/miles, production fork of slime linked from slime's README; image `radixark/miles:latest-cu12`). Motivation: native LoRA (`--lora-rank 8 --lora-alpha 16 --lora-dropout 0.0` = the paper's exact adapter config, partially reverting D1) plus Qwen3.5 GatedDeltaNet support and the same `--rollout-function-path` hook. Miles lacks slime's `--custom-advantage-function-path`; the parity-tested RMCT advantage fn is installed via `--custom-megatron-init-path slime_port.miles_init.megatron_init` (explicit rebind, loud failure if the target moves). All parity gates re-run on Miles before science use. Full-param remains available as a benchmark arm. | potentially (new runtime; LoRA restores paper adapter) | in progress (Phase 4) |
