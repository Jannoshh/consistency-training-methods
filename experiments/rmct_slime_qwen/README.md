# RMCT on slime (SGLang + Megatron), Qwen3.5

Migration of the rate-matching consistency training loop onto slime for
wall-clock speed, per `AGENT_PROMPT.md`. Phase log and contradiction record:
`phase0_report.md`.

## Layout

- `slime_port/` — the ported RMCT math and slime extension modules:
  - `rewards.py`, `advantages.py` — vendored verbatim from ctm @ the commit in
    `slime_port/__init__.py` (import line only changed);
  - `pipeline.py` — port of `RLTrainer._build_training_batch`'s math;
  - `rmct_rollout.py` — slime `--rollout-function-path` entry point;
  - `rmct_advantage.py` — slime `--custom-advantage-function-path` entry point
    (advantage broadcast + tinker-semantics centered KL vs the frozen base);
  - `rollout_writer.py` — ctm-schema `step_*.jsonl.zst` rollout persistence;
  - `rmct_config.py` — run config, loaded from `$RMCT_CONFIG` (JSON).
- `tests/` — Gate A parity suite. Synthetic parity always runs
  (`uv run python -m pytest experiments/rmct_slime_qwen/tests`); replay parity
  activates with `RMCT_ROLLOUT_DIR=<run>/rollouts`.
- `configs/dev_smoke.json` — Phase 3 bring-up config (not science settings).
- `preflight.sh` — instance validation; first thing run on every pod
  (`--dev` for the single-GPU tier).
- `scripts/sync_to_pod.sh`, `scripts/setup_pod.sh` — bundle shipping and pod
  preparation (writes `env.lock`).
- `scripts/derive_model_args.py` — Megatron MODEL_ARGS from a checkpoint's
  `config.json`.
- `scripts/run_dev_2b.sh` — dev-tier launcher (PHASE=rollout|train|full).
- `restart.sh` — cold-start recovery on a fresh pod.

## Dev-tier workflow (Phase 3)

```bash
# locally
runpodctl create pod ...            # per approved spec; see phase log
./scripts/sync_to_pod.sh <host> <port>
ssh -p <port> root@<host> 'cd /workspace/rmct && bash scripts/setup_pod.sh'
# on the pod, inside tmux
PHASE=rollout bash scripts/run_dev_2b.sh   # rollout path in isolation
PHASE=train   bash scripts/run_dev_2b.sh   # train path from saved rollout data
PHASE=full    bash scripts/run_dev_2b.sh   # 5-step loop, then kill/resume test
```

Gate A replay against a slime run: copy the run's `rollouts/` dir locally,
then `RMCT_ROLLOUT_DIR=... uv run python -m pytest
experiments/rmct_slime_qwen/tests -k replay`.

## Data

`data/dev_smoke/` (gitignored) is a template-bias fixture built with
`ctm_data.adapters.mcq_bias.materialize` (`suggested_answer`, no LLM calls) —
bring-up only. Science runs use the frozen `wrong_argument` artifacts per the
repository's frozen-artifact rules; the 9B experiment config is written in
Phase 3/4 and shown before any paid training step.

## Phase 3 results (2026-08-05, dev pod, H100 80GB, Qwen3.5-2B)

- **Rollout-only**: 5 steps; prefix cache hit 0.88–0.92 after warmup;
  parse_rate 0.34–0.58 (1024-token budget truncates the thinking model —
  smoke-only, the science budget is 20480).
- **Train-only** (replayed rollout data): RUN_EXIT=0, pg_loss nonzero,
  grad_norm 3.2–5.0, `rmct/kl_policy_base` ≈ 0.001.
- **Full synchronous loop**: RUN_EXIT=0. On-policy after each optimizer step,
  `train/train_rollout_logprob_abs_diff` = 0.0183 / 0.0183 / 0.0175
  (Gate B statistic on trained weights; slime's reference is ~0.011 on
  Qwen3-4B — same order, stable across steps, no growth). One step's batch
  was correctly removed end-to-end for zero signal (all-skipped → zeroed
  train step). Cache hit drops to ~0.06 in the full loop because weight
  updates invalidate SGLang's prefix cache each step — expected.
- **Gate A replay on the full loop's own records**: 38/38 parity tests pass —
  every persisted reward, p_hat, p_ref, and advantage reproduced bitwise by
  the original ctm `_build_training_batch`.
- Off-by-one found and fixed: the base `_torch_dist` conversion carries
  `latest_checkpointed_iteration = 1`, so fresh runs silently skipped
  generation 0 (the "5-rollout" smoke ran 4 generations). Launch scripts now
  pin `--start-rollout-id 0` on fresh runs.
- **Kill/resume test: passed** (after fixing two real bugs it exposed).
  Round 1: relaunch restarted from scratch — `--load` never pointed at the
  run's own checkpoints — and the rollout writer crashed with
  `FileExistsError` on the regenerated step. Fixes: launch scripts load
  `${RUN_DIR}/checkpoints` when `latest_checkpointed_iteration.txt` exists;
  the writer supersedes a regenerated step (`*.superseded-N`, index entry
  moved to a `superseded` key) so replay sees only the trained sequence
  while both attempts stay on disk. Round 2: hard-killed (`pkill -9` +
  `ray stop --force`) after checkpoint iter 3; relaunch loaded iter 3, ran
  only the remaining generation (on-policy from resumed weights, with
  `--calculate-per-token-loss` active), saved iter 4, RUN_EXIT=0. Gate A
  replay over the killed+resumed run's records: 38/38.
- Loss reduction (deviation D6): ctm's `ppo_loss` is a global token mean;
  slime's default sum-of-sample-means ÷ `global_batch_size` (=1 here) is an
  unnormalized sum. `--calculate-per-token-loss` matches ctm exactly and is
  now in both launch scripts.

## Phase 4 results (2026-08-06, 1× H200, Miles @ c9e79e3)

### Parity on Miles (Qwen3.5-4B, full-parameter)
- **Gate A**: 38/38 — replay of the Miles run's own rollout records reproduces
  every reward/p_hat/p_ref/advantage bitwise in the original ctm math. Also
  passes on the LoRA run's records (reward channel is arm-independent).
- **Gate B**: `train_rollout_logprob_abs_diff = 0.0107` on a real-signal
  optimizer step (slime Phase 3 reference: ~0.018; slime upstream reference on
  Qwen3-4B: ~0.011). pg_loss −0.0124, grad_norm 0.89 — healthy.
- Zero-signal batches (degenerate smoke statistics) are correctly no-op'd
  end-to-end via the zero-mask fallback (exact ctm skip semantics; Miles
  cannot digest empty batches).

### LoRA arm status: blocked upstream (see D11)
Sampling, serving, reward math, and adapter routing all verified; but the
Megatron-side actor forward under the LoRA config (bshd + GDN) recomputes
logprobs ~11.8 nats off the sampling engine and produces zero grad_norm —
training through it would be garbage. Full-param + bshd cannot even run
(`hf_attention.py` asserts packed_seq_params). Wait for upstream fixes to
Miles' dense-Qwen3.5 GDN paths before using the LoRA arm.

### 9B rollout throughput (1× H200, standalone SGLang, real wrong_argument prompts)
| config | tok/s (out) | notes |
|---|---|---|
| bf16 baseline, 256 concurrent | 6,451 | prefix cache 0.70, parse 0.71 |
| + MTP NEXTN-3 | 5,715 | **slower** — spec decode loses at deep batch |
| FP8 KV cache, 256 concurrent | 6,968 | +8% over bf16 |
| FP8 KV, full science fan-out (1,024) | **7,364** | prefix cache **0.913**, parse 0.79; one science step's generation ≈ 12 min on ONE H200 |

Completion p50 was 5,542 tokens with 29% truncation at an 8,192 budget —
the paper's 20,480 budget is genuinely needed (truncation → parse failure →
biased p_hat). MTP is now moot for this workload (helps latency at small
batch, not throughput at 128-way fan-out), which also dissolves the
LoRA-vs-MTP serving conflict.

## Phase 5 shakeout (2026-08-06, 4× H200, full-param 9B, real science shape)

2-step runs at the exact paper shape (4 datapoints × 128+128 rollouts,
max_new_tokens 20480, real wrong_argument data), TP2 × DP2 training + 4
SGLang engines:

| | bf16 | FP8 KV |
|---|---|---|
| warm step time | **234s** | 228s |
| generation wait | 150s | 140s |
| actor train | 55s | — |
| Gate B abs_diff | **0.0117** | 0.0225 |
| parse_rate | 0.998–1.000 | 0.998–1.000 |
| bias gap | 0.21–0.29 | 0.23–0.25 |

- **Gate B holds at 9B science scale** (0.0117 ≈ upstream's 0.011 reference).
  Gate A replay of these records: 38/38.
- **FP8 KV rejected for science runs**: doubles sampling↔training divergence
  for ~3% wall-clock — the full loop is not KV-bandwidth-bound.
- **Real-data parse_rate ≈ 1.0** — the low smoke parse rates were fixture
  artifacts; at 20480 tokens virtually every rollout reaches an answer, so
  ~all 512 samples per step carry gradient.
- **Run-cost model (measured)**: 100 steps ≈ 6.5 h ≈ $120 on 4× H200
  synchronous; ~$90 on 8 GPUs; async overlap (Phase 5 remainder) targets
  the 150s generation wait, bounding step time near max(gen, train).
- Batching redesign shipped in this shakeout: the rollout fn emits exactly
  rollout_batch × n_train samples per generation (skipped rollouts get
  zero loss masks = exact ctm skip semantics), `--global-batch-size` equals
  that count — works at any DP size, retires the shared-rollout_id hack.
- New Miles bug (Phase 6 blocker): end-of-run distributed checkpoint save
  fails at TP2×DP2 ("rank args Namespace mismatch" in save validation);
  benchmark runs use NOSAVE=1. Needs a fix or workaround before long runs
  (checkpoint-every-8 is a science requirement).
