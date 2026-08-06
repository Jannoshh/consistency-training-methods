# Task: Stand up fast RMCT (rate-matching consistency training) on rented GPUs using slime

You are migrating RMCT — the GRPO rate-matching loop in this repository — from its
current backends (Tinker / LocalBackend) onto slime (SGLang rollout + Megatron
training) to improve wall-clock training speed. Work through the phases in order.
Each phase has a gate — do not proceed past a gate that hasn't passed.

## What RMCT actually is (read the code, not this summary)

Source of truth: `ctm/training/rl.py` (loop), `ctm/core/rewards.py` (reward),
`ctm/core/advantages.py` (advantage estimators), and the resolved config in
`experiments/rmct_paper_vast_more_methods/experiment.yaml`. In brief:

- Each datapoint has K perturbation prompts: index 0 is the neutral reference,
  the rest are biased cues. Per step, the trainer samples ~128 rollouts on the
  neutral prompt (→ `p_ref`) and ~128 per cue (→ `p_hat`), classifies each
  response with a trait classifier (answer parser or LLM judge; `None` =
  abstention), and rewards `r = -(p_hat - p_ref) * (trait - baseline)`.
- A separate anchor term pins reference-prompt behavior to the *initial base
  model's* rates, measured once at startup by sampling from the frozen base.
- `kl_coef` defaults to 0.05 (paper config uses it) — there IS a reference-model
  logprob term. Loss is PPO-style or importance sampling.
- Unparsed rollouts are discarded or resampled per `unparsed_handling`; parse
  rate is a monitored signal, not noise.
- Paper config: temperature 1.0, `max_new_tokens` 20480 (reasoning model),
  `advantage_estimator: grpo_normalized`, LoRA rank 8.

**Consequences that drive every optimization decision below:**

1. The workload is overwhelmingly rollout-bound: ~256+ rollouts per datapoint
   at up to 20k tokens, with gradients on only a subset. Sampling + judging is
   >90% of wall-clock. SGLang throughput, prefix caching, speculative decoding,
   and async scheduling are where the speed lives; Megatron sharding is almost
   irrelevant by comparison.
2. All 128 rollouts in a group share the *entire prompt*, and the neutral/cued
   prompts within an item share long prefixes. SGLang's radix/prefix cache must
   be on and its hit rate measured — this is close to free 2x+ on prefill.
3. Anything that changes the *rates* changes the science: temperature, length
   truncation (a truncated response becomes a parse failure and distorts
   `p_hat`), sampling staleness under async, FP8 sampling drift. Every such knob
   is an experiment change and must be recorded, not silently tuned.

## Decisions already made

- **Framework:** slime (`https://github.com/THUDM/slime`). Rollout via SGLang,
  training via Megatron-LM. Do **not** substitute vLLM — slime is deliberately
  SGLang-only.
- **RMCT math is ported, never reimplemented.** `ctm/core/rewards.py` and
  `ctm/core/advantages.py` are backend-free pure math — import or copy them
  verbatim into slime's custom rollout/reward path, along with the setting's
  perturbation functions and trait classifier. slime owns its own driver loop,
  so this is a standalone fast path with a custom rollout function, not a new
  `TrainingBackend` implementation.
- **Policy model:** Qwen3.5-9B (dense ~9.65B, hybrid Gated DeltaNet + gated
  full attention ~1:4, natively multimodal, 262k context), trained
  **full-parameter**. Text-only RL; freeze or strip the vision tower. This is a
  deliberate departure from the repo's gpt-oss-20b rank-8 LoRA paper runs — the
  rate-matching method, data, and provenance rules carry over unchanged, the
  policy model and training regime do not. Do not add LoRA "for parity".
- **Data and setting:** the mcq-bias sycophancy setting, reusing the frozen
  training artifacts and perturbation functions from
  `experiments/rmct_paper_vast_more_methods/` (wrong_argument bias on
  LogiQA/HellaSwag; trait classifier is the answer parser — no API judge in the
  hot loop). Rollout counts, temperature, and generation limits start from that
  experiment's `rate_matching` block. There is no Qwen3.5 experiment YAML yet —
  writing one (new artifact paths per the frozen-artifact rules, never
  overwriting the gpt-oss ones) is part of Phase 3, and it must be shown to the
  operator before the first paid training step.
- **Hardware:** rented — RunPod first, vast.ai as fallback. GPU count/model not
  fixed — selecting them is part of your job. On RunPod prefer Secure Cloud for
  the run tier (Community Cloud instances vary wildly in interconnect and can
  vanish like spot capacity); a Community pod is fine for the dev tier. Use a
  RunPod network volume for checkpoints/artifacts so pod death doesn't take the
  data — but still treat object storage as the durable copy (Phase 6).

## Execution context

- The CTM repository lives at `~/dev/Coding/consistency-training-methods`
  (branch with the latest backend speedups: `training-profiling-rebase`).
  Its CLAUDE.md rules apply, most importantly: **show every training/eval
  command with resolved parameters and get explicit approval before any paid
  or remote run.** Pod rentals count — present the instance type, $/hr, and
  expected duration before renting.
- Pod lifecycle goes through `runpodctl` (installed and authenticated) or the
  RunPod console; `RUNPOD_API_KEY` is in the repo's `.env`. Long-running work
  runs under `tmux` over SSH — the remote-kernels MCP is for interactive
  debugging only, not for hosting training.
- Deliverables (scripts, `env.lock`, reports) live in the repo under a new
  `experiments/rmct_slime_qwen/` directory unless the operator says otherwise.
- If a needed API key is missing from `.env`, stop and ask — do not create
  accounts or new keys.

## Phase 0 — Verify, don't assume

Your priors about these moving parts are stale. Before writing any config, read:

- slime README + docs: quick start, customization (especially the custom
  rollout-function / reward extension points), fault tolerance, reproducibility,
  debugging, `sglang-config`, `pd-disaggregation`, `delta-weight-sync`
- Whether slime supports **externally computed per-sample advantages** (RMCT's
  advantage estimators are custom — `grpo_normalized`/`snr_scaling`/
  `matched_pair` — and must be computed by the ported ctm code, not slime's
  built-in GRPO grouping). If slime insists on computing advantages itself,
  find the override point now, not in Phase 3.
- Qwen3.5-9B model card: architecture details, recommended SGLang launch flags
- SGLang docs: MTP/NEXTN speculative decoding, FP8 KV cache, radix cache
  behavior with very large same-prefix fan-out (128 rollouts/prompt)
- Whether slime's Qwen3.5 support covers hybrid DeltaNet layers on both the
  Megatron and SGLang side (Qwen3Next support is a useful proxy)

**Gate:** report anything contradicting this document's assumptions before
proceeding.

## Phase 1 — Hardware selection

Two-tier. Do not debug the pipeline on expensive hardware.

**Dev tier (bring-up):** 1× 80GB+ GPU (H100/H200/A100-80G), running
**Qwen3.5-2B** — same architecture family, every code path exercised cheaply.

**Run tier:** 4–8× H200 SXM or H100 SXM, or B200 when the price-per-token wins.
RunPod list prices as of 2026-08 (re-check, they move): H100 SXM $3.29/hr
(High availability), H200 SXM $4.59/hr, 141GB (Medium), B200 $6.79/hr, 180GB
(Low availability, **max 5 GPUs per node**). B200 at ~2.1× H100's price buys
~2–2.5× its throughput plus 180GB VRAM — roughly break-even on price-per-token
and clearly ahead when memory binds, but Low availability means slow re-rents
after a pod dies and the 5-GPU cap rules out 8× configs. Verify the slime image
supports sm_100 (CUDA ≥ 12.8) before committing; fall back to H200/H100 if the
stack fights you — debugging kernel support on a new architecture is not this
project's job. (RTX PRO 6000 at $2.09/hr with 96GB is notable for the dev tier:
same Blackwell generation as B200, so it exercises the sm_1xx kernel path
cheaply — but PCIe only, so never for the multi-GPU run tier.) Because RMCT is
rollout-bound, bias GPU allocation toward inference: with PD disaggregation or
async, expect most GPUs serving SGLang and a minority training. 4× H200 or
4× B200 is the floor for full-parameter 9B.

**Hard requirements — verify before any long rental:**

- **NVLink/NVSwitch:** `nvidia-smi topo -m` must show `NV#` between GPUs.
  `PHB`/`SYS`/`PIX` across the board = PCIe riser node — destroy and re-rent.
- Disk: 3–4× checkpoint size free (Megatron 9B checkpoints run ~40–150GB),
  **plus** headroom for rollout logs — RMCT persists every sampled response as
  compressed JSONL by default, and at 256 × 20k-token rollouts per item this is
  not small.
- CUDA driver compatible with the slime image.
- Outbound network to Hugging Face, your object storage, **and the judge
  endpoint** (if the trait classifier is an API judge, e.g. OpenRouter) — test
  latency and rate limits from the actual instance.

**Deliverable:** `preflight.sh` checking all of the above, exits non-zero on
failure. First thing run on every new instance.

## Phase 2 — Environment

Use slime's official Docker image. Do **not** build the Megatron/SGLang/Ray
stack from source. Pin everything. **Deliverable:** `env.lock` — image digest,
`pip freeze`, driver/CUDA versions, plus the ctm repo commit the ported reward
code was taken from.

## Phase 3 — Dev-tier bring-up (Qwen3.5-2B, 1 GPU)

Get the complete loop executing small:

1. Checkpoint conversion (HF → Megatron) and back
2. Rollout-only path in isolation — including the RMCT custom rollout function
   with a toy config (a few datapoints, `n_rollouts` 8) and the trait
   classifier wired in
3. Train-only path in isolation
4. Full synchronous loop, 5 steps
5. Kill mid-run, resume from checkpoint

**Gate A — reward parity (RMCT-specific, run first, no GPU needed).** Replay
rollouts from an existing Tinker/LocalBackend RMCT run's rollout JSONL
(`rollout_log: all`) through the ported reward + advantage path. This is pure
math on identical inputs: rewards and advantages must match the original run to
float tolerance. If they don't, the port is wrong — fix before touching slime.

**Gate B — logprob parity.** The most important check in the project. Use
slime's rollout-then-train replay path to compare SGLang sampling logprobs
against Megatron training-forward logprobs on identical sequences. Report max
abs diff, mean abs diff, and KL. If out of tolerance, **stop and fix**. A
silent train/inference mismatch produces healthy-looking curves that never
improve — and hybrid linear attention is exactly where non-standard kernels
break parity. This project has already been bitten by this bug class once: a
rollout engine silently served base weights instead of the trained adapter, so
every method sampled from the same model. Parity must be checked against the
*trained* weights after at least one optimizer step, not just at initialization
(where policy and base coincide and the check proves nothing).

## Phase 4 — Scale to 9B on the run tier

Starting configuration — hypotheses to validate against measured memory and
throughput, not settings to copy:

**Megatron**
- `--tensor-model-parallel-size 2` with NVLink, `1` without; PP 1;
  DP = world / TP
- Distributed optimizer **on** (shards the ~154GB full-param AdamW footprint);
  gradient checkpointing on; 8-bit optimizer states if still tight.

**SGLang** (flags prefixed `--sglang-` through slime)
- Radix/prefix cache on; **measure hit rate** — with 128-rollout groups it
  should be very high, and a low number means misconfiguration.
- MTP speculative decoding: NEXTN, 3 steps, eagle-topk 1, 4 draft tokens —
  near-free 1.5–2.5x decode if MTP was trained into the model.
- FP8 weights / FP8 KV cache — but see the parity gate below; FP8 sampling
  drift feeds directly into rate estimates.
- `mem-fraction-static` tuned to measured free memory.

**RMCT loop (replaces generic GRPO settings)**
- Groups are per-(item, cue) rollout populations, not standard GRPO groups.
  Group sizes, rollout counts, temperature, and `max_new_tokens` come from the
  experiment YAML — do **not** cut generation length for speed: truncation
  turns responses into parse failures and biases `p_hat`. Instead, measure the
  completion-length distribution and set the budget above p99.
- The **anchor** needs one-time base-model rate measurement at startup: sample
  the reference rollouts from the frozen base, cache the resulting rates to
  disk keyed by config hash so instance loss doesn't repeat the cost.
- `kl_coef` per experiment config (paper: 0.05). This requires reference-model
  logprobs — do not drop it "for speed"; that is an experiment change.
- Judge/parse path: bound classifier concurrency explicitly; an API judge is a
  second throughput ceiling with its own tail.

**Gate:** re-run logprob parity at 9B, again with FP8 enabled. FP8 widens the
gap; if divergence is material, add truncated importance-sampling correction
rather than ignoring it — and record the deviation, because sampling-policy
drift shifts measured rates.

## Phase 5 — Async and tail elimination

This is the big win: with 20k-token reasoning rollouts, long-tail generation
dominates step time. Move to `train_async.py` / `examples/fully_async`, and
consider APRIL-style active partial rollout management for the tail.

Two RMCT-specific constraints:

- The current loop's contract is fully on-policy
  (`refresh_policy_every_n_steps: 1`). Async introduces staleness between the
  sampling policy and the trained policy — use the importance-sampling loss
  path and record the staleness (steps between weight syncs) as an explicit
  deviation. Compare a short sync vs async run's training curves before
  committing.
- Partial-rollout schemes must respect `unparsed_handling`: an aborted rollout
  is a parse failure, and systematically aborting slow rollouts selects against
  long responses, which biases rates. Resumption (continue the partial rollout
  after weight sync) is acceptable; truncation-as-completion is not.

Measure end-to-end throughput before and after.

## Phase 6 — Durability

Rented instances disappear without warning; assume mid-run.

- Checkpoint every N steps to external object storage (S3/R2), never only
  instance disk — and ship the rollout JSONLs and manifests too; they are the
  run's provenance record.
- Test resume by actually killing the job.
- `restart.sh`: rebuild a working environment from `env.lock` on a fresh host
  in under 15 minutes (including the cached base-model anchor rates).
- Read slime's fault-tolerance docs before the first long run.

## Instrumentation (required from Phase 3 onward)

Per step:

- Rollout tok/s, training tok/s, effective end-to-end tok/s
- Wall-clock split: rollout / **judge-and-parse** / train / weight-sync / idle
- Fraction of sampled tokens that are rate-only vs gradient-bearing
- Prefix-cache hit rate
- GPU utilization per phase; peak memory per GPU
- Straggler ratio: p99/p50 sequence completion time within a rollout batch
- Parse rate and (if resampling) resample amplification / give-up counts
- Judge latency p50/p99 and in-flight concurrency
- Logprob divergence sampler vs trainer

Rough decode-throughput baselines for orientation (generic sanity checks, not
RMCT targets — RMCT step time is dominated by the rate-sampling fan-out):

| Configuration | Effective end-to-end |
|---|---|
| 8× H200, MTP + FP8 + async | 40–60k tok/s |
| 8× H100, MTP + FP8 + async | 30–45k tok/s |
| 4× H200, MTP + FP8 + async | 20–30k tok/s |
| Synchronous, no spec decode | ~1/3 of the above |

More than 2x below the relevant line: profile (slime ships a trace viewer)
before tuning.

## Rules of engagement

- No silent deviations — state what changed and why. **Anything that can move
  measured rates (temperature, truncation, staleness, FP8 drift, judge model)
  is a science change, not a tuning knob**, and goes in the deviation record.
- Never skip a parity gate to make progress. A fast run with a broken gradient
  is worse than no run.
- Prefer reading documentation over guessing flag names; version drift is
  constant in this stack.
- Report blockers rather than masking them with workarounds.
- Report measured numbers only; if you didn't measure it, say so.

## Deliverables

1. `preflight.sh` — hardware, environment, and judge-endpoint validation
2. `env.lock` — pinned environment incl. ctm source commit of the ported math
3. `restart.sh` — cold-start recovery on a fresh host
4. The RMCT custom rollout/reward module for slime, with the reward-parity
   replay test
5. Launch scripts for dev tier (2B, 1 GPU) and run tier (9B, N GPUs)
6. Parity reports: reward parity (Gate A) and logprob parity at each gate
7. Throughput report with the full metric set, before/after async
8. Written record of every deviation from this plan — including every
   rate-affecting knob — and its justification
