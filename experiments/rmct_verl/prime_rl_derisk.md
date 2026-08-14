# prime-rl derisk assessment (2026-08-14)

Question: would PrimeIntellect's prime-rl (github.com/PrimeIntellect-ai/prime-rl,
assessed @ 68e0c50 / v0.8.0) work as the RMCT training stack, now that we are
moving to dense ~8B models? Method: two research agents (source fit + field
evidence), no GPU spend.

## Verdict in one line

Yes, it would work — it is architecturally the best fit of any framework
examined — but it will NOT materially beat the verl recipe's step time, and
adopting it means a (small, surgical) fork plus taking on its least-exercised
configuration. Decision should follow a $0–5 measurement, not architecture.

## Why RMCT fits unusually well

- Environment layer (verifiers v1): an env mints its own Tasks, so sampling
  BOTH prompt variants per datapoint is native (user_sim env is a two-agent
  template); `Trace.info` carries variant/parse_ok/trait; every rollout is
  persisted unconditionally (`rollouts/step_N/.../traces.jsonl`).
- `Algorithm.score_group` receives the whole finalized cohort — exactly where
  p_ref/p_hat across the two populations belongs. Per-token advantages are the
  native currency; frozen-base logprobs are already plumbed (`OPDAlgorithm.
  score_rollout` is a 6-line template); no built-in reward-KL to fight.
- Zero-loss semantics: assign advantage 0.0 + the stock `ZeroAdvantageFilter`
  (do NOT use `trainable=False` — untrainable traces are dropped before
  finalize and would break rate measurement).
- Neat simplification: (p_hat − p_ref) is constant within a group, so plain
  GRPO mean-centering of reward = −(p_hat−p_ref)·trait reproduces the RMCT
  advantage exactly; the custom algorithm is mostly zero-loss cases + KL fold.
- Port estimate: ~350–450 LOC. Env lives OUTSIDE the repo; the fork is three
  files (`orchestrator/algo/rmct.py`, config union entry, registry line).
  Trainer untouched. Custom credit assignment has no config hook (confirmed:
  docs say a new scheme is "a new named algorithm in the repo") — so a fork
  is unavoidable, but it is this small.
- Ops niceties we currently hand-roll: unconditional rollout persistence,
  per-token debug export (advantages, importance ratios, infer-vs-trainer
  logprobs — a built-in Gate-B surface), CPU-only unit-test path, single-node
  1+1-GPU default deployment, LoRA with hot adapter reload into vLLM
  (`/load_lora_adapter`, no merge pass).

## Why it will probably not be faster

prime-rl's async is a FIXED one-step overlap achieved by disaggregating GPUs
(separate inference and trainer GPUs), not by hiding generation behind
training on shared GPUs. Under an optimally sized split, throughput equals
colocated sync: max(T_gen/g_i, T_train/g_t) at optimum = (T_gen+T_train)/G.
What it actually buys: no per-step vLLM sleep/wake + KV teardown + weight
resharding (in our measured verl step that overhead is ~14s of 2,197s), and
straggler tolerance (long rollouts span weight updates instead of blocking —
relevant at 20k-token tails). Neither rescues a generation-dominated step.

## Concentrated risks (field evidence)

- LoRA forces weight broadcast onto the FILESYSTEM path every step — the
  transport with the worst bug history (incl. a 3-month reward-collapse
  investigation correlated with it; a broadcast-failure bug that marks
  checkpoints STABLE anyway is open with pending fix).
- "Single-node dense 8B + LoRA + custom algorithm" is close to the least-
  exercised corner: zero dense-8B examples in-repo (all showcase configs are
  MoE or ≤4B toys); the seq_len-vs-max_tokens sizing footgun is documented
  in issues.
- vLLM 0.26 pinned by wheel URL and patched deeply (own NaN-logprob incident
  from their fp32 lm-head patch); upgrades are a project.
- Open, unguarded importance-ratio overflow on stale rollouts (0·inf=nan).
- Async depth not tunable (TARGET_LAG=1 hardcoded); maintainers argue this is
  correct, and they are exceptionally responsive (real mitigating factor).
- Install: uv-only, Linux/apt-only, SSH-auth git submodules.

## Positives specific to our new direction

Moving off Qwen3.5 removes prime-rl's whole hybrid-attention bug class
(their GDN/Nemotron packed-state-leak bugs mirror what we fled). Dense Qwen3
and Llama have first-class custom trainer impls; new-model PRs must ship a
KL-mismatch table < 0.015 — the hygiene we want.

## Decision procedure (recommended, ~$0–5)

1. 1-GPU trainer-only bench, no inference server needed:
   `uv run trainer @ <cfg> --data.fake --bench --bench.output-json` at real
   seq_len/LoRA rank, with `model.ac="None"`, `model.ac_offloading="None"`
   (AC defaults ON at ~25% cost — naive benchmarks understate prime-rl).
   Compare against the training-phase share of verl's measured step.
2. Same session: verify one vLLM serves frozen-BASE logprobs alongside the
   loaded adapter (inference from serving_tokens.py model-name dispatch;
   unverified — if false, the ref model costs a second deployment).
3. If (1) shows verl's training phase carries meaningful non-compute overhead
   or (2) confirms the free ref endpoint → port (days-scale, small fork).
   Otherwise stay on the verl recipe and simply re-point it at the dense 8B.

## Status relative to existing stacks

| | Miles/slime | verl (our recipe) | prime-rl |
|---|---|---|---|
| RMCT expressible | ported, parity-verified | ported, parity-verified, GPU-proven | port ~450 LOC + 3-file fork |
| LoRA on dense 8B | should work unpatched (untested) | works (measured at 9B) | works (CI-covered), filesystem sync |
| async overlap | no (task #10 open) | no (v1 port needed) | native, fixed 1-step, disaggregated |
| dependency freshness | research fork | pinned 2b0fe51, v0 deprecated | freshest (vLLM 0.26, transformers 5.6) |
