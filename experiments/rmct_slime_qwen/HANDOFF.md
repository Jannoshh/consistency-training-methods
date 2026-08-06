# RMCT → slime/Miles migration — handoff (2026-08-06)

State of the effort to move rate-matching consistency training (RMCT) onto a
SGLang+Megatron RL stack for wall-clock speed, per `AGENT_PROMPT.md`. Read
this alongside `phase0_report.md` (deviation table D1–D12) and `README.md`
(per-phase measured results).

## TL;DR

- **The RMCT math is fully ported and parity-verified on two frameworks**
  (slime and its Miles fork). Gate A (reward parity vs original ctm code,
  bitwise): passes everywhere, 38/38, including on real runs' own rollout
  records. Gate B (sampling↔training logprob agreement on trained weights):
  0.018 on slime @ 2B, **0.0107 on Miles @ 4B full-param** — both healthy.
- **Full-parameter training on Miles is the working, verified arm.**
- **LoRA on Miles is blocked by an upstream bug** (Megatron-side actor
  forward under the LoRA config recomputes logprobs ~11.8 nats off, zero
  grad_norm). Everything else on the LoRA path was fixed and verified.
- **9B rollout throughput is measured**: 7,364 tok/s on one H200 at the real
  science fan-out with FP8 KV (+8% over bf16), prefix cache 0.913. MTP
  speculative decoding is a measured LOSS at this batch depth — dropped.
- One science step's generation ≈ 12 min on one H200 → a 16-step paper-shape
  9B run is ~50 min generation on 8 DP engines + training.

## Where things run

- Pod (STOPPED, volume retained): RunPod `68inz1c9naspzg` "rmct-miles-dev",
  1× H200, image `radixark/miles:latest-cu12`, $4.59/hr. `/workspace` (network
  volume) holds the synced experiment dir + rollout logs; `/root` (local NVMe,
  dies with the pod) held models — `setup_pod.sh` rebuilds them in ~20 min.
- RunPod API key: switched 2026-08-06 to the `rpa_6MOB…` key from `.env`
  (earlier pods were accidentally on a different account and are gone).
- Resume recipe: start pod (or create new with the sshd `--args` from
  `README.md`/git history — the Miles image has no RunPod entrypoint) →
  `scripts/sync_to_pod.sh <ip> <port>` → `MODEL=Qwen/Qwen3.5-4B bash
  scripts/setup_pod.sh` (and 9B) → `scripts/run_miles.sh`.

## What was verified, in order

1. **Phase 3 (slime, Qwen3.5-2B, 1× H100)**: full loop, Gate A+B, hard
   kill/resume, rollout persistence in ctm schema. Four real bugs found and
   fixed (resume load path, skipped generation 0, writer overwrite on
   regenerated steps, D6 loss-reduction mismatch → `--calculate-per-token-loss`).
2. **Phase 4 (Miles, Qwen3.5-4B/9B, 1× H200)**: framework switched for LoRA
   support (D10). Full-param loop parity-verified (Gates A+B). LoRA blocked
   upstream (D11). Throughput baseline measured (README table).

## Upstream bugs found (candidates for filing against radixark/miles & sglang-miles)

1. **mbridge dense-Qwen3.5 base export garbles weights** (bridge-mode
   `update_weights` pushes corrupted base weights; raw-mode export is fine).
   Repro: full-param + `--megatron-to-hf-mode bridge` → deterministic garbage
   generation. Workaround in `run_miles.sh`: `--lora-base-cpu-backup` (skips
   base push entirely; SGLang keeps the pristine HF checkpoint).
2. **SGLang LoRA memory pool assumes uniform module shapes across layers** —
   breaks on hybrid GDN/attention models (`LoRA buffer shape [10240,8] !=
   weight [6144,8]`). Workaround: MLP-only target modules.
3. **Megatron-side LoRA (bshd+GDN) actor forward is wrong**: logprob recompute
   ~11.8 nats off SGLang's sampled logprobs, grad_norm 0. No workaround —
   this is what blocks the LoRA arm. (Full-param+bshd can't run either:
   `hf_attention.py` asserts packed_seq_params.)

## Key file map (all under `experiments/rmct_slime_qwen/`)

- `slime_port/` — ported RMCT math (Gate-A-tested `pipeline.py`, vendored
  `rewards.py`/`advantages.py`), rollout fn, advantage fn (tinker-semantics
  KL, DP-global centering), Miles init hook (`miles_init.py` — installs the
  advantage fn since Miles dropped slime's flag), `framework.py` compat shim,
  rollout writer (supersede-on-resume).
- `scripts/run_miles.sh` — the run-tier launcher (all workarounds encoded,
  commented). `run_dev_2b.sh` is the slime dev-tier equivalent.
- `scripts/bench_sglang.py` — standalone throughput bench with the exact RMCT
  request shape; results in `logs_miles_pod/bench_results.jsonl`.
- `scripts/convert_predecessor_pairs.py` — paper data recovery from
  `/Users/jannes/dev/MATS/rmct/dataset_dumps/test/distractor_argument_g4/`
  (D8/D12); artifact + SHA-256 manifest in `data/run_9b/` (gitignored, rebuild
  with the command in the manifest).
- `configs/run_9b.json` — science config (paper-frozen values, LR 1e-5 per D7).
- `tests/` — Gate A suite; `RMCT_ROLLOUT_DIR=<rollouts> pytest` replays any run.
- `env.lock` — Miles pod environment (miles c9e79e3, sglang-miles cb05a44,
  torch 2.11.0+cu129).

## Open decisions / next steps

1. ~~Multi-GPU shakeout~~ **DONE** (4× H200, README Phase 5 section): Gate A+B
   hold at 9B science scale; 234s/step warm; FP8 KV rejected (2× parity cost,
   ~3% speed); measured cost model: 100 steps ≈ $120 on 4× H200 sync.
2. **Long run**: needs (a) what 100 steps means — 6 epochs over the paper's 64
   datapoints vs 1 epoch over ~400 converted rows (both recorded science
   changes) — and (b) the Miles end-of-run save crash fixed first (below).
3. **NEW Miles bug (blocks long runs)**: distributed checkpoint save at
   TP2×DP2 fails validation ("rank args Namespace mismatch"). Benchmarks use
   NOSAVE=1; science runs need checkpoint_every=8, so fix/workaround first.
4. **LoRA arm (the arm the user wants): definitively blocked upstream.**
   Probe evidence (2026-08-06, `upstream_issues.md` #1): under the only
   runnable LoRA config (bshd), the Megatron actor forward returns UNIFORM
   logits (every logprob = -log(vocab) = -12.422) — the dense-Qwen3.5 LoRA
   model path is non-functional, not merely drifted. thd is hard-rejected
   for LoRA ("GDN does not support packed sequence"), and full-param+bshd
   asserts. Options: file the drafted issues + track upstream (dense GDN
   fixes are actively landing there), authorize a deeper self-patch effort
   (uncertain, deep Megatron surgery), or run science on the parity-verified
   full-param arm meanwhile.
5. **Phase 5 remaining**: async rollout/training overlap (targets the 150s
   generation wait per step), tail elimination of p99 stragglers.
6. **Phase 6**: object-storage checkpoint sync, restart.sh drill on Miles.

## Cost log

Dev slime pod (Phase 3): ~$10. Miles debugging + benches (Phase 4): ~$23 of a
$30 cap (includes one ~$9 idle SSH-outage mistake — keep-alives now standard).
