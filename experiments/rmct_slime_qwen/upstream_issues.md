# Upstream issue drafts (radixark/miles, sglang-miles)

Evidence gathered 2026-08-06 on `radixark/miles:latest-cu12` (miles c9e79e3,
sglang-miles cb05a44, Megatron 8f83233, torch 2.11.0+cu129, H200). Repro
commands use `scripts/run_miles.sh` from this experiment dir; logs in
`logs_miles_pod/` and `logs_shakeout_pod/`.

## 1. [miles] Dense Qwen3.5 + LoRA (Megatron backend): model forward yields UNIFORM logits

**Severity: LoRA training on dense Qwen3.5 is silently broken end to end.**

- Config: `--train-backend megatron`, Qwen3.5-4B (dense), `--lora-rank 8
  --lora-alpha 16 --target-modules <decoder.layers.*.mlp wildcards>
  --megatron-to-hf-mode bridge --qkv-format bshd --micro-batch-size 1`.
- Symptom: every recomputed per-token logprob is exactly −12.422 =
  −log(248320) = −log(vocab_size): the actor forward returns constant
  (uniform) logits. `train_rollout_logprob_abs_diff` ≈ 11.7; all PPO ratios
  clip; grad_norm = 0. Shift-checks rule out off-by-one gathering (mean
  |diff| is identical for aligned, +1, −1 alignments).
- LoRA cannot avoid bshd: with `--qkv-format thd` megatron-core raises
  "GDN does not support packed sequence for now" (only in the LoRA-wrapped
  path — full-parameter + thd trains CORRECTLY on the same model, abs_diff
  0.0107, so the plugin GDN handles packing when not LoRA-wrapped).
- Full-parameter + bshd is also unusable, differently:
  `miles_plugins/models/hf_attention.py:191 assert packed_seq_params is not
  None`.
- The 35B-A3B MoE LoRA example may well work; this is dense-specific
  (compare fsdp fix eba5ff5 "apply the GDN packing patch to dense qwen3_5",
  which has no Megatron-backend counterpart).

## 1b. [miles] Bridge-LoRA actor silently zero-loads the language model (root cause of #1's uniform logits)

Full causal chain, all verified by weight-norm bisection:
- `_setup_lora_model_via_bridge` builds with `load_weights=False`; with our
  P3 (`load_weights=True`) the model is healthy post-build (embedding norm
  328.5, generates correctly).
- Miles defaults `args.load` to the base torch_dist conversion when unset;
  that conversion carries `latest_checkpointed_iteration.txt`, so
  `initialize_model_and_optimizer`'s load block treats it as resumable.
- Loading it into the bridge-built model cannot be name-mapped and silently
  ZEROES the language weights — the (tied) word embedding first — leaving
  exactly-uniform logits (−log V) and zero grad_norm (all PPO ratios clip).
- `load_other_checkpoint("ref", args.ref_load)` does the same for the ref.
Fixes P5/P6/P7 in `scripts/apply_patches.py`; with them, LoRA training is
healthy (abs_diff 0.013, nonzero grads). Suggested upstream fixes: hard-error
on unmappable loads instead of silent zeroing; don't default `--load` to
`--ref-load` for bridge-LoRA runs.

## 2. [miles] mbridge base-weight export garbles dense Qwen3.5

- `--megatron-to-hf-mode bridge` `update_weights` pushes corrupted base
  weights to SGLang → deterministic multilingual token-soup generation.
  Raw-mode export of the same checkpoint serves correctly.
- Repro: full-param dense 4B, bridge mode, generate after first
  update_weights. Workaround: for LoRA runs `--lora-base-cpu-backup`
  (skip_base_sync) keeps SGLang on the pristine HF checkpoint.

## 3. [sglang-miles] LoRA memory pool assumes uniform module shapes across layers

- Hybrid GDN/attention models violate the assumption:
  `AssertionError: LoRA buffer shape torch.Size([10240, 8]) does not match
  weight shape torch.Size([6144, 8])` when loading a zero-init q/k/v/o
  adapter for Qwen3.5-4B (`/load_lora_adapter` or `--lora-paths`).
- Through Miles' registration path the same mismatch loads misaligned and
  garbles generation instead of crashing. MLP-only targets (uniform shapes)
  serve correctly (zero-adapter output byte-identical to base).

## 5. [miles] LoRA adapter checkpoints cannot be resumed through the normal load path

Evidence 2026-08-07, 9B bridge-LoRA on 2× H200 (TP2), logs in `logs_9b_lora/`.

- `save_checkpoint_with_lora` writes `iter_N/adapter/` but never
  `latest_checkpointed_iteration.txt` → a later run with `--load <ckpt_dir>`
  finds nothing and silently starts fresh (start_rollout_id=0, prior rollout
  records superseded). Creating the tracker manually instead crashes the
  generic loader: "NotImplementedError: unknown checkpoint format in iter_N".
- The working mechanism is `--lora-adapter-path <iter_N>/adapter`
  (Megatron-native per-rank shards + `training_state_rank*.pt`), but for
  bridge-LoRA runs the flag is consumed inside the very `load_checkpoint`
  call that must be skipped (see issue 1b) — our patch P8 calls
  `load_lora_adapter` directly in that branch. Verified: 128 adapter tensors
  per rank + optimizer state restore, training proceeds. Shape-changed
  resumes additionally need `--override-opt-param-scheduler`.
- Remaining bug: on the resumed run's second rollout, the SGLang engine dies
  with a CUDA device-side assert during the adapter weight push
  (`torch_memory_saver` free assert; router then 503s). Fresh-start runs
  survive the identical update_weights path, so the assert is specific to
  pushing a resumed adapter.

## 4. [miles/Megatron] dist-ckpt save at DP>1 rejects args Namespace over `rank`

- Every multi-rank save fails common-state validation: "Mismatched keys:
  [(('args',), Namespace, Namespace)]" — the only differing field is
  `args.rank` (per-process by construction).
- Fix (applied locally, `scripts/apply_patches.py` P1): store a
  rank-normalized copy: `state_dict['args'] = Namespace(**{**vars(args),
  'rank': 0})`.
