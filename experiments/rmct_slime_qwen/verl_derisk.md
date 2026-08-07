# verl derisk assessment (2026-08-07)

Question: should RMCT-LoRA on Qwen3.5-9B migrate from Miles (Megatron+SGLang)
to verl? Method: three parallel research agents over verl source, docs, issue
trackers, and field reports; no GPU spend. **Verdict: conditionally promising —
verl-FSDP is the only stack that offers BOTH packed-GDN training and LoRA, but
three specific risks must clear a cheap smoke test before migrating.**

## The decisive finding

Miles' LoRA slowness is structural: Megatron has no packed-sequence (THD)
support for Qwen3.5 GDN — verl's own Megatron example scripts force bshd with
`use_dynamic_bsz=False` for exactly this reason, so a Miles→verl-Megatron move
buys nothing. But verl's **FSDP backend merged real packed-GDN support**
(PR #6660, 2026-07-20): packed `cu_seqlens` flow into patched
`Qwen3_5GatedDeltaNet` forwards using FLA kernels (`verl/models/transformers/
qwen3_5.py`), with a distributed test and a validated Qwen3.5-27B GRPO SP8
run (output err ~3.6e-4). Before that patch, packing silently leaked
linear-attention state across sequence boundaries (#5639) — it still does for
Qwen3-Next, so the model_type gate matters. FSDP LoRA is first-class
(peft; `lora_rank/alpha/target_modules`, `lora_adapter_path` resume,
`checkpoint.save_lora_only` → ~150 MiB checkpoints).

verl-FSDP + LoRA + packing is therefore the unique combination that attacks
our ~81 min/step (bshd micro-batch-1 serialization) at the root.

## Risks that must clear before migration (the smoke-test gate)

1. **Open crash at our exact size**: verl#6549 "illegal memory access
   training qwen3.5 9b/27b (vLLM+FSDP2)", inside `torch_chunk_gated_delta_rule`;
   no maintainer reply. Likely root cause: transformers' 3D mRoPE
   `position_ids` corrupting flash-attn varlen `cu_seqlens`
   (transformers#44910/#44643, #6284 chain) — **fixed in transformers 5.4**.
   Mitigation: pin transformers ≥ 5.4; if the crash persists it is a
   different bug and a hard blocker.
2. **LoRA GRPO collapse, reproduced and undiagnosed**: verl#3784/#3226/#3159 —
   clean step 1, garbled rollouts from step 2, full-FT clean on identical
   config; all closed without root cause. Signature implicates adapter→engine
   weight sync. Mitigation: use `lora.merge=True` (adapter merged into base,
   full-weight sync — the plain serving path, no adapter machinery at all)
   and gate explicitly on rollout text quality at steps 2–3.
3. **LoRA serving on hybrid models is still broken everywhere**:
   verl FSDP+LoRA+SGLang adapter sync broken on main (#7287 chain, SGLang
   pinned ==0.5.8); SGLang hybrid LoRA memory-pool bugs still open upstream
   (sglang#31523 with a Qwen3.5-4B repro, #30168 GDN in_proj naming); vLLM has
   real GDN LoRA plumbing (in_proj_qkv/z/ba targets) but an unmerged
   partial-packed-group regression (vllm#47639, broken v0.21→0.26).
   Mitigation: merge-mode sidesteps ALL of it; if adapter-mode is ever wanted,
   target whole packed groups or MLP-only (our Miles conclusion, independently
   re-derived by vLLM users and by Osmosis).

## Supporting field evidence

- Only published working LoRA RL run on Qwen3.5 (Osmosis, 122B-A10B):
  Megatron+SGLang, rank 32/α32, targets qkv/o/gate_up/down, **GDN projections
  left alone**, LoRA +44% token throughput over full FT — with heavy patching.
- End-to-end verl GRPO+LoRA handbook run (Qwen2.5): works, ~$40/9.5h on
  RunPod.
- verl docs: LoRA rank ≥ 32 recommended for RL convergence, LR ~10× SFT —
  a deviation candidate for our r8/α16 paper config on any stack (D-record
  required if adopted).
- Qwen3.5 FSDP examples ship `use_remove_padding=True, use_dynamic_bsz=False`;
  known-good pins from #6660: FLA 0.5.1, transformers 5.3.0.dev+ (use ≥5.4),
  and `VLLM_USE_FLASHINFER_MOE_FP16=0` for MoE variants (n/a for dense 9B).
- Precision papercut: verl ignores `_keep_in_fp32_modules` (#7092, open) —
  check which modules Qwen3.5 marks fp32 and whether it matters at 9B.

## RMCT port shape (extension-point analysis, verl HEAD 2b0fe51)

RMCT fits verl **without forking it** — an external recipe (`recipe/` pattern,
like DAPO/PRIME), ~400–450 new lines:

- **Dataset**: two rows per datapoint (`variant ∈ {reference, training}`)
  sharing a `group_id`; `rollout.n=128`, temperature 1.0. A custom agent loop
  (`@register("rmct")`, ~80 lines) picks which variant's messages to sample
  and emits `variant`/`parse_ok`/`trait` via `extra_fields`.
- **Advantage/KL**: override `RayPPOTrainer._update_actor` (~120 lines) —
  the batch there already contains `old_log_probs` AND `ref_log_prob`
  (under LoRA the ref is the adapter-disabled frozen base — exactly RMCT's
  KL target); group by uid, compute p_ref/p_hat, RMCT advantage,
  batch-centered KL, write `advantages` + `response_mask`, call super().
  verl's own `kl_penalty` clamps to ±20 — reimplement, don't reuse.
- **Skip semantics**: zeroing `response_mask` is first-class (rollout-
  correction does exactly this); masked rows drop out of the loss
  normalization denominator (DP-all-reduced token count) — cleaner than our
  Miles zero-mask hack. Guards exist for the all-masked-batch edge.
- **Disable verl KL** (`use_kl_in_reward=False`, `use_kl_loss=False`) but
  force `self.use_reference_policy=True` in the trainer subclass, else the
  ref worker never runs.
- **Persistence**: `trainer.rollout_data_dir` dumps all completions +
  reward extras (p_ref/p_hat/parse_ok ride along) — our per-step record
  requirement is config, not code.
- **Known costs**: reference rows pay a wasted train-forward with zero mask
  (acceptable; fixable later with a fit() override that slices them out);
  verl v0 trainer is `@deprecated` mid-migration to a v1 TransferQueue API —
  pin a verl SHA, budget a v1 port later.

## Smoke-test plan (gate before any migration work)

One cheap pod (1× H100/H200), stock verl GRPO example — NOT the RMCT port:
1. Qwen3.5-9B, FSDP2 + vLLM, transformers ≥ 5.4, `lora_rank=32`,
   `target_modules` = attn+MLP linears (no GDN projections), `lora.merge=True`,
   `use_remove_padding=True`, `use_dynamic_bsz=False`, `sp_size=1`,
   `save_lora_only=True`, any toy dataset (gsm8k), ~10 steps.
2. Gates: no illegal-memory-access; rollout text quality inspected at steps
   2–3 specifically; nonzero grad_norm; step-time/token-throughput recorded
   vs Miles' 81 min baseline shape.
3. If green → port RMCT as a verl recipe (custom rollout/advantage; Gate A
   suite carries over unchanged; Gate B re-run at 2B then 9B).
4. If red on #1 after the transformers pin → verl is out; fall back to the
   Miles micro-batch>1 probe.

## SMOKE TEST RESULT (2026-08-07, 1× H200, $7.38 total): **PASS — GO for migration**

10-step GRPO+LoRA on Qwen3.5-9B, FSDP2 + vLLM 0.18.1, stock gsm8k. All three
gates cleared:

| gate | result |
|---|---|
| illegal memory access (verl#6549) | **none** in 10 steps (transformers 5.10.4 carries the position_ids fix) |
| LoRA rollout collapse at steps 2–3 | **absent** — rewards stable then RISING (0.17→0.34), response_length stable ~960–1000 |
| grad_norm | 0.04–0.05 throughout, no NaN/zero |

Extras: `rollout_probs_diff_mean ≈ 0.0026` (sampling↔training logprob
agreement — 5× tighter than our Miles Gate B 0.014); warm step 115s at
smoke shape (gen 14.5s, logprobs 33.5s, update_actor 51.3s, merged weight
sync 14.2s); LoRA-only checkpoints 353 MB; ran on verl's **v1 TransferQueue
trainer** (default) — the RMCT recipe must either pin `trainer.use_v1=false`
or target the v1 override surface.

Environment lessons (encoded in `/workspace/verl_cache` on the RunPod volume:
built flash-attn wheel, pip freeze, train script, full log):
- CUDA-13 wheels (vLLM ≥0.20, the verlai/verl:vllm024 image) need driver
  ≥580; on a driver-570 pod use **vLLM 0.18.1 + torch 2.10.0+cu128**.
- flash-attn has no prebuilt wheel past torch 2.8 — source build with
  **MAX_JOBS≤8** (32 parallel nvcc jobs hit the container cgroup OOM killer).
- LoRA target regex verified on the real checkpoint:
  `.*language_model\.layers\.[0-9]+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)`
  (128 of 359 linears matched, zero GDN/vision hits; model class resolves to
  AutoModelForImageTextToText).

## Bottom line

verl-FSDP+vLLM is validated for Qwen3.5-9B LoRA RL: the two crash-classes
did not manifest, logprob agreement beats our Megatron numbers, and packing
works. Migration (RMCT recipe port) is in progress; remaining risks are
ordinary porting work, not framework viability.
