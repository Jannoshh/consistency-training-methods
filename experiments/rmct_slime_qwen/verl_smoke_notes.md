# verl GRPO+LoRA smoke test — environment plan and gotchas

Target: 1× H200 141GB on RunPod, stock `verl.trainer.main_ppo` GRPO on gsm8k,
Qwen3.5-9B, FSDP2 + vLLM rollout, LoRA r=32 α=64 on attention+MLP only, merged
weight sync, ~10 steps. Companion script: `run_smoke.sh`.

Everything below is derived from the verl clone at commit `2b0fe51` (current
main) plus Docker Hub / HF verification. Items I could not verify are marked
**UNVERIFIED** rather than guessed.

---

## 1. Docker image

**Use `verlai/verl:vllm024.dev2`.**

Evidence:

- It is the exact build target of the current stable vLLM Dockerfile:
  `docker/Dockerfile.stable.vllm:2` — `# Target: verlai/verl:vllm024.dev2`.
- It is the image verl's own CI runs against: `.github/workflows/vllm.yml:75`
  sets `IMAGE: "verl-ci-cn-beijing.cr.volces.com/verlai/verl:vllm024.dev2"`.
- Confirmed present on Docker Hub via the registry API, pushed
  2026-07-27T07:06:54Z. (`vllm024.dev1`, 2026-07-23, also exists.)
- `docker/README.md:89` confirms this is the current stable line:
  "2026/07/21: update vllm stable image to vllm==0.24.0 (torch==2.11.0,
  CUDA 13.0.2, Ubuntu 24.04)".
- Note the tag the docs advertise, `verlai/verl:vllm024.latest`
  (`docker/README.md:27`), **does not exist** on Docker Hub — only the
  `.dev1`/`.dev2` builds of the 0.24.0 line have been pushed. Use `.dev2`.

Version stack, from the Dockerfile ARGs (`docker/Dockerfile.stable.vllm:4-22`):

| component | version |
|---|---|
| CUDA | 13.0.2 (base `nvidia/cuda:13.0.2-devel-ubuntu24.04`) |
| Python | 3.12 |
| torch | 2.11.0 |
| vLLM | 0.24.0 (x86_64) |
| transformers | 5.3.0 |
| flash-attn | 2.8.3 |

This clears the "vLLM ≥ 0.18, CUDA ≥ 12.4" bar comfortably.

**No sshd.** There is no `openssh` package anywhere under `docker/`. RunPod SSH
access requires installing `openssh-server` yourself; `run_smoke.sh env` does it.

**verl is not baked in.** `docker/Dockerfile.stable.vllm:178` runs
`pip install git+…/verl.git@v0.7.1 && pip uninstall -y verl` — it installs verl
only to resolve its dependency closure, then removes the package. So the image
has the deps but not verl, and you must install the clone yourself.

## 2. Pip layer on top of the image

```bash
pip3 install --no-cache-dir "transformers>=5.5.3,!=5.6.0,<5.11"   # REQUIRED
pip3 install --no-cache-dir flash-linear-attention                # optional here
apt-get install -y openssh-server                                 # RunPod SSH
pip3 install --no-deps -e /workspace/verl                         # verl itself
```

- **transformers must end up ≥5.5.3, but the image's ARG does not settle it.**
  Current verl main requires `transformers>=5.5.3,!=5.6.0,<5.11` (`setup.py:43`,
  `requirements.txt:20`). The Dockerfile ARG pins 5.3.0
  (`Dockerfile.stable.vllm:12`, installed at `:144`), which would *not* satisfy
  that — but a later layer (`:178`) pip-installs verl v0.7.1 for its dependency
  closure, and that can pull transformers forward as a side effect. So the
  effective in-container version is **not** determined by the ARG alone. The
  install command above is idempotent (no-op if already satisfied), and the
  script prints `pip show transformers` first so the real version is on record.
  The exclusion of 5.6.0 is deliberate: it "ships a broken flash-attention path
  (crashes on `s_aux=None` for sink-less models)", per the comment at
  `setup.py:41-42` referencing huggingface/transformers#45588. This pin is also
  what covers the 3D `position_ids` / flash-attn varlen concern.
- **`--no-deps` on the verl install** keeps pip from re-resolving the image's
  carefully pinned torch/vLLM stack.
- **flash-linear-attention is optional at `sp_size=1`**, which is the more
  useful finding than a version pin. verl's Qwen3.5 patch has torch fallbacks
  for both varlen GatedDeltaNet kernels: `_packed_causal_conv1d_fallback`
  (`verl/models/transformers/qwen3_5.py:146`) and the per-sequence split loop in
  `_packed_chunk_gated_delta_rule` (`qwen3_5.py:158-188`). FLA is only *required*
  for Ulysses sequence parallelism, where the code hard-raises
  `"Qwen3.5 Ulysses SP requires FLA chunk_gated_delta_rule cp_context support"`
  (`qwen3_5.py:170-171`). Since we run `sp_size=1`, a failed FLA build degrades
  speed, not correctness — so the script warns instead of aborting.
  FLA is not pinned anywhere in verl's `setup.py`/`requirements.txt`.
- The `*_implementation` selectors that force the FLA kernels
  (`rms_norm_gated_implementation`, `causal_conv1d_implementation`,
  `chunk_gated_delta_rule_implementation`) are fields on **`VeOmniEngineConfig`**
  (`verl/workers/config/engine.py:296`, fields at `:421-423`), **not**
  `FSDPEngineConfig`. Do not try to set them on the FSDP path — they are inert
  there.

**UNVERIFIED:** the transformers version actually resolved inside the running
container, and whether vLLM 0.24.0 ships the `qwen3_5` model. The Qwen3.5
example headers in the clone say `vllm==0.18.0, transformers@<cc7ab9be>`
(e.g. `examples/grpo_trainer/run_qwen3_5_27b_fsdp.sh:2`), i.e. they were written
against a transformers *commit*, not a release. The `preflight` stage imports
`transformers.models.qwen3_5` and `vllm.model_executor.models.qwen3_5`
explicitly so this fails on CPU in seconds rather than after the GPU is billed.

## 3. LoRA target_modules for Qwen3.5-9B

Qwen3.5 is a **hybrid** architecture: each decoder layer binds *either*
`self_attn` (`Qwen3_5Attention`) *or* `linear_attn` (`Qwen3_5GatedDeltaNet`),
selected by `config.layer_types`. verl's FLOPs counter confirms the two layer
kinds and their parameter shapes (`verl/utils/flops_counter.py:271-308`), and
notes that full-attention `q_proj` is 2× the query size because it also emits a
sigmoid gate.

Module names:

| block | linear modules |
|---|---|
| `self_attn` (full attention) | `q_proj` (fused query+gate), `k_proj`, `v_proj`, `o_proj` |
| `linear_attn` (GatedDeltaNet) | `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`, `out_proj` (plus non-Linear `conv1d`, `dt_bias`, `A_log`, `norm`) |
| `mlp` | `gate_proj`, `up_proj`, `down_proj` |

The 9B checkpoint is **dense** (not MoE), 32 layers, `layer_types` repeating
`[linear, linear, linear, full]` — full attention at layers 3, 7, 11, 15, 19,
23, 27, 31. It is also **multimodal**: the config carries a `vision_config`, and
verl loads it as `Qwen3_5ForConditionalGeneration`
(`verl/models/transformers/qwen3_5.py:26-29`), so the LLM backbone sits under
`model.language_model.layers.{i}.…`.

`_keep_in_fp32_modules` is **not set** on the Qwen3.5 classes. So the
"#7092 fp32-modules caveat" does not apply through that mechanism here. There is
still a dtype concern, but it is handled: verl explicitly casts fp32 LoRA
adapter params down to the bf16 base dtype because "FSDP requires all params in
a flat group to share dtype"
(`verl/workers/engine/fsdp/transformer_impl.py:352-362`). Watch for that log
line; if it reports casting an unexpected number of params, investigate.

### The value to use

```
.*language_model\.layers\.[0-9]+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)
```

**Why a regex and not a list.** peft treats a *list* of strings as **suffix**
matches and a *single string* as a **regex fullmatch** against the full module
path. The suffix form
`["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]` does
correctly avoid every GatedDeltaNet linear — `out_proj` does not end with
`o_proj`, and none of the `in_proj_*` names collide — but it would *also* hit
the vision tower's linears, which never receive gradients on a text-only gsm8k
run. Anchoring on `language_model` requires the regex form.

**Do NOT use verl's default `target_modules=all-linear`** — it would adapt the
GatedDeltaNet `in_proj_*`/`out_proj` layers, which is exactly what we are
avoiding.

Verified by construction (`re.fullmatch` against realistic paths): the regex
matches all seven attention/MLP linears in a full-attention layer and matches
none of `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`, `out_proj`,
`conv1d`, the `model.visual.*` linears, or `lm_head`.

**UNVERIFIED:** the exact module-path prefix on the real checkpoint. If it is
not `model.language_model.…`, the regex matches zero modules and peft raises
loudly — a fast, obvious failure, not a silent wrong-modules run. The
`preflight` stage materializes the model on the meta device and prints matched
vs unmatched module names so you can confirm before training.

The prefix is load-bearing and depends on **which auto class verl picks**, so
the preflight resolves it the same way training does, via verl's own
`get_hf_auto_model_class` (`verl/utils/model.py:686-710`). That function checks
`type(hf_config) in AutoModelForImageTextToText._model_mapping` *before* falling
back to the architecture-name table, so Qwen3.5-9B — which declares
`architectures[0] = Qwen3_5ForConditionalGeneration` and carries a
`vision_config` — resolves to **`AutoModelForImageTextToText`**, not
`AutoModelForCausalLM`. Building the preflight tree with `AutoModelForCausalLM`
would yield different module paths than training and make the check actively
misleading; the script prints the resolved class name so this is visible.
(A text-only `Qwen3_5ForCausalLM` load would have no `language_model` segment at
all — paths would be `model.layers.{i}.…` — which is exactly the case the regex
would silently miss without this check.)

## 4. Two config traps that would abort the run

**(a) `lora.merge` needs `++`, not `+`.** verl's own shipped example
`examples/tuning/lora/run_qwen3_8b_merge_fsdp.sh:56` writes
`+actor_rollout_ref.model.lora.merge=True`. That is a bug: the key already
exists in the composed config (`verl/trainer/config/model/hf_model.yaml:100-105`
defines `lora:` with `merge: False`), and Hydra's `+` *appends*, erroring with
`Could not append to config. An item is already at 'model.lora.merge'`. I
confirmed this against Hydra's own `OverridesParser`. Use `++`, which appends
*or* overrides and is therefore correct either way.

**(b) `save_lora_only` needs `++` too, for the opposite reason.** It is a
`CheckpointConfig` dataclass field (`verl/trainer/config/config.py:51`) that
appears in **no** YAML, so a bare override fails with `Could not override`. `++`
handles it.

Also verified: passing the regex bare fails Hydra's grammar
(`mismatched input '[' expecting <EOF>`) because of the `[`/`(`/`|`. The value
must carry **literal** quotes into Hydra, so the script emits
`actor_rollout_ref.model.target_modules='<regex>'`.

## 5. Where the nonobvious overrides come from

| override | source |
|---|---|
| `model.lora_rank` / `lora_alpha` / `target_modules` / `exclude_modules` | the FSDP peft path reads these (`verl/workers/engine/fsdp/transformer_impl.py:340-349`). The `model.lora.*` block is the *Megatron* schema (`verl/workers/config/model.py:131-132`) — its `target_modules` defaults are Megatron names (`linear_qkv`, `linear_proj`, …) and are irrelevant on FSDP. Only `lora.merge` is read on the FSDP path. |
| `model.lora.merge=True` | `verl/workers/engine_workers.py:669` sets `peft_merge`; `transformer_impl.py:968` branches to `_merged_lora_per_tensor_param()`, streaming merged base+LoRA weights and returning `peft_config=None` so the rollout takes a plain full-weight update — no adapter serving. |
| `rollout.load_format=safetensors` | required for LoRA so vLLM loads the base model from disk (`docs/advance/ppo_lora.rst:37`). |
| `actor.checkpoint.save_lora_only=True` | `verl/trainer/config/config.py:39-51` — saves adapter weights only (~150 MiB vs ~54 GiB for a 27B). |
| `actor.fsdp_config.ulysses_sequence_parallel_size` | `verl/trainer/config/engine/fsdp.yaml:45`. The flat `actor.ulysses_sequence_parallel_size` still exists but is marked `[DEPRECATED]` at `verl/trainer/config/actor/dp_actor.yaml:33`. `docs/advance/ppo_lora.rst:199` still shows the deprecated form. |
| `trainer.total_training_steps=10` | `verl/trainer/ppo/ray_trainer.py:437-440` — overrides the epoch-derived count, giving a hard stop at 10. |
| `ppo_mini_batch_size` semantics | counted in **prompts**; verl multiplies by `rollout.n` internally (`ray_trainer.py:1328`). Constraint `train_batch_size >= ppo_mini_batch_size` (`verl/workers/config/actor.py:229`). So 32 prompts / mini 8 / n=4 → 32 sequences per optimizer step, 4 optimizer steps per training step. |
| no custom reward | `examples/data_preprocess/gsm8k.py:47,69` tags rows `data_source="openai/gsm8k"`, which the default reward manager routes to the builtin scorer (`verl/utils/reward_score/__init__.py:44-47`). |
| `rollout.prompt_length` / `response_length` | not set directly; they interpolate from `data.max_*` (`verl/trainer/config/rollout/rollout.yaml:35,39`). |
| ref model memory | with `lora_rank>0`, verl sets `ref_in_actor=True` and reuses the actor with the adapter disabled, so no separate reference model is materialized. |

## 6. VRAM sizing

Qwen3.5-9B bf16 ≈ 18 GB of frozen base weights. LoRA r=32 on attn+MLP only is a
few hundred MB including optimizer state. vLLM takes
`gpu_memory_utilization=0.4` ≈ 56 GB of the 141 GB. Actor offloading is off
(`param_offload=False`, `optimizer_offload=False`) since it all fits, and
`free_cache_engine=True` (default) releases the KV cache during the training
phase. Merged sync does the merge in place — `merged_lora_context` restores the
un-merged base weights on exit (`transformer_impl.py:1035-1048`) — so it does
not need a second full copy of the weights.

`layered_summon` is left at its default `False`; the docs recommend it only for
70B+ or <48 GB GPUs (`docs/advance/ppo_lora.rst:52`). Turn it on if you see an
OOM during weight sync.

## 7. What to watch during the smoke

Gotchas to actively look for:

- **Rollout garbling at step 2–3.** The classic merged-sync failure: step 1
  looks fine (vLLM still holds the freshly loaded base weights) and output turns
  to noise once the first merged update lands. Dump actual response text, don't
  just trust reward. A collapse in mean response length or reward to ~0 at step
  2–3 is the same signal.
- **`CUDA error: an illegal memory access was encountered`**, typically during
  the GatedDeltaNet varlen path or the weight-sync bucket copy. If it appears,
  first retry with `use_remove_padding=False` to isolate the varlen kernels from
  the sync path.
- **LoRA dtype casting log** from `transformer_impl.py:352-362` — confirm the
  count is what you expect for r=32 on attn+MLP.
- **peft "Target modules not found"** — means the regex prefix guess was wrong;
  the preflight stage should have caught this first.

Metrics to grep from the console log:

Metric keys below are verified against this clone, not assumed —
`actor/pg_loss` and `actor/entropy_loss` at `verl/workers/utils/losses.py:119,129`,
`kl_loss` (no `actor/` prefix) at `losses.py:141`, `actor/grad_norm` at
`verl/trainer/ppo/ray_trainer.py:1752`, and `critic/rewards/mean` /
`response_length/mean` / `timing_s/*` / `perf/*` at
`verl/trainer/ppo/metric_utils.py:560,573,645,685-686`. Note the perf keys are
`perf/throughput` and `perf/time_per_step` — there is no `perf/mfu`.

```bash
grep -E "actor/grad_norm|actor/pg_loss|actor/entropy_loss|kl_loss|critic/rewards/mean" smoke_*.log
grep -E "response_length/mean|prompt_length/mean" smoke_*.log
grep -E "timing_s/(step|gen|update_actor|ref|adv)" smoke_*.log
grep -E "perf/(throughput|time_per_step|total_num_tokens)" smoke_*.log
grep -iE "Traceback|illegal memory access|OOM|out of memory|assert|NaN|inf" smoke_*.log
```

Pass criteria for the smoke: 10 steps complete; `grad_norm` finite and roughly
stable (not 0, not exploding, not NaN); response text still coherent English at
step 10; a checkpoint written at step 5 and 10 containing adapter weights only
(a few hundred MB, not tens of GB); and a sane `timing_s/step` breakdown so we
know where the time actually goes before scaling up.
