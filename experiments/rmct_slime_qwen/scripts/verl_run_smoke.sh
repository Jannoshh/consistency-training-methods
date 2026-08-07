#!/usr/bin/env bash
# verl GRPO + LoRA smoke test | Qwen3.5-9B | 1x H200 141GB | FSDP2 training + vLLM rollout
#
# Image: verlai/verl:vllm024.dev2   (CUDA 13.0.2 / torch 2.11.0 / vLLM 0.24.0 / flash-attn 2.8.3)
# verl is NOT baked into that image (docker/Dockerfile.stable.vllm:178 installs it for its
# dependency closure and then `pip uninstall -y verl`), so we install this clone with --no-deps.
#
# Usage:
#   ./run_smoke.sh env        # pip layer on top of the image (run once per pod)
#   ./run_smoke.sh preflight  # cheap CPU-only checks; MUST pass before spending GPU hours
#   ./run_smoke.sh data       # gsm8k parquet prep
#   ./run_smoke.sh model      # download Qwen/Qwen3.5-9B
#   ./run_smoke.sh train      # the 10-step GRPO+LoRA run
#   ./run_smoke.sh all        # env -> preflight -> data -> model -> train
#
# Every hydra override below is derived from the config schema in this clone; see NOTES.md
# for the file:line each nonobvious one comes from.

set -xeuo pipefail

STAGE="${1:-all}"

########################### user-adjustable ###########################
WORKSPACE=${WORKSPACE:-/workspace}
VERL_DIR=${VERL_DIR:-${WORKSPACE}/verl}
MODEL_PATH=${MODEL_PATH:-${WORKSPACE}/models/Qwen3.5-9B}
HF_MODEL_ID=${HF_MODEL_ID:-Qwen/Qwen3.5-9B}
DATA_DIR=${DATA_DIR:-${WORKSPACE}/data/gsm8k}
CKPTS_DIR=${CKPTS_DIR:-${WORKSPACE}/ckpts/verl_smoke}
LOG_DIR=${LOG_DIR:-${WORKSPACE}/logs}

PROJECT_NAME=${PROJECT_NAME:-verl_smoke}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_5_9b_lora_grpo_gsm8k}

# --- smoke-test sizing ---
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-10}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}          # prompts per training step
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}     # prompts per optimizer step (x rollout.n internally)
MICRO_BATCH_PER_GPU=${MICRO_BATCH_PER_GPU:-1}     # sequences per forward
ROLLOUT_N=${ROLLOUT_N:-4}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}
SAVE_FREQ=${SAVE_FREQ:-5}                          # -> checkpoints at step 5 and step 10

# --- LoRA ---
LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-64}
ACTOR_LR=${ACTOR_LR:-1.0e-05}                      # LoRA wants ~10x the full-FT LR

# Attention + MLP linears ONLY. Deliberately excludes the GatedDeltaNet
# (in_proj_qkv / in_proj_z / in_proj_b / in_proj_a / out_proj / conv1d) and the
# vision tower. peft treats a single STRING as a regex fullmatch against the full
# module path, which is what lets us anchor on `language_model`; a LIST would match
# by suffix only and would also hit the vision tower. See NOTES.md.
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-'.*language_model\.layers\.[0-9]+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)'}

# --- VRAM ---
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.4}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-1}
NNODES=${NNODES:-1}
SP_SIZE=${SP_SIZE:-1}
########################### end user-adjustable ###########################

mkdir -p "${LOG_DIR}" "${DATA_DIR}" "${CKPTS_DIR}"

########################### stage: env ###########################
stage_env() {
    # verl main requires transformers >=5.5.3 (setup.py:43 / requirements.txt:20). The image's
    # ARG pins 5.3.0 (docker/Dockerfile.stable.vllm:12) but a later layer (:178) pip-installs
    # verl v0.7.1 for its deps, which may pull transformers forward as a side effect -- so the
    # effective version in the container is NOT determined by the ARG alone. This install is
    # idempotent: a no-op if already satisfied, the needed upgrade if not.
    pip3 show transformers || true
    pip3 install --no-cache-dir "transformers>=5.5.3,!=5.6.0,<5.11"

    # flash-linear-attention: OPTIONAL at ulysses sp_size=1. verl's Qwen3.5 patch has torch
    # fallbacks for both varlen kernels (_packed_causal_conv1d_fallback at
    # verl/models/transformers/qwen3_5.py:146 and the per-sequence split loop in
    # _packed_chunk_gated_delta_rule at :158-188). FLA is only *required* for Ulysses SP,
    # which needs chunk_gated_delta_rule with cp_context support (qwen3_5.py:170-171).
    # Installed anyway for speed; do not fail the run if the build fails.
    pip3 install --no-cache-dir flash-linear-attention || \
        echo "WARN: flash-linear-attention failed to install; falling back to verl's torch path (slower, still correct at sp_size=1)"

    # RunPod needs sshd; the verl image ships none (no openssh package anywhere in docker/).
    if ! command -v sshd >/dev/null 2>&1; then
        apt-get update && apt-get install -y --no-install-recommends openssh-server && mkdir -p /run/sshd
    fi

    # verl itself, from this clone, without letting pip re-resolve the image's pinned stack.
    pip3 install --no-deps -e "${VERL_DIR}"
}

########################### stage: preflight ###########################
# CPU-only. Every check here is a thing that would otherwise fail *after* the GPU is billed.
stage_preflight() {
    python3 - <<'PYEOF'
import sys, re, importlib
ok = True

def check(label, fn):
    global ok
    try:
        print(f"[ OK ] {label}: {fn()}")
    except Exception as e:
        ok = False
        print(f"[FAIL] {label}: {type(e).__name__}: {e}")

import transformers, torch
check("transformers version", lambda: transformers.__version__)
check("torch version", lambda: torch.__version__)
check("transformers has qwen3_5",
      lambda: importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5").__name__)
check("vllm has qwen3_5",
      lambda: importlib.import_module("vllm.model_executor.models.qwen3_5").__name__)
check("verl imports", lambda: importlib.import_module("verl").__file__)
check("verl qwen3_5 patch imports",
      lambda: importlib.import_module("verl.models.transformers.qwen3_5").__file__)
check("peft imports", lambda: importlib.import_module("peft").__version__)

# The load-bearing check: does the LoRA regex hit attention+MLP and miss the GatedDeltaNet?
import os
model_path = os.environ["MODEL_PATH"]
pattern = os.environ["LORA_TARGET_MODULES"]
try:
    from transformers import AutoConfig
    import torch.nn as nn
    # Use verl's OWN class selector (verl/utils/model.py:686-710), not AutoModelForCausalLM.
    # Qwen3.5-9B declares architectures[0]=Qwen3_5ForConditionalGeneration and carries a
    # vision_config, so verl resolves it to AutoModelForImageTextToText. Building the tree with
    # any other auto class would give different module paths than training actually sees, which
    # would make this whole check misleading.
    from verl.utils.model import get_hf_auto_model_class
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=False)
    auto_class = get_hf_auto_model_class(hf_config=cfg)
    print(f"[INFO] verl will instantiate via: {auto_class.__name__}")
    print(f"[INFO] architectures: {getattr(cfg, 'architectures', None)}")
    with torch.device("meta"):
        model = auto_class.from_config(cfg)
    linears = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    hit = [n for n in linears if re.fullmatch(pattern, n)]
    miss = [n for n in linears if not re.fullmatch(pattern, n)]
    bad = [n for n in hit if any(t in n for t in
           ("in_proj", "out_proj", "conv1d", "linear_attn", "visual", "vision"))]
    print(f"[INFO] total nn.Linear modules: {len(linears)}")
    print(f"[INFO] matched by LoRA regex : {len(hit)}")
    print(f"[INFO] sample matched        : {hit[:4]}")
    print(f"[INFO] sample NOT matched    : {miss[:8]}")
    if not hit:
        ok = False
        print("[FAIL] LoRA regex matched ZERO modules -- peft would raise. Fix LORA_TARGET_MODULES.")
    elif bad:
        ok = False
        print(f"[FAIL] LoRA regex caught GatedDeltaNet/vision modules: {bad[:8]}")
    else:
        print("[ OK ] LoRA regex hits attention+MLP only; no GDN/vision modules caught.")
except Exception as e:
    ok = False
    print(f"[FAIL] target_modules check: {type(e).__name__}: {e}")

sys.exit(0 if ok else 1)
PYEOF
}

########################### stage: data ###########################
stage_data() {
    # examples/data_preprocess/gsm8k.py writes train.parquet / test.parquet and tags every row
    # data_source="openai/gsm8k" (gsm8k.py:47,69), which verl's default reward manager routes to
    # the builtin scorer (verl/utils/reward_score/__init__.py:44-47). No custom reward needed.
    python3 "${VERL_DIR}/examples/data_preprocess/gsm8k.py" --local_save_dir "${DATA_DIR}"
}

########################### stage: model ###########################
stage_model() {
    huggingface-cli download "${HF_MODEL_ID}" --local-dir "${MODEL_PATH}"
}

########################### stage: train ###########################
DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="${DATA_DIR}/train.parquet"
    data.val_files="${DATA_DIR}/test.parquet"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.shuffle=False
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    # FSDP LoRA is built from these peft-named fields
    # (verl/workers/engine/fsdp/transformer_impl.py:340-349).
    actor_rollout_ref.model.lora_rank=${LORA_RANK}
    actor_rollout_ref.model.lora_alpha=${LORA_ALPHA}
    # The inner single quotes are LITERAL and required: Hydra's override grammar rejects a bare
    # value containing [ ] ( ) | -- verified with hydra's OverridesParser. The shell strips the
    # outer double quotes, so Hydra receives target_modules='<regex>'.
    actor_rollout_ref.model.target_modules="'${LORA_TARGET_MODULES}'"
    # Merged full-weight sync to vLLM -- no adapter serving.
    # `++` (not `+`) because lora.merge ALREADY EXISTS in the composed config
    # (verl/trainer/config/model/hf_model.yaml:100-105); verl's own shipped example
    # examples/tuning/lora/run_qwen3_8b_merge_fsdp.sh uses `+` and would abort. See NOTES.md.
    ++actor_rollout_ref.model.lora.merge=True
)

ACTOR=(
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_PER_GPU}
    actor_rollout_ref.actor.use_dynamic_bsz=False
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${NGPUS_PER_NODE}
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    # non-deprecated location for ulysses SP (verl/trainer/config/engine/fsdp.yaml:45;
    # actor.ulysses_sequence_parallel_size is marked DEPRECATED in actor/dp_actor.yaml:33)
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=${SP_SIZE}
    # save_lora_only is a CheckpointConfig dataclass field (verl/trainer/config/config.py:51)
    # with no YAML entry, so it must be appended -- `++` appends or overrides.
    ++actor_rollout_ref.actor.checkpoint.save_lora_only=True
)

REF=(
    # With lora_rank>0 verl sets ref_in_actor=True and reuses the actor with the adapter
    # disabled, so no separate reference model is materialized.
    actor_rollout_ref.ref.strategy=fsdp2
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH_PER_GPU}
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.ref.fsdp_config.param_offload=False
    actor_rollout_ref.ref.use_torch_compile=False
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    # required for LoRA: lets vLLM load the base model off disk (docs/advance/ppo_lora.rst:37)
    actor_rollout_ref.rollout.load_format=safetensors
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH_PER_GPU}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.enable_prefix_caching=False
)

TRAINER=(
    trainer.critic_warmup=0
    trainer.logger='["console"]'
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.default_local_dir="${CKPTS_DIR}"
    trainer.balance_batch=False
    trainer.val_before_train=False
    trainer.test_freq=-1
    trainer.save_freq=${SAVE_FREQ}
    # hard stop at 10 steps regardless of epoch length (verl/trainer/ppo/ray_trainer.py:437-440)
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS}
    trainer.total_epochs=1
)

stage_train() {
    local ts
    ts=$(date +%Y%m%d_%H%M%S)
    python3 -m verl.trainer.main_ppo \
        "${DATA[@]}" \
        "${MODEL[@]}" \
        "${ACTOR[@]}" \
        "${REF[@]}" \
        "${ROLLOUT[@]}" \
        "${TRAINER[@]}" \
        2>&1 | tee "${LOG_DIR}/smoke_${ts}.log"
}

########################### dispatch ###########################
export MODEL_PATH LORA_TARGET_MODULES

case "${STAGE}" in
    env)       stage_env ;;
    preflight) stage_preflight ;;
    data)      stage_data ;;
    model)     stage_model ;;
    train)     stage_train ;;
    all)       stage_env; stage_model; stage_preflight; stage_data; stage_train ;;
    *)         echo "unknown stage: ${STAGE}" >&2; exit 1 ;;
esac
