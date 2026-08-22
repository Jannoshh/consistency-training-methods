#!/usr/bin/env bash
# VDCT pod bring-up on top of verlai/verl:vllm024.dev2 (CUDA 13.0.2 / torch 2.11.0
# / vLLM 0.24.0 / flash-attn 2.8.3). Qwen3-8B, LoRA, 1x GPU.
#
# The image ships verl's dependency closure but NOT verl itself
# (docker/Dockerfile.stable.vllm:178 installs it for deps then pip-uninstalls),
# so the clone is installed here with --no-deps to keep the pinned stack intact.
#
# Package installs go through `uv pip install --system`: we are LAYERING onto the
# image's python, not building a fresh venv -- a venv would discard the pinned
# torch/vLLM/flash-attn closure that is the whole reason for this image.
#
# Usage (on the pod):
#   ./setup_pod.sh env        # uv, apt, pip layer, verl clone + install
#   ./setup_pod.sh model      # download Qwen3-8B to the network volume
#   ./setup_pod.sh preflight  # CPU-only checks; MUST pass before GPU spend
#   ./setup_pod.sh all

set -xeuo pipefail
STAGE="${1:-all}"

WORKSPACE=${WORKSPACE:-/workspace}
VERL_DIR=${VERL_DIR:-${WORKSPACE}/verl}
VERL_SHA=${VERL_SHA:-2b0fe51}
CTM_DIR=${CTM_DIR:-${WORKSPACE}/my_ctm}
MODEL_PATH=${MODEL_PATH:-${WORKSPACE}/models/Qwen3-8B}
HF_MODEL_ID=${HF_MODEL_ID:-Qwen/Qwen3-8B}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-q_proj,v_proj}

stage_env() {
    command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"

    # RunPod needs sshd; the verl image ships no openssh package.
    if ! command -v sshd >/dev/null 2>&1; then
        apt-get update && apt-get install -y --no-install-recommends openssh-server && mkdir -p /run/sshd
    fi

    # verl main requires transformers >=5.5.3 (setup.py:43); the image ARG pins 5.3.0
    # and only a later layer may pull it forward, so the effective version is not
    # determined by the ARG. Idempotent.
    uv pip install --system --no-cache "transformers>=5.5.3,!=5.6.0,<5.11"
    uv pip install --system --no-cache zstandard   # slime_port.rollout_writer imports it

    if [ ! -d "${VERL_DIR}/.git" ]; then
        git clone --filter=blob:none https://github.com/volcengine/verl "${VERL_DIR}"
    fi
    git -C "${VERL_DIR}" checkout -q "${VERL_SHA}"
    # --no-deps: do not let pip re-resolve the image's pinned torch/vLLM stack.
    uv pip install --system --no-deps -e "${VERL_DIR}"
}

stage_model() {
    export PATH="${HOME}/.local/bin:${PATH}"
    uv pip install --system --no-cache "huggingface_hub[cli]"
    hf download "${HF_MODEL_ID}" --local-dir "${MODEL_PATH}"
}

stage_preflight() {
    MODEL_PATH="${MODEL_PATH}" LORA_TARGET_MODULES="${LORA_TARGET_MODULES}" python3 - <<'PYEOF'
import importlib, os, sys
ok = True

def check(label, fn):
    global ok
    try:
        print(f"[ OK ] {label}: {fn()}")
    except Exception as e:
        ok = False
        print(f"[FAIL] {label}: {type(e).__name__}: {e}")

import torch, transformers
check("torch", lambda: torch.__version__)
check("transformers", lambda: transformers.__version__)
check("verl imports", lambda: importlib.import_module("verl").__file__)
check("peft", lambda: importlib.import_module("peft").__version__)
check("vllm", lambda: importlib.import_module("vllm").__version__)

# Blackwell: the whole reason this image (CUDA 13 / torch 2.11) was chosen over
# the older CUDA 12.4 line. Fail here, not after the GPU is billed.
check("cuda available", lambda: torch.cuda.is_available())
check("device", lambda: torch.cuda.get_device_name(0))
check("compute capability", lambda: torch.cuda.get_device_capability(0))
check("sm supported by this torch build",
      lambda: f"{torch.cuda.get_arch_list()} (need sm_100 for B200)")

# LoRA targets: verl's FSDP path uses the FLAT model.lora_rank/lora_alpha/
# target_modules keys (verl/workers/config/model.py:122-132); the nested
# model.lora dict is the MEGATRON config and is inert for us. peft matches a
# LIST by suffix, so confirm the suffixes exist and hit what we expect.
targets = os.environ["LORA_TARGET_MODULES"].split(",")
def lora_hits():
    import torch.nn as nn
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(os.environ["MODEL_PATH"])
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg)
    linears = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    hit = [n for n in linears if any(n.endswith(t) for t in targets)]
    if not hit:
        raise AssertionError(f"no nn.Linear matches {targets}")
    return f"{len(hit)}/{len(linears)} linears, e.g. {hit[:2]}"
check(f"lora targets {targets}", lora_hits)

sys.exit(0 if ok else 1)
PYEOF
}

case "${STAGE}" in
    env) stage_env ;;
    model) stage_model ;;
    preflight) stage_preflight ;;
    all) stage_env; stage_model; stage_preflight ;;
    *) echo "unknown stage ${STAGE}"; exit 2 ;;
esac
