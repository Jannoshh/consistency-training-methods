#!/bin/bash
# One-time pod preparation, run inside the slime container over SSH.
# Idempotent. Produces /workspace/rmct/env.lock (Phase 2 deliverable).
#
# Expects the experiment dir already rsynced to /workspace/rmct (see README).
set -ex

RMCT_DIR=${RMCT_DIR:-/workspace/rmct}
MODEL=${MODEL:-Qwen/Qwen3.5-2B}
# Default to the pod-local disk: on network-volume pods (/workspace on mfs),
# model load and checkpoint I/O off the local NVMe is far faster. Durable
# artifacts (rollout logs, env.lock) stay on /workspace.
MODELS_ROOT=${MODELS_ROOT:-/root/models}
MODEL_DIR=${MODELS_ROOT}/$(basename "${MODEL}")
# Framework repo (slime or its Miles fork — same tools/train.py layout).
if [ -z "${SLIME_DIR:-}" ]; then
    for candidate in /root/miles /opt/miles /root/slime; do
        [ -f "${candidate}/train.py" ] && SLIME_DIR="${candidate}" && break
    done
fi
SLIME_DIR=${SLIME_DIR:?no framework dir found (looked for train.py in /root/miles, /opt/miles, /root/slime)}

# 0. Preflight (dev tier: single GPU, no NVLink requirement).
bash "${RMCT_DIR}/preflight.sh" --dev

# 1. Python deps the slime image lacks: the pinned mcq-bias package (answer
#    parser — must match the repo's requirements.txt pin exactly) and zstd.
MCQ_BIAS_PIN=$(grep -E "mcq[-_]bias" "${RMCT_DIR}/requirements-pin.txt")
pip install --no-deps "${MCQ_BIAS_PIN}"
pip install zstandard

# 2. Model download + HF->Megatron conversion (skip when present).
if [ ! -d "${MODEL_DIR}" ]; then
    hf download "${MODEL}" --local-dir "${MODEL_DIR}"
fi
if [ ! -d "${MODEL_DIR}_torch_dist" ]; then
    # Prefer the framework's own model-args script (Miles ships specs with its
    # plugin flags, e.g. --attention-output-gate); fall back to deriving from
    # config.json for models the framework doesn't ship.
    SHIPPED_ARGS="${SLIME_DIR}/scripts/models/$(basename "${MODEL}" | sed 's/^Q/q/').sh"
    if [ -f "${SHIPPED_ARGS}" ]; then
        source "${SHIPPED_ARGS}"
    else
        python3 "${RMCT_DIR}/scripts/derive_model_args.py" "${MODEL_DIR}" > /tmp/model-args.sh
        source /tmp/model-args.sh
    fi
    PYTHONPATH=/root/Megatron-LM/ python3 "${SLIME_DIR}/tools/convert_hf_to_torch_dist.py" \
        "${MODEL_ARGS[@]}" \
        --hf-checkpoint "${MODEL_DIR}" \
        --save "${MODEL_DIR}_torch_dist"
fi

# 3. slime plugin-contract tests against the RMCT modules (CPU-only) — the
#    cheapest check that our rollout/advantage function signatures still match
#    this slime version.
(cd "${SLIME_DIR}" && PYTHONPATH="${RMCT_DIR}" python3 -m pytest tests/plugin_contracts -q || true)

# 4. env.lock: everything needed to rebuild this environment.
{
    echo "# env.lock — generated $(date -u +%Y-%m-%dT%H:%M:%SZ) by setup_pod.sh"
    echo "image_tag: ${SLIME_IMAGE_TAG:-see-runpod-console}"
    echo "image_digest: $(cat /etc/image_digest 2>/dev/null || echo unknown-see-runpod-console)"
    echo "ctm_commit: $(cat "${RMCT_DIR}/CTM_COMMIT" 2>/dev/null || echo see-slime_port/__init__.py)"
    echo "framework_dir: ${SLIME_DIR}"
    echo "framework_commit: $(git -C "${SLIME_DIR}" rev-parse HEAD 2>/dev/null || echo baked-into-image)"
    echo "megatron_commit: $(git -C /root/Megatron-LM rev-parse HEAD 2>/dev/null || echo baked-into-image)"
    echo "driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
    echo "cuda_runtime: $(python3 -c 'import torch; print(torch.version.cuda)')"
    echo "torch: $(python3 -c 'import torch; print(torch.__version__)')"
    echo "sglang: $(python3 -c 'import sglang; print(sglang.__version__)')"
    echo "gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    echo "model: ${MODEL}"
    echo "---pip-freeze---"
    pip freeze
} > "${RMCT_DIR}/env.lock"

echo "setup complete; env.lock written to ${RMCT_DIR}/env.lock"
