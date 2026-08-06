#!/usr/bin/env bash
# Preflight validation for RMCT-on-slime instances. Run first on every new pod.
# Exits non-zero on any hard failure. Soft warnings are prefixed WARN.
#
# Usage: ./preflight.sh [--dev]
#   --dev   single-GPU dev tier: skip the NVLink topology requirement.
set -uo pipefail

DEV_TIER=0
[[ "${1:-}" == "--dev" ]] && DEV_TIER=1
FAIL=0
fail() { echo "FAIL: $*" >&2; FAIL=1; }
warn() { echo "WARN: $*" >&2; }
ok() { echo "  ok: $*"; }

echo "== GPU =="
if ! command -v nvidia-smi >/dev/null; then
    fail "nvidia-smi not found"
else
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    N_GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')
    DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    ok "$N_GPU GPU(s), driver $DRIVER"

    # CUDA driver capability: slime image is CUDA 12.9 → driver >= 575 preferred,
    # >= 525 minimally usable via forward compat. Hard-fail below 525.
    DRIVER_MAJOR=${DRIVER%%.*}
    if (( DRIVER_MAJOR < 525 )); then
        fail "driver $DRIVER too old for CUDA 12.9 images"
    elif (( DRIVER_MAJOR < 575 )); then
        warn "driver $DRIVER predates CUDA 12.9 native support; relies on forward compat"
    fi

    if (( N_GPU > 1 )) && (( DEV_TIER == 0 )); then
        echo "== Interconnect topology =="
        TOPO=$(nvidia-smi topo -m 2>/dev/null || true)
        echo "$TOPO" | head -$((N_GPU + 2))
        # Every GPU-GPU pair must be NV#; PHB/SYS/PIX anywhere = PCIe riser node.
        PAIRS=$(echo "$TOPO" | grep -oE '\b(NV[0-9]+|PHB|SYS|PIX|PXB|NODE)\b' | sort | uniq -c)
        echo "$PAIRS"
        # Only the GPU x GPU block matters: fields 2..(1+ngpu) of GPU rows.
        # GPU<->NIC columns legitimately show PIX/NODE/SYS on multi-NIC hosts.
        NGPU=$(echo "$TOPO" | grep -c '^GPU')
        if echo "$TOPO" | awk -v n="$NGPU" '/^GPU/ {for (i = 2; i <= n + 1; i++) print $i}' \
                | grep -qE '^(PHB|SYS|PIX|PXB|NODE)$'; then
            fail "non-NVLink GPU pair found (PHB/SYS/PIX/PXB/NODE) — destroy and re-rent"
        else
            ok "all GPU pairs NVLink"
        fi
    fi
fi

echo "== Disk =="
# Megatron 9B full-param checkpoints: ~40-150GB each; keep 3-4x plus rollout
# JSONL headroom. Dev tier (2B) scales down.
NEED_GB=$(( DEV_TIER == 1 ? 200 : 600 ))
for MOUNT in /workspace /root /; do
    [[ -d "$MOUNT" ]] || continue
    FREE_GB=$(df -BG --output=avail "$MOUNT" 2>/dev/null | tail -1 | tr -dc '0-9')
    [[ -n "$FREE_GB" ]] && break
done
if [[ -z "${FREE_GB:-}" ]]; then
    fail "could not determine free disk space"
elif (( FREE_GB < NEED_GB )); then
    fail "only ${FREE_GB}GB free on $MOUNT; need >= ${NEED_GB}GB"
else
    ok "${FREE_GB}GB free on $MOUNT (need ${NEED_GB}GB)"
fi

echo "== Host resources =="
CPUS=$(nproc)
MEM_GB=$(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo)
ok "$CPUS CPUs, ${MEM_GB}GB RAM"
# slime holds a pinned-CPU snapshot of actor params for ref logprobs: 9B bf16
# ≈ 20GB + optimizer/dataloader headroom.
MIN_MEM=$(( DEV_TIER == 1 ? 48 : 160 ))
(( MEM_GB < MIN_MEM )) && fail "need >= ${MIN_MEM}GB host RAM (ref-model CPU snapshot + Megatron)"

echo "== Network =="
check_url() {
    local name=$1 url=$2
    local code
    code=$(curl -s -o /dev/null -m 15 -w '%{http_code}' "$url")
    if [[ "$code" =~ ^(2|3) ]]; then
        ok "$name reachable ($code)"
    else
        fail "$name unreachable (HTTP $code): $url"
    fi
}
check_url "Hugging Face" "https://huggingface.co/api/models/Qwen/Qwen3.5-9B"
check_url "GitHub" "https://github.com/THUDM/slime"
check_url "Docker Hub" "https://hub.docker.com/v2/repositories/slimerl/slime"
# Object storage endpoint (durable checkpoint copy). Set CKPT_ENDPOINT_URL to enable.
if [[ -n "${CKPT_ENDPOINT_URL:-}" ]]; then
    check_url "object storage" "$CKPT_ENDPOINT_URL"
else
    warn "CKPT_ENDPOINT_URL unset — object-storage reachability not checked"
fi
# Judge endpoint: the mcq-bias trait classifier is a local answer parser (no
# API judge in the hot loop). If a judge is ever configured, set JUDGE_URL.
if [[ -n "${JUDGE_URL:-}" ]]; then
    check_url "judge endpoint" "$JUDGE_URL"
    T=$(curl -s -o /dev/null -m 15 -w '%{time_total}' "$JUDGE_URL")
    ok "judge latency ${T}s"
fi

echo "== HF download throughput (8MB sample) =="
SPEED=$(curl -s -o /dev/null -m 60 -r 0-8388607 -w '%{speed_download}' \
    "https://huggingface.co/Qwen/Qwen3.5-2B/resolve/main/model.safetensors.index.json" || echo 0)
SPEED_MB=$(awk "BEGIN {printf \"%.0f\", $SPEED/1048576}")
if (( SPEED_MB < 5 )); then
    warn "HF download ${SPEED_MB}MB/s — model pulls will be slow"
else
    ok "HF download ~${SPEED_MB}MB/s"
fi

echo
if (( FAIL )); then
    echo "PREFLIGHT FAILED"
    exit 1
fi
echo "PREFLIGHT PASSED"
