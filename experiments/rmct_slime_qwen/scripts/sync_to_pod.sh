#!/bin/bash
# Rsync the experiment bundle to a pod's /workspace/rmct. Run locally.
# Usage: ./sync_to_pod.sh <ssh-host-or-alias> [port]
set -euo pipefail

HOST=$1
PORT=${2:-22}
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)"
EXP="${REPO_ROOT}/experiments/rmct_slime_qwen"

# Provenance stamps regenerated on every sync.
git -C "${REPO_ROOT}" rev-parse HEAD > "${EXP}/CTM_COMMIT"
grep -E "mcq[-_]bias" "${REPO_ROOT}/requirements.txt" > "${EXP}/requirements-pin.txt"

rsync -avz -e "ssh -p ${PORT}" \
    --exclude '__pycache__' --exclude '.pytest_cache' \
    "${EXP}/" "root@${HOST}:/workspace/rmct/"
echo "synced to root@${HOST}:/workspace/rmct (ctm commit $(cat "${EXP}/CTM_COMMIT"))"
