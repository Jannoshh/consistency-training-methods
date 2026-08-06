#!/bin/bash
# Cold-start recovery on a fresh pod (Phase 6 target: < 15 minutes).
# Run LOCALLY. Rents nothing itself — assumes a pod already exists (rent per
# the approved spec first), then rebuilds the working environment on it from
# the synced bundle + env.lock pins and resumes from the latest checkpoint in
# object storage / the network volume.
#
# Usage: ./restart.sh <ssh-host> [port]
set -euo pipefail

HOST=$1
PORT=${2:-22}
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# 1. Ship the experiment bundle (includes env.lock from the previous host).
"${SCRIPT_DIR}/scripts/sync_to_pod.sh" "${HOST}" "${PORT}"

# 2. Rebuild the environment and verify hardware.
ssh -p "${PORT}" "root@${HOST}" "cd /workspace/rmct && bash scripts/setup_pod.sh"

# 3. Restore checkpoints/rollout logs from durable storage if the volume is
#    empty (object-storage sync lands in Phase 6; until then the RunPod
#    network volume is the recovery source).
ssh -p "${PORT}" "root@${HOST}" "ls /workspace/runs 2>/dev/null || echo 'WARN: no runs on volume — restore from object storage before resuming'"

echo "environment ready. Resume training with the matching run script and --load pointing at the latest checkpoint."
