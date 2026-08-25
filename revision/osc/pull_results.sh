#!/usr/bin/env bash
# Author: Yuqi Zhang
# Pull revision run outputs back from OSC. One-way: OSC -> local.
set -euo pipefail

REMOTE="${REMOTE:-cardinal}"
REMOTE_RESULTS="${REMOTE_RESULTS:-Project/Qpocket/revision/results}"
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCAL_RESULTS="${LOCAL_ROOT}/revision/results"

mkdir -p "${LOCAL_RESULTS}"
echo "[pull] ${REMOTE}:${REMOTE_RESULTS} -> ${LOCAL_RESULTS}"
rsync -avh --progress "${REMOTE}:${REMOTE_RESULTS}/" "${LOCAL_RESULTS}/"
echo "[pull] done."
