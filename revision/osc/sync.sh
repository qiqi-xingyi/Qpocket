#!/usr/bin/env bash
# Author: Yuqi Zhang
# Push the working tree to OSC Cardinal. Run from the project root on the
# local machine.
#
#   bash revision/osc/sync.sh
#
# Results are NOT pulled by this script — use pull_results.sh, which is
# one-way in the other direction so a sync can never overwrite run output.
set -euo pipefail

REMOTE="${REMOTE:-cardinal}"
REMOTE_ROOT="${REMOTE_ROOT:-Project/Qpocket}"
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "[sync] ${LOCAL_ROOT} -> ${REMOTE}:${REMOTE_ROOT}"

# .git is not transferred, so a run on the remote cannot read the commit
# itself. Stamp it into a file that IS transferred. The stamp describes the
# source at transfer time, which is not the same claim as observing the
# repository at run time, and the run record labels it as such.
STAMP="${LOCAL_ROOT}/revision/configs/source_commit.json"
{
    printf '{\n'
    printf '  "commit": "%s",\n' "$(git -C "${LOCAL_ROOT}" rev-parse HEAD 2>/dev/null || echo null)"
    printf '  "branch": "%s",\n' "$(git -C "${LOCAL_ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo null)"
    printf '  "dirty": %s,\n' "$([ -n "$(git -C "${LOCAL_ROOT}" status --porcelain 2>/dev/null)" ] && echo true || echo false)"
    printf '  "stamped_at_utc": "%s",\n' "$(date -u +%Y-%m-%dT%H:%M:%S+00:00)"
    printf '  "stamped_from_host": "%s"\n' "$(hostname)"
    printf '}\n'
} > "${STAMP}"
echo "[sync] stamped source commit -> revision/configs/source_commit.json"

rsync -avh --progress \
    --exclude '.git/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '.DS_Store' \
    --exclude 'result/' \
    --exclude 'logs/' \
    --exclude 'revision/results/' \
    "${LOCAL_ROOT}/" "${REMOTE}:${REMOTE_ROOT}/"

# The SBATCH --output directive is resolved by SLURM at job launch, before
# any line of the script body runs. If the log directory does not exist the
# element fails before it can create one, so it is created here instead --
# and it must be created remotely, because revision/results/ is excluded
# from the transfer above.
echo "[sync] ensuring remote output directories exist"
ssh "${REMOTE}" "mkdir -p '${REMOTE_ROOT}/revision/results/slurm'"

echo "[sync] done."
