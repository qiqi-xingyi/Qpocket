#!/usr/bin/env bash
# Author: Yuqi Zhang
#
# G1 scoring — carry each cell's candidates through the shared downstream
# and measure the frozen endpoints. One array element per cell.
#
#   ssh cardinal
#   cd ~/Project/Qpocket
#   python revision/osc/make_score_cells.py
#   sbatch --array=1-252 revision/osc/submit_g1_score.sh
#
#SBATCH --job-name=qpr_score
#SBATCH --account=PGS0423
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
# Decoding the largest cell holds roughly 225k candidates in memory. The
# request also decides the core count -- Cardinal allocates about 4.84 GB
# per core -- and this is a single-threaded job, so more memory than one
# core's worth is charged and not used.
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=revision/results/slurm/qpr_score_%A_%a.out
#SBATCH --error=revision/results/slurm/qpr_score_%A_%a.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-${PWD}}"
cd "${REPO_ROOT}"
mkdir -p revision/results/slurm

CELLS="${CELLS:-revision/configs/g1_score_cells.tsv}"
RESULTS_ROOT="${RESULTS_ROOT:-revision/results/g1_pilot}"

if [[ ! -f "${CELLS}" ]]; then
    echo "ERROR: scoring manifest not found: ${CELLS}" >&2
    echo "Run: python revision/osc/make_score_cells.py" >&2
    exit 1
fi

IDX="${SLURM_ARRAY_TASK_ID:?this script must be submitted as an array job}"
LINE="$(awk -F'\t' -v i="${IDX}" 'NR>1 { sub(/\r$/, "") } NR>1 && $1==i {print; found=1} END{if(!found) exit 3}' "${CELLS}")" || {
    echo "ERROR: no cell with index ${IDX} in ${CELLS}" >&2
    exit 3
}
ARM="$(echo "${LINE}"  | cut -f2)"
TASK="$(echo "${LINE}" | cut -f3)"
REP="$(echo "${LINE}"  | cut -f4)"

echo "[score ${IDX}] arm=${ARM} task=${TASK} repeat=${REP}"
echo "[score ${IDX}] node=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-?}"

module load miniconda3/24.1.2-py310
# shellcheck disable=SC1091
source activate "/fs/scratch/PGS0423/${USER}/envs/qpocket-revision"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python revision/score_cell.py \
    --arm "${ARM}" \
    --task-id "${TASK}" \
    --repeat "${REP}" \
    --results-root "${RESULTS_ROOT}"

echo "[score ${IDX}] complete"
