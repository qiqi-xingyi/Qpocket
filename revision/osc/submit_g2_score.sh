#!/usr/bin/env bash
# Author: Yuqi Zhang
#
# Score the G2 comparator arms to the frozen endpoints.
#
#   sbatch --array=1-N --export=ALL,ARM=R revision/osc/submit_g2_score.sh
#
#SBATCH --job-name=qpr_g2s
#SBATCH --account=PGS0423
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=revision/results/slurm/%x_%A_%a.out
#SBATCH --error=revision/results/slurm/%x_%A_%a.err

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-${PWD}}"
cd "${REPO_ROOT}"
mkdir -p revision/results/slurm

ARM="${ARM:?set ARM=R|RB|I via --export}"
CELLS="${CELLS:-revision/configs/g2_cells.tsv}"

IDX="${SLURM_ARRAY_TASK_ID:?submit as an array job}"
LINE="$(awk -F'\t' -v i="${IDX}" 'NR>1 { sub(/\r$/, "") } NR>1 && $1==i {print; found=1} END{if(!found) exit 3}' "${CELLS}")" || {
    echo "ERROR: no cell ${IDX} in ${CELLS}" >&2; exit 3; }
TASK="$(echo "${LINE}" | cut -f3)"
REP="$(echo "${LINE}"  | cut -f4)"

echo "[score ${ARM} cell ${IDX}] task=${TASK} repeat=${REP} node=$(hostname)"

module load miniconda3/24.1.2-py310
# shellcheck disable=SC1091
source activate "/fs/scratch/PGS0423/${USER}/envs/qpocket-revision"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# A cell whose arm produced nothing is skipped rather than failed: the
# absence is already recorded by the arm's own run record.
if [[ ! -f "revision/results/g2_local/${ARM}/${TASK}/repeat_${REP}/conformations.npz" ]]; then
    echo "[score ${ARM} cell ${IDX}] no conformations; nothing to score"
    exit 0
fi

python revision/score_cell.py \
    --arm "${ARM}" \
    --task-id "${TASK}" \
    --repeat "${REP}" \
    --coords-npz \
    --results-root revision/results/g2_local

echo "[score ${ARM} cell ${IDX}] complete"
