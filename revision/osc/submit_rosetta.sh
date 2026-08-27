#!/usr/bin/env bash
# Author: Yuqi Zhang
#
# Arm R -- one SLURM array element per (task, repeat) cell.
#
#   python revision/osc/make_cells.py --arms R --repeats 1 \
#          --out revision/configs/g2_cells.tsv
#   sbatch --array=1-144 --export=ALL,METHOD=kic,NSTRUCT=50 \
#          revision/osc/submit_rosetta.sh
#   sbatch --array=1-144 --export=ALL,METHOD=backrub,NSTRUCT=50 \
#          --job-name=qpr_brb revision/osc/submit_rosetta.sh
#
#SBATCH --job-name=qpr_ros
#SBATCH --account=PGS0423
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --output=revision/results/slurm/%x_%A_%a.out
#SBATCH --error=revision/results/slurm/%x_%A_%a.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-${PWD}}"
cd "${REPO_ROOT}"
mkdir -p revision/results/slurm

CELLS="${CELLS:-revision/configs/g2_cells.tsv}"
NSTRUCT="${NSTRUCT:-200}"
METHOD="${METHOD:-kic}"
NTRIALS="${NTRIALS:-1000}"

IDX="${SLURM_ARRAY_TASK_ID:?submit as an array job}"
LINE="$(awk -F'\t' -v i="${IDX}" 'NR>1 { sub(/\r$/, "") } NR>1 && $1==i {print; found=1} END{if(!found) exit 3}' "${CELLS}")" || {
    echo "ERROR: no cell with index ${IDX} in ${CELLS}" >&2; exit 3; }
TASK="$(echo "${LINE}" | cut -f3)"
REP="$(echo "${LINE}"  | cut -f4)"

echo "[cell ${IDX}] method=${METHOD} task=${TASK} repeat=${REP} nstruct=${NSTRUCT}"
echo "[cell ${IDX}] node=$(hostname)"

module load rosetta/3.12
module load miniconda3/24.1.2-py310
# shellcheck disable=SC1091
source activate "/fs/scratch/PGS0423/${USER}/envs/qpocket-revision"

export OMP_NUM_THREADS=1

if [[ "${METHOD}" == "induced_fit" ]]; then
    python revision/arms/induced_fit_arm.py \
        --task-id "${TASK}" \
        --repeat "${REP}" \
        --nstruct "${NSTRUCT}"
else
    python revision/arms/rosetta_arm.py \
        --method "${METHOD}" \
        --task-id "${TASK}" \
        --repeat "${REP}" \
        --nstruct "${NSTRUCT}" \
        --ntrials "${NTRIALS}"
fi

echo "[cell ${IDX}] complete"
