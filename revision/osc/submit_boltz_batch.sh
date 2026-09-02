#!/usr/bin/env bash
# Author: Yuqi Zhang
#
# Boltz over a slice of the cell list, several cases per allocation.
#
# Each prediction takes under a minute, so asking the scheduler for one
# GPU per case makes it queue hundreds of allocations for three hours of
# work. One allocation walks a stride of the list instead.
#
#SBATCH --job-name=qpr_bzb
#SBATCH --account=PGS0423
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=revision/results/slurm/%x_%A_%a.out
#SBATCH --error=revision/results/slurm/%x_%A_%a.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-${PWD}}"
mkdir -p revision/results/slurm

CELLS="${CELLS:-revision/configs/g2_neutral_cells.tsv}"
TASKS="${TASKS:-inputs/kras_neutral_tasks.csv}"
OUT="${OUT:-revision/results/g2_neutral_B}"
NSLICE="${NSLICE:-10}"
IDX="${SLURM_ARRAY_TASK_ID:?submit as an array job}"

module load miniconda3/24.1.2-py310
# shellcheck disable=SC1091
source activate "/fs/scratch/PGS0423/${USER}/envs/qpocket-revision"

n_done=0; n_fail=0
# Stride assignment keeps every slice the same length without needing the
# list size in advance.
while IFS=$'\t' read -r idx arm task rep rest; do
    [[ "${idx}" == "index" ]] && continue
    (( (idx - 1) % NSLICE == IDX - 1 )) || continue
    if python revision/arms/boltz_arm.py --task-id "${task}" \
            --tasks "${TASKS}" --samples 20 --output-root "${OUT}"; then
        n_done=$((n_done + 1))
    else
        n_fail=$((n_fail + 1))
    fi
done < "${CELLS}"

echo "[slice ${IDX}/${NSLICE}] done=${n_done} failed=${n_fail}"
