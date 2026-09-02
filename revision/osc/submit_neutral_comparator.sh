#!/usr/bin/env bash
# Author: Yuqi Zhang
#
# A comparator on the neutral fragments, so it has a background of its own.
# Ranking a method's Switch-II fragment against its own background is the
# only way to ask the localisation question of two methods symmetrically;
# comparing raw response values across methods compares numbers that were
# never on one scale.
#
#SBATCH --job-name=qpr_nc
#SBATCH --account=PGS0423
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=revision/results/slurm/%x_%A_%a.out
#SBATCH --error=revision/results/slurm/%x_%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-${PWD}}"
mkdir -p revision/results/slurm

ARM="${ARM:?set ARM=I|B}"
CELLS="${CELLS:-revision/configs/g2_neutral_cells.tsv}"
TASKS="${TASKS:-inputs/kras_neutral_tasks.csv}"
NSTRUCT="${NSTRUCT:-50}"

IDX="${SLURM_ARRAY_TASK_ID:?submit as an array job}"
LINE="$(awk -F'\t' -v i="${IDX}" 'NR>1 { sub(/\r$/, "") } NR>1 && $1==i {print}' "${CELLS}")"
TASK="$(echo "${LINE}" | cut -f3)"
[[ -n "${TASK}" ]] || { echo "no cell ${IDX}" >&2; exit 3; }

echo "[neutral ${ARM} cell ${IDX}] task=${TASK} node=$(hostname)"
module load miniconda3/24.1.2-py310
# shellcheck disable=SC1091
source activate "/fs/scratch/PGS0423/${USER}/envs/qpocket-revision"
export OMP_NUM_THREADS=1

if [[ "${ARM}" == "I" ]]; then
    module load rosetta/3.12
    python revision/arms/induced_fit_arm.py --task-id "${TASK}" \
        --tasks "${TASKS}" --nstruct "${NSTRUCT}" \
        --output-root revision/results/g2_neutral_I
else
    # CPU rather than GPU: the GPU queue on this cluster was four days
    # deep, and a case takes fifteen minutes on sixteen cores against
    # fifty seconds on a card. The ensembles agree -- 12.69 A spread on
    # CPU against 12.94 A on GPU for the same case -- so the choice costs
    # time and not comparability.
    python revision/arms/boltz_arm.py --task-id "${TASK}" \
        --tasks "${TASKS}" --samples 20 \
        --accelerator "${ACCEL:-cpu}" \
        --output-root revision/results/g2_neutral_B
fi
echo "[neutral ${ARM} cell ${IDX}] complete"
