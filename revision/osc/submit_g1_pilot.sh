#!/usr/bin/env bash
# Author: Yuqi Zhang
#
# G1 KRAS pilot — one SLURM array element per (arm, task, repeat) cell.
#
#   ssh cardinal
#   cd ~/Qpocket
#   python revision/osc/make_cells.py --arms P M --repeats 3
#   sbatch --array=1-216 revision/osc/submit_g1_pilot.sh
#
# The array range must match the cell count printed by make_cells.py.
# Submitting a wider range than the manifest holds fails the element
# rather than silently running nothing.
#
#SBATCH --job-name=qpr_g1
#SBATCH --account=PGS0423
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
# Measured: 0.24 GB baseline + 0.36 GB per million proposals, so the
# largest cell (2,007,040 per tau across three taus) peaks near 2.4 GB.
# The request also decides the core count -- Cardinal allocates roughly
# 4.84 GB per core -- and these arms are single-threaded, so anything
# above one core's worth of memory is charged and not used.
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --output=revision/results/slurm/qpr_g1_%A_%a.out
#SBATCH --error=revision/results/slurm/qpr_g1_%A_%a.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-${PWD}}"
cd "${REPO_ROOT}"
mkdir -p revision/results/slurm

CELLS="${CELLS:-revision/configs/g1_cells.tsv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/revision/results/g1_pilot}"

# The proposal budget in the manifest is PER TAU, matching how the quantum
# arm's shot allocation was actually spent: that arm ran the full
# allocation once for each of its three taus, so a cell's total proposal
# count is n_proposals x len(taus). The budget is also length-adaptive, so
# a single flat number would over-budget short fragments and under-budget
# long ones.
LEAKAGE_MODE="${LEAKAGE_MODE:-full}"

if [[ ! -f "${CELLS}" ]]; then
    echo "ERROR: cell manifest not found: ${CELLS}" >&2
    echo "Run: python revision/osc/make_cells.py --arms P M --repeats 3" >&2
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
NRES="$(echo "${LINE}" | cut -f5)"
N_PROPOSALS="$(echo "${LINE}" | cut -f6)"

if [[ -z "${N_PROPOSALS}" || ! "${N_PROPOSALS}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: cell ${IDX} has no valid n_proposals (${N_PROPOSALS@Q})." >&2
    echo "Regenerate the manifest with make_cells.py." >&2
    exit 4
fi

echo "[cell ${IDX}] arm=${ARM} task=${TASK} repeat=${REP} n_res=${NRES} n_proposals=${N_PROPOSALS}"
echo "[cell ${IDX}] node=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-?}"

module load miniconda3/24.1.2-py310
# shellcheck disable=SC1091
source activate "/fs/scratch/PGS0423/${USER}/envs/qpocket-revision"

# The arms are single-threaded by design; stop BLAS from oversubscribing
# the one allocated core and skewing the recorded CPU time.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Run the interpreter directly rather than through srun. This is a single
# process on a single node, so a job step buys nothing -- and srun refuses
# to start when the site's SLURM_CPUS_PER_TASK (derived from the memory
# request) disagrees with the SLURM_TRES_PER_TASK implied by
# --cpus-per-task, which is exactly the situation here.
python revision/run_arm.py \
    --arm "${ARM}" \
    --task-id "${TASK}" \
    --repeat "${REP}" \
    --budget-mode proposal \
    --n-proposals "${N_PROPOSALS}" \
    --leakage-mode "${LEAKAGE_MODE}" \
    --output-root "${OUTPUT_ROOT}"

echo "[cell ${IDX}] complete"
