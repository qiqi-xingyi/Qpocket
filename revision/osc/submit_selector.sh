#!/usr/bin/env bash
# Author: Yuqi Zhang
#
# Run one selector experiment on Cardinal.
#
#   sbatch --job-name=qpr_sub  --export=SCRIPT=subspace_selector.py \
#          revision/osc/submit_selector.sh
#   sbatch --job-name=qpr_dl   --export=SCRIPT=dl_ranker.py \
#          revision/osc/submit_selector.sh
#
#SBATCH --account=PGS0423
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=08:00:00
#SBATCH --output=revision/results/slurm/%x_%j.out
#SBATCH --error=revision/results/slurm/%x_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-${PWD}}"
cd "${REPO_ROOT}"
mkdir -p revision/results/slurm

SCRIPT="${SCRIPT:?set SCRIPT=<name>.py via --export}"
if [[ ! -f "revision/${SCRIPT}" ]]; then
    echo "ERROR: revision/${SCRIPT} not found" >&2
    exit 1
fi

echo "[selector] script=${SCRIPT} node=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-?}"

module load miniconda3/24.1.2-py310
# shellcheck disable=SC1091
source activate "/fs/scratch/PGS0423/${USER}/envs/qpocket-revision"

# These jobs are dominated by dense linear algebra, so BLAS is left free to
# use the allocated cores rather than pinned to one as the sampling arms
# were.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

python "revision/${SCRIPT}" ${SCRIPT_ARGS:-}

echo "[selector] complete"
