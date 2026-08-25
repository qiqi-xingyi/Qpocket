#!/usr/bin/env bash
# Author: Yuqi Zhang
# One-time environment setup for the revision experiments on OSC Cardinal.
#
#   ssh cardinal
#   bash ~/Project/Qpocket/revision/osc/setup_env.sh
#
# The conda environment lives on scratch, not in $HOME: the solver writes
# tens of thousands of small files and $HOME has a file-count quota.
set -euo pipefail

PROJ_ACCOUNT="PGS0423"
SCRATCH="/fs/scratch/${PROJ_ACCOUNT}/${USER}"
ENV_PREFIX="${SCRATCH}/envs/qpocket-revision"
REPO_ROOT="${REPO_ROOT:-${HOME}/Project/Qpocket}"

echo "[setup] repo    : ${REPO_ROOT}"
echo "[setup] env     : ${ENV_PREFIX}"

if [[ ! -f "${REPO_ROOT}/environment.yml" ]]; then
    echo "[setup] ERROR: ${REPO_ROOT}/environment.yml not found." >&2
    echo "[setup] Sync the repo to OSC first (see osc/sync.sh)." >&2
    exit 1
fi

module load miniconda3/24.1.2-py310

# Keep package cache on scratch too — same quota reason as the env itself.
export CONDA_PKGS_DIRS="${SCRATCH}/conda_pkgs"
mkdir -p "${CONDA_PKGS_DIRS}" "${SCRATCH}/envs"

if [[ -d "${ENV_PREFIX}" ]]; then
    echo "[setup] environment already exists — updating in place"
    conda env update --prefix "${ENV_PREFIX}" \
        --file "${REPO_ROOT}/environment.yml" --prune
else
    conda env create --prefix "${ENV_PREFIX}" \
        --file "${REPO_ROOT}/environment.yml"
fi

# shellcheck disable=SC1091
source activate "${ENV_PREFIX}"
python -c "import numpy, scipy; print('[setup] numpy', numpy.__version__, '| scipy', scipy.__version__)"
python -c "import qiskit; print('[setup] qiskit', qiskit.__version__)" 2>/dev/null \
    || echo "[setup] NOTE: qiskit not in environment.yml — CPU arms (P/M) do not need it."

echo "[setup] done. Activate with:"
echo "  module load miniconda3/24.1.2-py310 && source activate ${ENV_PREFIX}"
