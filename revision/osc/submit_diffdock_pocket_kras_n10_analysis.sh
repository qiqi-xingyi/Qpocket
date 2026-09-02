#!/usr/bin/env bash

#SBATCH --job-name=qpr_ddp_n10_eval
#SBATCH --account=PGS0423
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/fs/scratch/PGS0423/%u/diffdock/kras_7rpz_n10/analysis_slurm_%j.out
#SBATCH --error=/fs/scratch/PGS0423/%u/diffdock/kras_7rpz_n10/analysis_slurm_%j.err

set -euo pipefail

DD_ROOT="/fs/scratch/PGS0423/${USER}/diffdock"
RUN_ROOT="${DD_ROOT}/kras_7rpz_n10"
RESULT_DIR="${RUN_ROOT}/results/index0___7RPZ_chainA_protein.pdb___6IC.sdf"
DD_PYTHON="/fs/scratch/PGS0423/${USER}/envs/diffdock-pocket/bin/python"

export CUDA_VISIBLE_DEVICES=""

"${DD_PYTHON}" "${RUN_ROOT}/input/analyze_diffdock_pocket_ensemble.py" \
  --native-protein "${RUN_ROOT}/input/7RPZ_chainA_protein.pdb" \
  --native-ligand "${RUN_ROOT}/input/6IC.sdf" \
  --result-dir "${RESULT_DIR}" \
  --output "${RUN_ROOT}/analysis.json"
