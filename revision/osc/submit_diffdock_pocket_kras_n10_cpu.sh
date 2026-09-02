#!/usr/bin/env bash
# Ten-sample, full 30-step KRAS DiffDock-Pocket pilot on an OSC CPU node.

#SBATCH --job-name=qpr_ddp_7rpz_n10
#SBATCH --account=PGS0423
#SBATCH --partition=debug
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/fs/scratch/PGS0423/%u/diffdock/kras_7rpz_n10/slurm_%j.out
#SBATCH --error=/fs/scratch/PGS0423/%u/diffdock/kras_7rpz_n10/slurm_%j.err

set -euo pipefail

DD_ROOT="/fs/scratch/PGS0423/${USER}/diffdock"
DD_REPO="${DD_ROOT}/repo"
DD_PYTHON="/fs/scratch/PGS0423/${USER}/envs/diffdock-pocket/bin/python"
RUN_ROOT="${DD_ROOT}/kras_7rpz_n10"
INPUT_ROOT="${RUN_ROOT}/input"
OUTPUT_ROOT="${RUN_ROOT}/results"
SOURCE_PDB="${INPUT_ROOT}/7RPZ.pdb"
PROTEIN_PDB="${INPUT_ROOT}/7RPZ_chainA_protein.pdb"
LIGAND_SDF="${INPUT_ROOT}/6IC.sdf"

mkdir -p "${INPUT_ROOT}" "${OUTPUT_ROOT}"
awk 'substr($0,1,6) == "ATOM  " && substr($0,22,1) == "A" {print} END {print "END"}' \
  "${SOURCE_PDB}" > "${PROTEIN_PDB}"

cd "${DD_REPO}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TORCH_HOME="${DD_ROOT}/torch"
export CUDA_VISIBLE_DEVICES=""

echo "node=$(hostname)"
"${DD_PYTHON}" -c \
  "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'threads', torch.get_num_threads())"

"${DD_PYTHON}" inference.py \
  --complex_name 7RPZ_6IC_n10_full30 \
  --protein_path "${PROTEIN_PDB}" \
  --ligand "${LIGAND_SDF}" \
  --model_dir workdir/score_model \
  --filtering_model_dir workdir/confidence_model \
  --out_dir "${OUTPUT_ROOT}" \
  --batch_size 10 \
  --samples_per_complex 10 \
  --inference_steps 30 \
  --actual_steps 30 \
  --keep_local_structures \
  --no_final_step_noise

RESULT_DIR="${OUTPUT_ROOT}/index0___7RPZ_chainA_protein.pdb___6IC.sdf"
"${DD_PYTHON}" "${INPUT_ROOT}/analyze_diffdock_pocket_ensemble.py" \
  --native-protein "${PROTEIN_PDB}" \
  --native-ligand "${LIGAND_SDF}" \
  --result-dir "${RESULT_DIR}" \
  --output "${RUN_ROOT}/analysis.json"
