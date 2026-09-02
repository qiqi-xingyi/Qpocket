#!/usr/bin/env bash
# Full-step DiffDock-Pocket re-docking check on one real KRAS complex.
# Computation is intentionally restricted to an OSC CPU compute node because
# the released PyTorch 1.13.1 environment cannot execute on OSC H100/sm_90.

#SBATCH --job-name=qpr_ddp_7rpz
#SBATCH --account=PGS0423
#SBATCH --partition=debug
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/fs/scratch/PGS0423/%u/diffdock/kras_7rpz_full/slurm_%j.out
#SBATCH --error=/fs/scratch/PGS0423/%u/diffdock/kras_7rpz_full/slurm_%j.err

set -euo pipefail

DD_ROOT="/fs/scratch/PGS0423/${USER}/diffdock"
DD_REPO="${DD_ROOT}/repo"
DD_PYTHON="/fs/scratch/PGS0423/${USER}/envs/diffdock-pocket/bin/python"
RUN_ROOT="${DD_ROOT}/kras_7rpz_full"
INPUT_ROOT="${RUN_ROOT}/input"
OUTPUT_ROOT="${RUN_ROOT}/results"
SOURCE_PDB="${INPUT_ROOT}/7RPZ.pdb"
PROTEIN_PDB="${INPUT_ROOT}/7RPZ_chainA_protein.pdb"
LIGAND_SDF="${INPUT_ROOT}/6IC.sdf"

mkdir -p "${INPUT_ROOT}" "${OUTPUT_ROOT}"

# DiffDock-Pocket expects the receptor and ligand in separate files.  Keep
# only chain-A ATOM records in the receptor; the 6IC ligand is supplied by SDF.
awk 'substr($0,1,6) == "ATOM  " && substr($0,22,1) == "A" {print} END {print "END"}' \
  "${SOURCE_PDB}" > "${PROTEIN_PDB}"

cd "${DD_REPO}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TORCH_HOME="${DD_ROOT}/torch"
export CUDA_VISIBLE_DEVICES=""

echo "node=$(hostname)"
echo "source_pdb=${SOURCE_PDB}"
echo "protein_atoms=$(grep -c '^ATOM' "${PROTEIN_PDB}")"
echo "ligand_atoms=$(awk 'NR == 4 {print $1}' "${LIGAND_SDF}")"
"${DD_PYTHON}" -c \
  "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'threads', torch.get_num_threads())"

"${DD_PYTHON}" inference.py \
  --complex_name 7RPZ_6IC_full30 \
  --protein_path "${PROTEIN_PDB}" \
  --ligand "${LIGAND_SDF}" \
  --model_dir workdir/score_model \
  --filtering_model_dir workdir/confidence_model \
  --out_dir "${OUTPUT_ROOT}" \
  --batch_size 1 \
  --samples_per_complex 1 \
  --inference_steps 30 \
  --actual_steps 30 \
  --keep_local_structures \
  --no_final_step_noise

find "${OUTPUT_ROOT}" -maxdepth 3 -type f -printf '%p %s bytes\n' | sort
