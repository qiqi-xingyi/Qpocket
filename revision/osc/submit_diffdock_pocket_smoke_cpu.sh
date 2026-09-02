#!/usr/bin/env bash
# CPU fallback smoke test for the pinned DiffDock-Pocket environment on OSC.
# The official PyTorch 1.13.1 build predates H100/sm_90, so this path preserves
# the released environment instead of changing PyTorch and PyG under the model.
# One sample and one denoising step test compatibility only.

#SBATCH --job-name=qpr_ddp_cpu
#SBATCH --account=PGS0423
#SBATCH --partition=debug
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/fs/scratch/PGS0423/%u/diffdock/smoke_cpu/slurm_%j.out
#SBATCH --error=/fs/scratch/PGS0423/%u/diffdock/smoke_cpu/slurm_%j.err

set -euo pipefail

DD_ROOT="/fs/scratch/PGS0423/${USER}/diffdock"
DD_REPO="${DD_ROOT}/repo"
DD_PYTHON="/fs/scratch/PGS0423/${USER}/envs/diffdock-pocket/bin/python"
DD_OUT="${DD_ROOT}/smoke_cpu/results"

cd "${DD_REPO}"
mkdir -p "${DD_OUT}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TORCH_HOME="${DD_ROOT}/torch"
export CUDA_VISIBLE_DEVICES=""

echo "node=$(hostname)"
"${DD_PYTHON}" -c \
  "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'threads', torch.get_num_threads())"

"${DD_PYTHON}" inference.py \
  --complex_name 3dpf_cpu_smoke \
  --protein_path example_data/3dpf_protein.pdb \
  --ligand example_data/3dpf_ligand.sdf \
  --model_dir workdir/score_model \
  --filtering_model_dir workdir/confidence_model \
  --out_dir "${DD_OUT}" \
  --batch_size 1 \
  --samples_per_complex 1 \
  --inference_steps 1 \
  --actual_steps 1 \
  --keep_local_structures \
  --no_final_step_noise

find "${DD_OUT}" -maxdepth 3 -type f -printf '%p %s bytes\n' | sort
