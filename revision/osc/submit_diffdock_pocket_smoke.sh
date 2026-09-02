#!/usr/bin/env bash
# Minimal DiffDock-Pocket installation smoke test on an OSC GPU compute node.
# This checks model loading and one reverse-diffusion inference. It is not a
# scientific comparison run: the denoising trajectory is shortened to two
# steps and only one sample is generated.
#
# Submit from the remote DiffDock-Pocket repository after applying
# diffdock_pocket_official_fix.patch:
#   sbatch ~/Project/Qpocket/revision/osc/submit_diffdock_pocket_smoke.sh

#SBATCH --job-name=qpr_ddp_smoke
#SBATCH --account=PGS0423
#SBATCH --partition=debug
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/fs/scratch/PGS0423/%u/diffdock/smoke/slurm_%j.out
#SBATCH --error=/fs/scratch/PGS0423/%u/diffdock/smoke/slurm_%j.err

set -euo pipefail

DD_ROOT="/fs/scratch/PGS0423/${USER}/diffdock"
DD_REPO="${DD_ROOT}/repo"
DD_PYTHON="/fs/scratch/PGS0423/${USER}/envs/diffdock-pocket/bin/python"
DD_OUT="${DD_ROOT}/smoke/results"

cd "${DD_REPO}"
mkdir -p "${DD_OUT}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TORCH_HOME="${DD_ROOT}/torch"

echo "node=$(hostname)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"${DD_PYTHON}" -c \
  "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

"${DD_PYTHON}" inference.py \
  --complex_name 3dpf_smoke \
  --protein_path example_data/3dpf_protein.pdb \
  --ligand example_data/3dpf_ligand.sdf \
  --model_dir workdir/score_model \
  --filtering_model_dir workdir/confidence_model \
  --out_dir "${DD_OUT}" \
  --batch_size 1 \
  --samples_per_complex 1 \
  --inference_steps 2 \
  --actual_steps 2 \
  --keep_local_structures \
  --no_final_step_noise

find "${DD_OUT}" -maxdepth 3 -type f -printf '%p %s bytes\n' | sort
