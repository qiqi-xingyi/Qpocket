# Author: Yuqi Zhang
"""ras_folding.refinement — SQD-inspired conformational subspace refinement.

Public API:
  RefinedCandidate, RefinementResult       -- result dataclasses
  rmsd_kernel_coupling                     -- coupling builder (KNN RMSD kernel)
  build_effective_hamiltonian              -- assemble H_eff[i,j]
  SubspaceDiagonalizationRefiner           -- end-to-end refiner
"""
from ras_folding.refinement.refined_types import (
    RefinedCandidate,
    RefinementResult,
)
from ras_folding.refinement.coupling import (
    pairwise_rmsd_matrix,
    rmsd_kernel_coupling,
)
from ras_folding.refinement.effective_hamiltonian import (
    build_effective_hamiltonian,
)
from ras_folding.refinement.subspace_diagonalization import (
    SubspaceDiagonalizationRefiner,
)

__all__ = [
    "RefinedCandidate",
    "RefinementResult",
    "pairwise_rmsd_matrix",
    "rmsd_kernel_coupling",
    "build_effective_hamiltonian",
    "SubspaceDiagonalizationRefiner",
]
