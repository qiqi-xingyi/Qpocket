# Author: Yuqi Zhang
"""ras_folding.densify — optional perturbation-based dense filler.

Public API:
  DenseFillResult                 -- result dataclass
  PerturbationDenseFiller         -- generate child candidates from valid parents
  perturb_bond_vectors            -- bond-angle perturbation primitive
  local_ca_rmsd                   -- unaligned CA RMSD between two coord arrays
  assign_dense_weights            -- per-parent mass-conserving weight policy
  write_dense_outputs             -- persist DenseFillResult to disk
"""
from ras_folding.densify.dense_types import DenseFillResult
from ras_folding.densify.angular_perturb import (
    local_ca_rmsd,
    perturb_bond_vectors,
)
from ras_folding.densify.weight_policy import assign_dense_weights
from ras_folding.densify.dense_filler import (
    PerturbationDenseFiller,
    write_dense_outputs,
)

__all__ = [
    "DenseFillResult",
    "PerturbationDenseFiller",
    "perturb_bond_vectors",
    "local_ca_rmsd",
    "assign_dense_weights",
    "write_dense_outputs",
]
