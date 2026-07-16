# Author: Yuqi Zhang
"""ras_folding.postprocess — final post-processing of refined candidates.

Public API:
  PredictionResult, BasinSummary           -- dataclasses
  PredictionPostProcessor                  -- end-to-end orchestrator
  ca_rmsd, pairwise_ca_rmsd                -- RMSD helpers (no Kabsch)
  dedup_by_bitstring, dedup_by_structure   -- dedup helpers
  cluster_basins_by_rmsd                   -- connected-component basin clustering
  compute_basin_summaries                  -- per-basin aggregates
  rank_basins                              -- basin score + ranking
  write_ca_pdb                             -- CA-only PDB export
"""
from ras_folding.postprocess.prediction_types import (
    BasinSummary,
    PredictionResult,
)
from ras_folding.postprocess.rmsd import ca_rmsd, pairwise_ca_rmsd
from ras_folding.postprocess.dedup import (
    dedup_by_bitstring,
    dedup_by_structure,
)
from ras_folding.postprocess.clustering import cluster_basins_by_rmsd
from ras_folding.postprocess.summary import compute_basin_summaries
from ras_folding.postprocess.selector import rank_basins
from ras_folding.postprocess.pdb_export import write_ca_pdb
from ras_folding.postprocess.prediction_postprocessor import (
    PredictionPostProcessor,
)

__all__ = [
    "BasinSummary",
    "PredictionResult",
    "PredictionPostProcessor",
    "ca_rmsd",
    "pairwise_ca_rmsd",
    "dedup_by_bitstring",
    "dedup_by_structure",
    "cluster_basins_by_rmsd",
    "compute_basin_summaries",
    "rank_basins",
    "write_ca_pdb",
]
