# Author: Yuqi Zhang
"""ras_folding.scoring — full physical scoring + MJ contact adapter.

Public API:
  MJContactTable, load_mj_table_default  -- MJ matrix adapter
  GeometryTerms                          -- distance / overlap / Rg helpers
  FullEnergyScorer                       -- aggregator producing E_full
"""
from ras_folding.scoring.mj_contact import (
    MJContactTable,
    load_mj_table_default,
)
from ras_folding.scoring.geometry_terms import GeometryTerms
from ras_folding.scoring.full_energy import FullEnergyScorer

__all__ = [
    "MJContactTable",
    "load_mj_table_default",
    "GeometryTerms",
    "FullEnergyScorer",
]
