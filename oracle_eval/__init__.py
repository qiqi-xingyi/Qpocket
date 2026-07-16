# Author: Yuqi Zhang
"""oracle_eval — post-hoc oracle-best evaluation.

Reads candidate coordinates from a finished KRAS run and selects the
RMSD-minimum candidate against a reference. This is an evaluation
upper-bound — NOT a real prediction step.

Public API:
  OracleCandidate, OracleBestResult
  CandidateReader
  RMSDOracleSelector
"""
from oracle_eval.types import OracleCandidate, OracleBestResult
from oracle_eval.candidate_reader import CandidateReader
from oracle_eval.rmsd_oracle import RMSDOracleSelector

__all__ = [
    "OracleCandidate",
    "OracleBestResult",
    "CandidateReader",
    "RMSDOracleSelector",
]
