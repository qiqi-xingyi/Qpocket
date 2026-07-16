# Author: Yuqi Zhang
"""RMSD oracle — pick the candidate with smallest CA-RMSD vs reference."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from oracle_eval.types import OracleBestResult, OracleCandidate
from ras_folding.postprocess.pdb_export import write_ca_pdb


_EPS = 1e-12


class RMSDOracleSelector:
    def __init__(self, use_kabsch: bool = True) -> None:
        self.use_kabsch = bool(use_kabsch)

    # ------------------------------------------------------------------ #
    def select_best(
        self,
        task_id: str,
        candidates: List[OracleCandidate],
        reference_coords: Optional[np.ndarray],
        output_dir: Path,
        sequence: Optional[Union[str, Sequence[str]]] = None,
    ) -> OracleBestResult:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if reference_coords is None:
            return self._empty_result(
                task_id, output_dir, n_considered=len(candidates),
                reference_available=False,
            )

        ref = np.asarray(reference_coords, dtype=np.float64)
        n_skipped = 0
        best: Optional[OracleCandidate] = None
        best_rmsd = math.inf

        for c in candidates:
            coords = np.asarray(c.coords, dtype=np.float64)
            if coords.shape != ref.shape:
                n_skipped += 1
                continue
            if not np.all(np.isfinite(coords)):
                n_skipped += 1
                continue
            d = self._rmsd(coords, ref)
            if d < best_rmsd:
                best_rmsd = d
                best = c

        if best is None:
            return self._empty_result(
                task_id, output_dir,
                n_considered=len(candidates),
                reference_available=True,
                n_skipped_shape=n_skipped,
            )

        # write oracle_best_ca.pdb — use the shared write_ca_pdb so residue
        # names track the actual sequence (PULCHRA cannot rebuild UNK).
        pdb_path = output_dir / "oracle_best_ca.pdb"
        sequence_available = sequence is not None
        sequence_length: Optional[int] = None
        if sequence_available:
            try:
                sequence_length = len(sequence)  # type: ignore[arg-type]
            except TypeError:
                sequence_length = None
        write_ca_pdb(best.coords, sequence, pdb_path)

        summary = {
            "task_id": task_id,
            "best_rmsd": float(best_rmsd),
            "best_source": best.source,
            "best_bitstring": best.bitstring,
            "best_is_dense": bool(best.is_dense),
            "parent_bitstring": best.parent_bitstring,
            "n_candidates_considered": len(candidates),
            "n_candidates_skipped_shape": int(n_skipped),
            "full_energy": best.full_energy,
            "refined_score": best.refined_score,
            "refined_weight": best.refined_weight,
            "use_kabsch": self.use_kabsch,
            "sequence_available_for_pdb": bool(sequence_available),
            "sequence_length": sequence_length,
        }
        (output_dir / "oracle_best_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8",
        )

        return OracleBestResult(
            task_id=task_id,
            best_candidate=best,
            best_rmsd=float(best_rmsd),
            reference_available=True,
            n_candidates_considered=len(candidates),
            output_pdb=pdb_path,
            summary=summary,
        )

    # ------------------------------------------------------------------ #
    def _rmsd(self, a: np.ndarray, b: np.ndarray) -> float:
        if self.use_kabsch:
            R = _kabsch(a - a.mean(0), b - b.mean(0))
            a_aligned = (a - a.mean(0)) @ R
            diffs = a_aligned - (b - b.mean(0))
        else:
            diffs = a - b
        return float(math.sqrt(
            max(float(np.mean(np.sum(diffs * diffs, axis=1))), 0.0)
        ))

    # ------------------------------------------------------------------ #
    def _empty_result(
        self,
        task_id: str,
        output_dir: Path,
        *,
        n_considered: int,
        reference_available: bool,
        n_skipped_shape: int = 0,
    ) -> OracleBestResult:
        summary = {
            "task_id": task_id,
            "reference_available": reference_available,
            "n_candidates_considered": n_considered,
            "n_candidates_skipped_shape": n_skipped_shape,
            "best_rmsd": None,
            "use_kabsch": self.use_kabsch,
        }
        (output_dir / "oracle_best_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8",
        )
        return OracleBestResult(
            task_id=task_id,
            best_candidate=None,
            best_rmsd=None,
            reference_available=reference_available,
            n_candidates_considered=n_considered,
            output_pdb=None,
            summary=summary,
        )


def _kabsch(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Optimal rotation (Kabsch) — only used when use_kabsch=True."""
    H = P.T @ Q
    U, _S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    return U @ D @ Vt


__all__ = ["RMSDOracleSelector"]
