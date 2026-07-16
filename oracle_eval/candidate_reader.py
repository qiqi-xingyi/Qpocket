# Author: Yuqi Zhang
"""Read candidate coordinates from a finished KRAS task directory.

Primary path: `<task_dir>/candidates/all_candidates_index.csv` plus
`all_candidates_coords.npz` (written by KrasFullBatchRunner — see
ras_folding/kras/full_batch_runner.py).

Fallback path: parse `<task_dir>/postprocess/top1_ca.pdb` only. This
yields a single candidate and a warning is emitted; oracle quality
will be limited to that single point.
"""
from __future__ import annotations

import csv
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from oracle_eval.types import OracleCandidate


def _f(x):
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _b(x) -> bool:
    s = str(x).strip().lower() if x is not None else ""
    return s in ("true", "1", "yes")


def _load_ca_only_pdb(path: Path) -> Optional[np.ndarray]:
    """Read CA atoms from a CA-only PDB into (n, 3) ndarray."""
    if not path.is_file():
        return None
    coords: List[List[float]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            try:
                coords.append([
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                ])
            except ValueError:
                continue
    if not coords:
        return None
    return np.asarray(coords, dtype=np.float64)


class CandidateReader:
    """Read all candidates for a task. Returns list[OracleCandidate]."""

    def read_task_candidates(
        self,
        task_dir: Path,
    ) -> List[OracleCandidate]:
        task_dir = Path(task_dir)
        task_id = task_dir.name

        idx_path = task_dir / "candidates" / "all_candidates_index.csv"
        npz_path = task_dir / "candidates" / "all_candidates_coords.npz"

        if idx_path.is_file() and npz_path.is_file():
            return self._read_archive(idx_path, npz_path, task_id)

        # fallback: read just postprocess/top1_ca.pdb
        warnings.warn(
            f"[oracle_eval] no candidates archive for task {task_id}; "
            f"falling back to postprocess/top1_ca.pdb only — oracle "
            f"quality will be limited to that single point.",
            stacklevel=2,
        )
        return self._fallback_top1(task_dir, task_id)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _read_archive(
        idx_path: Path, npz_path: Path, task_id: str,
    ) -> List[OracleCandidate]:
        with np.load(npz_path) as zf:
            coord_keys = set(zf.files)
            cache = {k: np.asarray(zf[k], dtype=np.float64) for k in zf.files}

        with idx_path.open("r", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))

        out: List[OracleCandidate] = []
        for r in rows:
            key = r.get("coords_key") or r.get("candidate_uid")
            if key is None or key not in cache:
                continue
            out.append(OracleCandidate(
                task_id=task_id,
                source=r.get("source") or "accepted",
                bitstring=r.get("bitstring") or None,
                coords=cache[key],
                full_energy=_f(r.get("full_energy")),
                refined_score=_f(r.get("refined_score")),
                refined_weight=_f(r.get("refined_weight")),
                is_dense=_b(r.get("is_dense")),
                parent_bitstring=r.get("parent_bitstring") or None,
                metadata={"candidate_uid": r.get("candidate_uid")},
            ))
        return out

    @staticmethod
    def _fallback_top1(
        task_dir: Path, task_id: str,
    ) -> List[OracleCandidate]:
        coords = _load_ca_only_pdb(
            task_dir / "postprocess" / "top1_ca.pdb",
        )
        if coords is None:
            return []
        return [OracleCandidate(
            task_id=task_id,
            source="postprocess",
            bitstring=None,
            coords=coords,
            metadata={"fallback_source": "postprocess/top1_ca.pdb"},
        )]


__all__ = ["CandidateReader"]
