# Author: Yuqi Zhang
"""I/O helpers for the reconstruct package.

read_ca_coords_from_pdb : (n, 3) CA coordinates only (no HETATM, no
                          alignment). Raises on missing CA / non-finite
                          values.
compute_ca_drift        : per-atom drift between two CA traces, no
                          Kabsch alignment. Returns a dict with all
                          fields the reconstruction summary expects.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

import numpy as np


def read_ca_coords_from_pdb(path: Union[str, Path]) -> np.ndarray:
    """Read CA atom coordinates from `path`.

    Returns shape ``(n_ca, 3)`` float64. Raises ``ValueError`` if no CA
    atoms are found or any coordinate is non-finite. HETATM records and
    non-CA atoms are ignored.
    """
    path = Path(path)
    coords = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            if len(line) < 54:
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
        raise ValueError(f"no CA atoms in {path}")
    arr = np.asarray(coords, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"non-finite CA coordinates in {path}")
    return arr


def compute_ca_drift(
    original_ca_pdb: Union[str, Path],
    rebuilt_fragment_pdb: Union[str, Path],
) -> Dict[str, Any]:
    """Per-atom CA drift between two PDBs (no Kabsch alignment).

    Returns a dict with the canonical reconstruction-summary keys:

      ca_drift_rmsd            float | None
      ca_drift_max             float | None
      ca_drift_n_ca            int   (only when shapes match)
      ca_drift_n_ca_original   int
      ca_drift_n_ca_rebuilt    int
      ca_drift_shape_mismatch  bool
      ca_drift_warning         "ca_count_mismatch" | None

    The threshold-driven warning ``"pulchra_ca_drift_exceeds_threshold"``
    is set later by the pipeline if ``ca_drift_max`` exceeds the user's
    threshold. This function never returns that warning — only the
    shape-mismatch warning.
    """
    a = read_ca_coords_from_pdb(original_ca_pdb)
    b = read_ca_coords_from_pdb(rebuilt_fragment_pdb)
    n_orig = int(a.shape[0])
    n_rebuilt = int(b.shape[0])
    if n_orig != n_rebuilt:
        return {
            "ca_drift_rmsd": None,
            "ca_drift_max": None,
            "ca_drift_n_ca_original": n_orig,
            "ca_drift_n_ca_rebuilt": n_rebuilt,
            "ca_drift_shape_mismatch": True,
            "ca_drift_warning": "ca_count_mismatch",
        }
    diffs = np.linalg.norm(a - b, axis=1)
    return {
        "ca_drift_rmsd": float(np.sqrt(float(np.mean(diffs * diffs)))),
        "ca_drift_max": float(np.max(diffs)),
        "ca_drift_n_ca": n_orig,
        "ca_drift_n_ca_original": n_orig,
        "ca_drift_n_ca_rebuilt": n_rebuilt,
        "ca_drift_shape_mismatch": False,
        "ca_drift_warning": None,
    }


__all__ = ["read_ca_coords_from_pdb", "compute_ca_drift"]
