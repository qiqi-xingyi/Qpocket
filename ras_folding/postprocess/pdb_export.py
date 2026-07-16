# Author: Yuqi Zhang
"""CA-only PDB export.

Produces a minimal CA-only PDB that downstream viewers (PyMOL, ChimeraX,
Bio.PDB) can read. We do NOT reconstruct sidechains, hydrogens, or
backbone N/C/O atoms — only the CA trace produced by the encoder.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import numpy as np


# 20-AA one-letter → three-letter map, plus a few common alternates.
_ONE_TO_THREE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}


def _resname_for(
    residue: Union[str, None],
) -> str:
    """Map a 1-letter or 3-letter residue identifier to a 3-letter resname.

    Unknown / None → "UNK"."""
    if residue is None:
        return "UNK"
    s = str(residue).strip().upper()
    if len(s) == 1:
        return _ONE_TO_THREE.get(s, "UNK")
    if len(s) == 3:
        return s
    return "UNK"


def _resolve_resnames(
    sequence: Optional[Union[str, Sequence[str]]],
    n_residues: int,
) -> List[str]:
    if sequence is None:
        return ["UNK"] * n_residues
    if isinstance(sequence, str):
        if len(sequence) != n_residues:
            return ["UNK"] * n_residues
        return [_resname_for(c) for c in sequence]
    seq = list(sequence)
    if len(seq) != n_residues:
        return ["UNK"] * n_residues
    return [_resname_for(s) for s in seq]


def _format_atom_record(
    *,
    serial: int,
    atom_name: str,
    res_name: str,
    chain_id: str,
    res_seq: int,
    x: float,
    y: float,
    z: float,
    occupancy: float = 1.0,
    temp_factor: float = 0.0,
    element: str = "C",
) -> str:
    """Format a single PDB ATOM record per the v3.x column spec.

    Columns are 1-based per the PDB spec; we lay them out with
    explicit padding to avoid format-string column drift.
    """
    name = atom_name.strip()
    if len(name) < 4:
        # PDB convention: atoms with element 1-char and name 1-3 chars
        # left-pad with one space (so "CA" sits in cols 14-15).
        name_field = f" {name:<3s}"
    else:
        name_field = name[:4]
    record = (
        f"ATOM  "                         # cols 1-6
        f"{serial:5d} "                   # cols 7-11 + space col 12
        f"{name_field:<4s}"               # cols 13-16
        f" "                              # col 17 altLoc
        f"{res_name[:3]:>3s}"             # cols 18-20
        f" "                              # col 21
        f"{chain_id[:1]:1s}"              # col 22
        f"{res_seq:4d}"                   # cols 23-26
        f" "                              # col 27 iCode
        f"   "                            # cols 28-30
        f"{x:8.3f}"                       # cols 31-38
        f"{y:8.3f}"                       # cols 39-46
        f"{z:8.3f}"                       # cols 47-54
        f"{occupancy:6.2f}"               # cols 55-60
        f"{temp_factor:6.2f}"             # cols 61-66
        f"          "                     # cols 67-76
        f"{element[:2]:>2s}"              # cols 77-78
    )
    return record


def write_ca_pdb(
    coords: np.ndarray,
    sequence: Optional[Union[str, Sequence[str]]],
    path: Union[str, Path],
    chain_id: str = "A",
    atom_name: str = "CA",
) -> None:
    """Write a CA-only PDB file.

    Parameters
    ----------
    coords : (n_residues, 3) float
    sequence : 1-letter str OR list of 3-letter codes OR None.
    path : file path
    chain_id : single character
    atom_name : "CA" by default
    """
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(
            f"coords must be (n, 3); got {coords.shape}"
        )
    n = int(coords.shape[0])
    if not np.all(np.isfinite(coords)):
        raise ValueError("non-finite coordinates")
    res_names = _resolve_resnames(sequence, n)

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for i in range(n):
            line = _format_atom_record(
                serial=i + 1,
                atom_name=atom_name,
                res_name=res_names[i],
                chain_id=chain_id,
                res_seq=i + 1,
                x=float(coords[i, 0]),
                y=float(coords[i, 1]),
                z=float(coords[i, 2]),
                element="C",
            )
            fh.write(line + "\n")
        fh.write("TER\n")
        fh.write("END\n")


__all__ = ["write_ca_pdb"]
