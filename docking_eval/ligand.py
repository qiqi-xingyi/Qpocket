# Author: Yuqi Zhang
"""Ligand extraction by resname + native-ligand box computation.

When the reference PDB contains multiple residue copies of the same
``ligand_resname`` (e.g. 7RT1 has 7L8 A 203 and 7L8 A 204), naively
concatenating all atoms into a single PDB makes OpenBabel emit a PDBQT
with multiple ROOT blocks, which AutoDock Vina rejects with::

    PDBQT parsing error: Unknown or inappropriate tag found in flex
    residue or ligand.
     > ROOT

Selecting a single residue copy is therefore a hard requirement, not a
nicety. ``extract_ligand_from_pdb_with_metadata`` exposes the selection
explicitly; ``extract_ligand_from_pdb`` is kept as a thin compat shim.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# Selection modes recognised by extract_ligand_from_pdb_with_metadata.
LIGAND_SELECTION_NEAREST = "nearest_to_pocket_center"
LIGAND_SELECTION_FIRST = "first"
LIGAND_SELECTION_ALL_SEPARATE = "all_separate"

_VALID_SELECTION_MODES = (
    LIGAND_SELECTION_NEAREST,
    LIGAND_SELECTION_FIRST,
    LIGAND_SELECTION_ALL_SEPARATE,
)


@dataclass
class _LigandCopy:
    """A single ligand residue copy parsed out of a PDB."""
    chain_id: str
    resseq: int
    icode: str
    resname: str
    lines: List[str]
    coords: np.ndarray  # (n_atoms, 3) float64

    @property
    def n_atoms(self) -> int:
        return int(self.coords.shape[0])

    @property
    def centroid(self) -> np.ndarray:
        return self.coords.mean(axis=0)

    @property
    def bbox_min(self) -> np.ndarray:
        return self.coords.min(axis=0)

    @property
    def bbox_max(self) -> np.ndarray:
        return self.coords.max(axis=0)


@dataclass
class LigandExtractionResult:
    """Result of selecting one ligand residue copy out of possibly many."""
    ligand_pdb: Path
    ligand_resname: str
    selected_chain_id: str
    selected_resseq: int
    selected_icode: Optional[str]
    n_ligand_copies_found: int
    selection_mode: str
    selected_distance_to_pocket: Optional[float]
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------- #
# parsing                                                                #
# ---------------------------------------------------------------------- #

def _parse_ligand_copies(
    pdb_path: Path,
    target_resname: str,
    target_chain: Optional[str],
) -> List[_LigandCopy]:
    """Return one _LigandCopy per (chain, resseq, icode) group whose
    resname matches ``target_resname`` (and chain matches ``target_chain``
    if provided). HETATM lines only — ATOM records are not considered
    candidate ligand atoms."""
    target_resname = target_resname.strip().upper()
    target_chain_norm = target_chain.strip() if target_chain else None

    # Preserve insertion order so "first" mode is reproducible.
    order: List[Tuple[str, int, str]] = []
    groups: Dict[Tuple[str, int, str], _LigandCopy] = {}

    with Path(pdb_path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.startswith("HETATM"):
                continue
            if len(line) < 27:
                continue
            res = line[17:20].strip().upper()
            if res != target_resname:
                continue
            ch = line[21:22].strip()
            if target_chain_norm is not None and ch != target_chain_norm:
                continue
            try:
                resseq = int(line[22:26].strip())
            except ValueError:
                continue
            icode = line[26:27]
            if icode == " ":
                icode = ""
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except (ValueError, IndexError):
                continue
            key = (ch, resseq, icode)
            if key not in groups:
                groups[key] = _LigandCopy(
                    chain_id=ch,
                    resseq=resseq,
                    icode=icode,
                    resname=res,
                    lines=[],
                    coords=np.zeros((0, 3), dtype=np.float64),
                )
                order.append(key)
            cp = groups[key]
            cp.lines.append(line.rstrip("\n"))
            cp.coords = np.vstack([cp.coords, np.asarray([x, y, z])])

    return [groups[k] for k in order]


def _candidate_dict(
    cp: _LigandCopy,
    pocket_center: Optional[np.ndarray],
) -> Dict[str, Any]:
    d = {
        "chain_id": cp.chain_id,
        "resseq": cp.resseq,
        "icode": cp.icode if cp.icode else None,
        "resname": cp.resname,
        "n_atoms": cp.n_atoms,
        "centroid": cp.centroid.tolist(),
        "bbox_min": cp.bbox_min.tolist(),
        "bbox_max": cp.bbox_max.tolist(),
        "distance_to_pocket_center": (
            None if pocket_center is None
            else float(np.linalg.norm(cp.centroid - pocket_center))
        ),
    }
    return d


# ---------------------------------------------------------------------- #
# selection                                                              #
# ---------------------------------------------------------------------- #

def _select_copy(
    copies: List[_LigandCopy],
    selection_mode: str,
    pocket_center: Optional[np.ndarray],
) -> Tuple[_LigandCopy, str, List[str]]:
    """Returns (selected_copy, effective_mode, warnings)."""
    warnings: List[str] = []

    if len(copies) == 1:
        return copies[0], selection_mode, warnings

    if selection_mode == LIGAND_SELECTION_ALL_SEPARATE:
        raise NotImplementedError(
            "all_separate ligand selection is not implemented yet"
        )

    if selection_mode == LIGAND_SELECTION_FIRST:
        warnings.append("multiple_ligands_found; selected_first")
        return copies[0], LIGAND_SELECTION_FIRST, warnings

    if selection_mode == LIGAND_SELECTION_NEAREST:
        if pocket_center is None:
            warnings.append(
                "multiple_ligands_found_but_no_pocket_center; fallback_to_first"
            )
            return copies[0], LIGAND_SELECTION_FIRST, warnings
        pc = np.asarray(pocket_center, dtype=np.float64).reshape(3)
        dists = [float(np.linalg.norm(c.centroid - pc)) for c in copies]
        idx = int(np.argmin(dists))
        return copies[idx], LIGAND_SELECTION_NEAREST, warnings

    raise ValueError(
        f"unknown ligand_selection_mode={selection_mode!r}; "
        f"expected one of {_VALID_SELECTION_MODES}"
    )


# ---------------------------------------------------------------------- #
# public API                                                             #
# ---------------------------------------------------------------------- #

def extract_ligand_from_pdb_with_metadata(
    pdb_path: Path,
    ligand_resname: str,
    output_pdb: Path,
    chain_id: Optional[str] = None,
    pocket_center: Optional[np.ndarray] = None,
    ligand_selection_mode: str = LIGAND_SELECTION_NEAREST,
) -> LigandExtractionResult:
    """Extract a single ligand residue copy and write it to ``output_pdb``.

    See module docstring for why selecting *one* copy is required.

    Selection modes:
        - "nearest_to_pocket_center": centroid distance to ``pocket_center``;
          falls back to "first" with a warning if pocket_center is None.
        - "first": PDB-order first residue copy.
        - "all_separate": raises NotImplementedError.
    """
    pdb_path = Path(pdb_path)
    output_pdb = Path(output_pdb)
    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    if not pdb_path.is_file():
        raise FileNotFoundError(f"PDB not found: {pdb_path}")
    if ligand_selection_mode not in _VALID_SELECTION_MODES:
        raise ValueError(
            f"unknown ligand_selection_mode={ligand_selection_mode!r}; "
            f"expected one of {_VALID_SELECTION_MODES}"
        )

    copies = _parse_ligand_copies(pdb_path, ligand_resname, chain_id)
    if not copies:
        raise ValueError(
            f"no HETATM records with resname={ligand_resname.strip().upper()!r}"
            + (f" chain={chain_id}" if chain_id else "")
            + f" in {pdb_path}"
        )

    pc_arr: Optional[np.ndarray] = None
    if pocket_center is not None:
        pc_arr = np.asarray(pocket_center, dtype=np.float64).reshape(3)

    selected, effective_mode, warnings = _select_copy(
        copies, ligand_selection_mode, pc_arr,
    )

    # Write only the selected copy.
    out_lines = list(selected.lines)
    out_lines.append("END")
    output_pdb.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    candidates = [_candidate_dict(c, pc_arr) for c in copies]
    selected_distance: Optional[float] = None
    if pc_arr is not None:
        selected_distance = float(
            np.linalg.norm(selected.centroid - pc_arr)
        )

    metadata = {
        "pdb_path": str(pdb_path),
        "requested_chain_id": chain_id,
        "requested_selection_mode": ligand_selection_mode,
        "effective_selection_mode": effective_mode,
        "pocket_center": (
            None if pc_arr is None else pc_arr.tolist()
        ),
    }

    return LigandExtractionResult(
        ligand_pdb=output_pdb,
        ligand_resname=ligand_resname.strip().upper(),
        selected_chain_id=selected.chain_id,
        selected_resseq=selected.resseq,
        selected_icode=(selected.icode if selected.icode else None),
        n_ligand_copies_found=len(copies),
        selection_mode=effective_mode,
        selected_distance_to_pocket=selected_distance,
        candidates=candidates,
        metadata=metadata,
        warnings=warnings,
    )


def extract_ligand_from_pdb(
    pdb_path: Path,
    ligand_resname: str,
    output_pdb: Path,
    chain_id: Optional[str] = None,
    pocket_center: Optional[np.ndarray] = None,
    ligand_selection_mode: str = LIGAND_SELECTION_NEAREST,
) -> Path:
    """Compat wrapper: writes only the selected residue copy.

    Returns the output path (matching the original signature). For full
    selection metadata, use ``extract_ligand_from_pdb_with_metadata``.
    """
    res = extract_ligand_from_pdb_with_metadata(
        pdb_path=pdb_path,
        ligand_resname=ligand_resname,
        output_pdb=output_pdb,
        chain_id=chain_id,
        pocket_center=pocket_center,
        ligand_selection_mode=ligand_selection_mode,
    )
    return res.ligand_pdb


# ---------------------------------------------------------------------- #
# box computation                                                        #
# ---------------------------------------------------------------------- #

def compute_ligand_box(
    ligand_pdb: Path,
    padding: float = 8.0,
    min_box_size: float = 20.0,
) -> Dict[str, Any]:
    """Native-ligand box: centroid + (bbox + padding) per axis, with a
    minimum side length of ``min_box_size``.

    Returns
    -------
    dict with center_x/y/z, size_x/y/z, n_ligand_atoms.
    """
    coords: List[List[float]] = []
    with Path(ligand_pdb).open("r", encoding="utf-8") as fh:
        for line in fh:
            if not (line.startswith("HETATM") or line.startswith("ATOM")):
                continue
            if len(line) < 54:
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
        raise ValueError(f"no atom coordinates in {ligand_pdb}")
    arr = np.asarray(coords, dtype=np.float64)
    centroid = arr.mean(axis=0)
    bbox = arr.max(axis=0) - arr.min(axis=0) + 2.0 * float(padding)
    size = np.maximum(bbox, float(min_box_size))
    return {
        "center_x": float(centroid[0]),
        "center_y": float(centroid[1]),
        "center_z": float(centroid[2]),
        "size_x": float(size[0]),
        "size_y": float(size[1]),
        "size_z": float(size[2]),
        "n_ligand_atoms": int(arr.shape[0]),
    }


__all__ = [
    "LigandExtractionResult",
    "extract_ligand_from_pdb",
    "extract_ligand_from_pdb_with_metadata",
    "compute_ligand_box",
    "LIGAND_SELECTION_NEAREST",
    "LIGAND_SELECTION_FIRST",
    "LIGAND_SELECTION_ALL_SEPARATE",
]
