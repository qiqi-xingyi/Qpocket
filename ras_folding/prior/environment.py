# Author: Yuqi Zhang
"""EnvironmentPriorBuilder — receptor environment + ligand from a PDB.

Reads a reference PDB, extracts heavy atoms, and produces a KDTree over
receptor heavy atoms (with the predicted fragment removed) plus the
selected ligand's heavy atoms separately.

Backward compatibility: this module does NOT touch any V1 code path.
It is consumed only by run_v2.py / V2 prior sampler.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.spatial import cKDTree
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "ras_folding.prior requires scipy.spatial.cKDTree"
    ) from exc


_WATERS = ("HOH", "WAT", "DOD", "TIP")


def _parse_pdb_heavy(
    pdb_path: Path,
    chain_id: str,
    fragment_resi_range: Tuple[int, int],
    ligand_resname: Optional[str],
) -> Dict[str, object]:
    """Return:
      - env_heavy: list[np.ndarray(3,)] — protein heavy atoms (ATOM) on
        any chain, with predicted-fragment residues removed; hydrogens
        excluded; waters excluded.
      - ligand_copies: list of dicts (resseq, chain, heavy_xyz, all_xyz,
        centroid, n_heavy) for HETATM blocks matching ligand_resname.
      - other_hetero_heavy: heavy atoms of other HETATMs (cofactors etc.)
        on the predicted chain — included in env (they are real atoms).
    """
    if not pdb_path.is_file():
        raise FileNotFoundError(f"PDB not found: {pdb_path}")
    start, end = fragment_resi_range
    env_heavy: List[np.ndarray] = []
    ligand_blocks: Dict[str, List[Tuple[str, np.ndarray]]] = {}
    in_model = True
    with pdb_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            rec = line[0:6].strip()
            if rec == "MODEL":
                if env_heavy or ligand_blocks:
                    break
                in_model = True
                continue
            if rec == "ENDMDL":
                break
            if rec not in ("ATOM", "HETATM"):
                continue
            if not in_model:
                continue
            altloc = line[16:17]
            if altloc not in (" ", "A"):
                continue
            atom_name = line[12:16].strip()
            if atom_name.startswith("H") or atom_name.startswith("D"):
                continue
            res_three = line[17:20].strip()
            if res_three in _WATERS:
                continue
            try:
                resseq = int(line[22:26])
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            chain = line[21:22]
            xyz = np.asarray([x, y, z], dtype=np.float64)
            if rec == "ATOM":
                if chain == chain_id and start <= resseq <= end:
                    continue  # exclude predicted fragment
                env_heavy.append(xyz)
            else:  # HETATM
                if (ligand_resname is not None
                        and res_three == ligand_resname):
                    key = f"{chain}/{resseq}"
                    ligand_blocks.setdefault(key, []).append(
                        (atom_name, xyz)
                    )
                else:
                    env_heavy.append(xyz)
    ligand_copies: List[Dict[str, object]] = []
    for key, atoms in ligand_blocks.items():
        chain_str, resseq_str = key.split("/")
        heavy = np.stack([a[1] for a in atoms], axis=0)
        ligand_copies.append({
            "key": key,
            "chain": chain_str,
            "resseq": int(resseq_str),
            "heavy": heavy,
            "centroid": heavy.mean(axis=0),
            "n_heavy": int(heavy.shape[0]),
        })
    env_arr = (np.stack(env_heavy, axis=0)
               if env_heavy else np.zeros((0, 3), dtype=np.float64))
    return {"env_heavy": env_arr, "ligand_copies": ligand_copies}


@dataclass
class EnvironmentPriorContext:
    env_atom_coords: np.ndarray
    env_tree: object  # cKDTree
    ligand_atom_coords: Optional[np.ndarray]
    ligand_centroid: Optional[np.ndarray]
    selected_ligand_resseq: Optional[int]
    selected_ligand_chain: Optional[str]
    ligand_selection_mode: str
    n_env_atoms: int
    n_ligand_atoms: int
    n_ligand_copies_found: int
    fragment_chain_id: str
    fragment_start_resi: int
    fragment_end_resi: int
    pdb_path: str
    metadata: Dict[str, object] = field(default_factory=dict)


def build_environment_prior(
    pdb_path: str,
    chain_id: str,
    start_resi: int,
    end_resi: int,
    ligand_resname: Optional[str],
    *,
    fragment_ca_centroid: Optional[np.ndarray] = None,
    selected_ligand_resseq: Optional[int] = None,
    selected_ligand_chain: Optional[str] = None,
) -> EnvironmentPriorContext:
    """Build an EnvironmentPriorContext from a reference PDB.

    Parameters
    ----------
    pdb_path : str
    chain_id : str — predicted chain
    start_resi, end_resi : int — predicted fragment range (inclusive)
    ligand_resname : str | None — ligand 3-letter code (None to skip)
    fragment_ca_centroid : (3,) — used to pick nearest ligand copy if
        multiple are present and no explicit selection is given.
    selected_ligand_resseq, selected_ligand_chain : optional explicit
        selection (overrides nearest-to-pocket).
    """
    parsed = _parse_pdb_heavy(
        Path(pdb_path), chain_id, (start_resi, end_resi),
        ligand_resname,
    )
    env_arr = parsed["env_heavy"]
    if env_arr.shape[0] == 0:
        raise ValueError(
            f"environment is empty for {pdb_path}: no heavy atoms outside "
            f"fragment {chain_id}:{start_resi}-{end_resi}"
        )
    env_tree = cKDTree(env_arr)
    lig_copies: List[Dict[str, object]] = parsed["ligand_copies"]
    n_lig_copies = len(lig_copies)
    selected_lig: Optional[Dict[str, object]] = None
    selection_mode = "none"
    if n_lig_copies == 0:
        pass
    elif (selected_ligand_resseq is not None
          and selected_ligand_chain is not None):
        for c in lig_copies:
            if (c["chain"] == selected_ligand_chain
                    and c["resseq"] == int(selected_ligand_resseq)):
                selected_lig = c
                selection_mode = "explicit"
                break
        if selected_lig is None:
            # explicit selection not found → fall back
            selection_mode = "explicit_not_found"
            if fragment_ca_centroid is not None:
                dists = [float(np.linalg.norm(c["centroid"] - fragment_ca_centroid))
                         for c in lig_copies]
                selected_lig = lig_copies[int(np.argmin(dists))]
                selection_mode = "nearest_to_pocket_center_fallback"
            else:
                selected_lig = lig_copies[0]
                selection_mode = "first_fallback"
    elif n_lig_copies == 1:
        selected_lig = lig_copies[0]
        selection_mode = "first"
    else:
        if fragment_ca_centroid is not None:
            dists = [float(np.linalg.norm(c["centroid"] - fragment_ca_centroid))
                     for c in lig_copies]
            selected_lig = lig_copies[int(np.argmin(dists))]
            selection_mode = "nearest_to_pocket_center"
        else:
            selected_lig = lig_copies[0]
            selection_mode = "first"
    if selected_lig is None:
        lig_xyz = None
        lig_centroid = None
        n_lig_atoms = 0
        sel_resseq = None
        sel_chain = None
    else:
        lig_xyz = selected_lig["heavy"]
        lig_centroid = selected_lig["centroid"]
        n_lig_atoms = int(selected_lig["n_heavy"])
        sel_resseq = int(selected_lig["resseq"])
        sel_chain = str(selected_lig["chain"])
    return EnvironmentPriorContext(
        env_atom_coords=env_arr,
        env_tree=env_tree,
        ligand_atom_coords=lig_xyz,
        ligand_centroid=lig_centroid,
        selected_ligand_resseq=sel_resseq,
        selected_ligand_chain=sel_chain,
        ligand_selection_mode=selection_mode,
        n_env_atoms=int(env_arr.shape[0]),
        n_ligand_atoms=n_lig_atoms,
        n_ligand_copies_found=n_lig_copies,
        fragment_chain_id=chain_id,
        fragment_start_resi=int(start_resi),
        fragment_end_resi=int(end_resi),
        pdb_path=str(pdb_path),
        metadata={
            "ligand_resname": ligand_resname,
        },
    )


__all__ = ["EnvironmentPriorContext", "build_environment_prior"]
