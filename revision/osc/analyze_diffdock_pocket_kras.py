#!/usr/bin/env python3
"""Measure one DiffDock-Pocket KRAS re-docking result in the receptor frame."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolAlign


BACKBONE = {"N", "CA", "C", "O", "OXT"}


def read_pdb_atoms(path: Path):
    atoms = {}
    for line in path.read_text().splitlines():
        if not line.startswith("ATOM  "):
            continue
        altloc = line[16]
        if altloc not in {" ", "A"}:
            continue
        atom_name = line[12:16].strip()
        resname = line[17:20].strip()
        chain = line[21].strip()
        resseq = int(line[22:26])
        icode = line[26].strip()
        element = line[76:78].strip() or atom_name[:1]
        key = (chain, resseq, icode, resname, atom_name)
        atoms[key] = {
            "coord": np.array(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                dtype=float,
            ),
            "element": element.upper(),
        }
    return atoms


def displacement_summary(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"n": 0, "mean_A": None, "rms_A": None, "max_A": None}
    return {
        "n": int(arr.size),
        "mean_A": float(arr.mean()),
        "rms_A": float(np.sqrt(np.mean(arr * arr))),
        "max_A": float(arr.max()),
    }


def protein_metrics(native_path: Path, predicted_path: Path):
    native = read_pdb_atoms(native_path)
    predicted = read_pdb_atoms(predicted_path)
    shared = sorted(set(native) & set(predicted))

    ca = []
    backbone = []
    sidechain = []
    by_residue = defaultdict(list)
    for key in shared:
        element = native[key]["element"]
        if element == "H":
            continue
        atom_name = key[-1]
        delta = float(np.linalg.norm(native[key]["coord"] - predicted[key]["coord"]))
        if atom_name == "CA":
            ca.append(delta)
        if atom_name in BACKBONE:
            backbone.append(delta)
        else:
            sidechain.append(delta)
            by_residue[key[:4]].append(delta)

    residue_rows = []
    for (chain, resseq, icode, resname), values in sorted(by_residue.items()):
        summary = displacement_summary(values)
        if summary["max_A"] is not None and summary["max_A"] > 1e-3:
            residue_rows.append(
                {
                    "residue": f"{chain}:{resname}{resseq}{icode}",
                    **summary,
                }
            )
    residue_rows.sort(key=lambda row: row["max_A"], reverse=True)

    return {
        "native_atom_records": len(native),
        "predicted_atom_records": len(predicted),
        "shared_atom_records": len(shared),
        "ca_displacement": displacement_summary(ca),
        "backbone_heavy_displacement": displacement_summary(backbone),
        "sidechain_heavy_displacement": displacement_summary(sidechain),
        "moved_sidechain_residues_gt_0.001A": residue_rows,
    }


def ligand_metrics(native_path: Path, predicted_path: Path):
    native = Chem.MolFromMolFile(str(native_path), removeHs=False, sanitize=True)
    predicted = Chem.MolFromMolFile(str(predicted_path), removeHs=False, sanitize=True)
    if native is None or predicted is None:
        raise RuntimeError("RDKit could not read one of the ligand SDF files")

    native_heavy = Chem.RemoveHs(native)
    predicted_heavy = Chem.RemoveHs(predicted)
    if native_heavy.GetNumAtoms() != predicted_heavy.GetNumAtoms():
        raise RuntimeError("native and predicted ligands have different heavy-atom counts")

    native_conf = native_heavy.GetConformer()
    predicted_conf = predicted_heavy.GetConformer()
    native_xyz = np.asarray(native_conf.GetPositions(), dtype=float)
    predicted_xyz = np.asarray(predicted_conf.GetPositions(), dtype=float)
    symbols_match = all(
        native_heavy.GetAtomWithIdx(i).GetSymbol()
        == predicted_heavy.GetAtomWithIdx(i).GetSymbol()
        for i in range(native_heavy.GetNumAtoms())
    )
    if not symbols_match:
        raise RuntimeError("native and predicted heavy-atom index orders differ")

    direct_delta = native_xyz - predicted_xyz
    direct_rmsd = math.sqrt(float(np.mean(np.sum(direct_delta * direct_delta, axis=1))))
    centroid_shift = float(np.linalg.norm(native_xyz.mean(axis=0) - predicted_xyz.mean(axis=0)))
    symmetry_rmsd = float(
        rdMolAlign.CalcRMS(native_heavy, predicted_heavy, maxMatches=100000)
    )
    native_copy = Chem.Mol(native_heavy)
    predicted_copy = Chem.Mol(predicted_heavy)
    aligned_internal_rmsd = float(
        rdMolAlign.GetBestRMS(native_copy, predicted_copy, maxMatches=100000)
    )

    return {
        "heavy_atoms": native_heavy.GetNumAtoms(),
        "direct_indexed_rmsd_A": direct_rmsd,
        "symmetry_aware_rmsd_without_alignment_A": symmetry_rmsd,
        "centroid_shift_A": centroid_shift,
        "best_aligned_internal_rmsd_A": aligned_internal_rmsd,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-protein", type=Path, required=True)
    parser.add_argument("--predicted-protein", type=Path, required=True)
    parser.add_argument("--native-ligand", type=Path, required=True)
    parser.add_argument("--predicted-ligand", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = {
        "protein": protein_metrics(args.native_protein, args.predicted_protein),
        "ligand": ligand_metrics(args.native_ligand, args.predicted_ligand),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
