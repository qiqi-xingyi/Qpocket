#!/usr/bin/env python3
"""Summarize a ranked DiffDock-Pocket ensemble against a native complex."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from analyze_diffdock_pocket_kras import ligand_metrics, protein_metrics


RANK_RE = re.compile(r"^rank(\d+)\.sdf$")
CONF_RE = re.compile(r"^rank(\d+)_confidence([-+0-9.eE]+)\.sdf$")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-protein", type=Path, required=True)
    parser.add_argument("--native-ligand", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    ranked_files = {}
    for path in args.result_dir.glob("rank*_confidence*.sdf"):
        match = CONF_RE.match(path.name)
        if match:
            rank = int(match.group(1))
            ranked_files[rank] = {
                "ligand": path,
                "protein": path.with_name(path.stem + "_protein.pdb"),
                "confidence": float(match.group(2)),
            }

    # When no confidence model is used the official writer only guarantees the
    # unsuffixed top-ranked files.  Keep that case as a valid fallback.
    if not ranked_files:
        ligand_path = args.result_dir / "rank1.sdf"
        ranked_files[1] = {
            "ligand": ligand_path,
            "protein": args.result_dir / "rank1_protein.pdb",
            "confidence": None,
        }

    predictions = []
    for rank, files in sorted(ranked_files.items()):
        ligand_path = files["ligand"]
        protein_path = files["protein"]
        ligand = ligand_metrics(args.native_ligand, ligand_path)
        protein = protein_metrics(args.native_protein, protein_path)
        predictions.append(
            {
                "rank": rank,
                "confidence": files["confidence"],
                "ligand": ligand,
                "protein": protein,
            }
        )

    predictions.sort(key=lambda row: row["rank"])
    if not predictions:
        raise RuntimeError(f"no ranked predictions found in {args.result_dir}")

    metric = "symmetry_aware_rmsd_without_alignment_A"
    best = min(predictions, key=lambda row: row["ligand"][metric])
    result = {
        "n_predictions": len(predictions),
        "rank1": predictions[0],
        "best_ligand_pose_by_native_rmsd": best,
        "success_counts": {
            "ligand_rmsd_lt_2A": sum(row["ligand"][metric] < 2.0 for row in predictions),
            "ligand_rmsd_lt_5A": sum(row["ligand"][metric] < 5.0 for row in predictions),
        },
        "predictions": predictions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
