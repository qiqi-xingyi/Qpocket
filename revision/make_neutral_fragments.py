#!/usr/bin/env python3
# Author: Yuqi Zhang
"""Neutral background fragments, chosen by a rule fixed in advance.

Reviewer 2 observes that every fragment in the study was chosen because it
is functionally interesting, so a large response in Switch-II has nothing
to be compared against:

    "No null/background distribution of response amplitude across
    additional (ideally non-functionally-selected) fragments is provided,
    so it is unclear whether Frag-B/Frag-F's larger response reflects
    genuine ligand-associated biology or simply that these fragments have
    larger inherent conformational search spaces."

He names the alternative explanation himself: inherent search-space size.
Fragment length is what governs that -- the encoded space is 64 raised to
the number of bonds -- so the background fragments are matched to the
lengths already in the study rather than chosen freely. A background set
of different lengths would confirm his objection instead of answering it.

A background fragment must also be the *same* fragment in every
structure. The response metric compares one fragment's projected occupancy
across the cases that share it -- the study's fragments each span seven to
nine structures -- so a window chosen independently per structure would
yield no response to compare against. Only residue ranges valid in every
structure are eligible here, and each selected range contributes one case
per structure exactly as the existing fragments do.

Selection is deterministic and stated before anything runs. Windows
overlapping the P-loop, Switch-I, Switch-II, or any existing fragment are
excluded; the remainder are enumerated in residue order and sampled at
even spacing. No seed, no tuning, and no opportunity to prefer windows
whose results turn out convenient.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# KRAS functional elements, by the conventional numbering used throughout
# the manuscript. A window touching any of these is not neutral.
FUNCTIONAL: Dict[str, Tuple[int, int]] = {
    "P-loop": (10, 17),
    "Switch-I": (30, 40),
    "Switch-II": (60, 76),
}
# Guard band on either side of an excluded element, so a "neutral" window
# does not sit immediately against a switch and inherit its motion.
MARGIN = 2


def observed_residues(pdb_path: Path, chain: str) -> List[int]:
    seen = set()
    for line in pdb_path.read_text(errors="ignore").splitlines():
        if line[:6] != "ATOM  " or line[21] != chain:
            continue
        if line[12:16].strip() != "CA":
            continue
        if line[16] not in (" ", "A"):
            continue
        try:
            seen.add(int(line[22:26]))
        except ValueError:
            pass
    return sorted(seen)


def excluded_spans(existing: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    spans = [(lo - MARGIN, hi + MARGIN) for lo, hi in FUNCTIONAL.values()]
    spans += [(lo - MARGIN, hi + MARGIN) for lo, hi in existing]
    return spans


def candidate_windows(residues: Sequence[int], length: int,
                      spans: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Contiguous windows of the given length that clear every excluded span."""
    present = set(residues)
    out: List[Tuple[int, int]] = []
    for start in residues:
        end = start + length - 1
        # Contiguity matters: a window spanning a gap in the model is not a
        # fragment the pipeline can reconstruct.
        if any(r not in present for r in range(start, end + 1)):
            continue
        if any(not (end < lo or start > hi) for lo, hi in spans):
            continue
        out.append((start, end))
    return out


def greedy_disjoint(windows: Sequence[Tuple[int, int]], k: int,
                    ) -> List[Tuple[int, int]]:
    """Windows that share no residue, taken left to right.

    Overlapping windows are not independent observations: two 15-residue
    windows five residues apart share two thirds of their content, and a
    background built from them would give a rank statistic far more
    confidence than the data support. Disjointness costs sample size and
    buys the right to treat the background as a sample.
    """
    chosen: List[Tuple[int, int]] = []
    last_end = -10 ** 9
    for lo, hi in sorted(windows):
        if lo > last_end:
            chosen.append((lo, hi))
            last_end = hi
    return evenly_spaced(chosen, k) if k < len(chosen) else chosen


def evenly_spaced(items: Sequence, k: int) -> List:
    if k <= 0 or not items:
        return []
    if len(items) <= k:
        return list(items)
    step = (len(items) - 1) / (k - 1) if k > 1 else 0
    return [items[int(round(i * step))] for i in range(k)]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks", default="inputs/kras_tasks.csv")
    p.add_argument("--structure-root", default="kras_select_systems")
    p.add_argument("--out", default="inputs/kras_neutral_tasks.csv")
    p.add_argument("--per-length", type=int, default=3,
                   help="neutral windows per fragment length")
    p.add_argument("--allow-overlap", action="store_true",
                   help="permit windows that share residues; the background "
                        "then has fewer independent observations than "
                        "members and its rank statistics are optimistic")
    args = p.parse_args(argv)

    rows = list(csv.DictReader((PROJECT_ROOT / args.tasks).open(
        encoding="utf-8")))
    # Lengths already in the study; the background must match them.
    lengths = sorted({int(r["end_resi"]) - int(r["start_resi"]) + 1
                      for r in rows})
    by_pdb: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        by_pdb.setdefault(r["ref_pdb"], []).append(r)

    out_rows: List[Dict[str, object]] = []
    report: List[Dict[str, object]] = []

    # Windows must clear the exclusions in every structure and be present
    # in every structure, so that each one is one fragment with a case per
    # structure -- the shape the response metric requires.
    per_pdb_valid: Dict[int, List[set]] = {L: [] for L in lengths}
    for pdb in sorted(by_pdb):
        sample = by_pdb[pdb][0]
        residues = observed_residues(
            PROJECT_ROOT / args.structure_root / pdb, sample["chain_id"])
        spans = excluded_spans([(int(r["start_resi"]), int(r["end_resi"]))
                                for r in by_pdb[pdb]])
        for L in lengths:
            per_pdb_valid[L].append(set(candidate_windows(residues, L, spans)))

    for L in lengths:
        common = sorted(set.intersection(*per_pdb_valid[L])
                        if per_pdb_valid[L] else set())
        if args.allow_overlap:
            picked = evenly_spaced(common, args.per_length)
            n_disjoint = None
        else:
            picked = greedy_disjoint(common, args.per_length)
            n_disjoint = len(greedy_disjoint(common, 10 ** 6))
        report.append({"length": L, "n_common_windows": len(common),
                       "n_disjoint_available": n_disjoint,
                       "n_selected": len(picked), "selected": picked,
                       "overlap_allowed": bool(args.allow_overlap),
                       "n_structures": len(by_pdb)})
        for idx, (lo, hi) in enumerate(picked):
            frag = f"Neu-L{L}-{idx + 1}"
            for pdb in sorted(by_pdb):
                sample = by_pdb[pdb][0]
                out_rows.append({
                    "case_id": f"{pdb[:4]}_{sample['mutation_group']}_{frag}",
                    "ref_pdb": pdb,
                    "chain_id": sample["chain_id"],
                    "start_resi": lo,
                    "end_resi": hi,
                    "ligand_resname": sample["ligand_resname"],
                    "scale_factor": sample["scale_factor"],
                    "scale_mode": sample["scale_mode"],
                    "has_native_ref": "true",
                    "mutation_group": sample["mutation_group"],
                    "ligand_family": sample["ligand_family"],
                    "pocket_module": "neutral_background",
                    "analysis_role": "background_control",
                })

    out_path = PROJECT_ROOT / args.out
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()),
                           lineterminator="\n")
        w.writeheader()
        w.writerows(out_rows)

    rule_path = out_path.with_name(out_path.stem + "_rule.json")
    rule_path.write_text(json.dumps({
        "functional_elements_excluded": FUNCTIONAL,
        "margin_residues": MARGIN,
        "existing_fragments_excluded": True,
        "lengths_matched_to_study": lengths,
        "per_length_per_structure": args.per_length,
        "selection": "windows qualifying in every structure are "
                     "intersected, enumerated in residue order, then "
                     "sampled at even spacing; deterministic, no seed",
        "shared_across_all_structures": True,
        "per_length": report,
    }, indent=2), encoding="utf-8")

    print(f"wrote {len(out_rows)} neutral fragments -> {out_path}")
    print(f"selection rule -> {rule_path}")
    print(f"\n{'len':>4} {'common':>8} {'disjoint':>9} {'picked':>7} "
          f"{'cases':>6}  best attainable rank-1 p")
    for r in report:
        nd = r["n_disjoint_available"]
        pmin = 1.0 / (r["n_selected"] + 1) if r["n_selected"] else float("nan")
        print(f"{r['length']:>4} {r['n_common_windows']:>8} "
              f"{nd if nd is not None else '-':>9} {r['n_selected']:>7} "
              f"{r['n_selected'] * r['n_structures']:>6}  {pmin:.3f}")
    print()
    for r in report:
        rng = ", ".join(f"{lo}-{hi}" for lo, hi in r["selected"])
        print(f"  L={r['length']}: {rng}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
