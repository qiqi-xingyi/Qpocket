#!/usr/bin/env python3
# Author: Yuqi Zhang
"""Aggregate the input-matched local comparators against the sampled arms.

Reviewer 2 names three comparators and they are not one kind of thing.
What separates them is where each one starts.

Kinematic closure is given the fragment as an extended chain and rebuilds
it, so it solves the problem this pipeline solves: produce a local
conformation from the anchors and the surrounding structure, without
having seen the answer.

Backrub and induced-fit docking start from the deposited conformation and
move away from it. Their agreement with the native measures how far they
travelled, not whether they could find it, and a small number from them
means the opposite of what the same number means for an arm that started
elsewhere. They are reported, because they were asked for and because
they say something real about local refinement, but they are reported
separately and are not read as a ranking against a de novo method.

The same asymmetry that ruled the sequence-to-structure comparison out of
a superiority claim applies here with its sign reversed, and it is
applied the same way.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# arm -> (label, results root, starts from the deposited conformation)
ARMS: Dict[str, Tuple[str, str, bool]] = {
    "Q":  ("hardware sampling",        "g1_pilot", False),
    "P":  ("prior, drawn exactly",     "g1_pilot", False),
    "M":  ("Metropolis on the prior",  "g1_pilot", False),
    "R":  ("Rosetta KIC",              "g2_local", False),
    "RB": ("Rosetta backrub",          "g2_local", True),
    "I":  ("RosettaLigand induced fit", "g2_local", True),
}


def fragment_lengths(root: Path) -> Dict[str, int]:
    """Residue count per task, read from the task table."""
    import csv
    out: Dict[str, int] = {}
    with (PROJECT_ROOT / "inputs/kras_tasks.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out[r["case_id"]] = int(r["end_resi"]) - int(r["start_resi"]) + 1
    return out


def load(root: Path) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for arm, (_lab, sub, _nat) in ARMS.items():
        base = root / sub / arm
        for f in sorted(base.glob("*/repeat_*/endpoint_record.json")):
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            ep = (d.get("results") or {}).get("endpoints") or {}
            prim = ep.get("primary") or {}
            sec = ep.get("secondary_oracle_best") or {}
            if prim.get("inframe_rmsd") is None:
                continue
            out[arm].append({
                "task": f.parent.parent.name,
                "length": None,
                "repeat": int(f.parent.name.split("_", 1)[1]),
                "inframe": float(prim["inframe_rmsd"]),
                "kabsch": float(prim.get("kabsch_rmsd") or float("nan")),
                "oracle": sec.get("inframe"),
                "n_pool": sec.get("n_pool"),
            })
    return out


def med(xs: List[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None and x == x]
    return statistics.median(xs) if xs else None


def wilcoxon(diffs: List[float]) -> Optional[float]:
    d = [x for x in diffs if x != 0.0]
    n = len(d)
    if n < 6:
        return None
    order = sorted(range(n), key=lambda i: abs(d[i]))
    rank = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            rank[order[k]] = avg
        i = j + 1
    w = min(sum(rank[i] for i in range(n) if d[i] > 0),
            sum(rank[i] for i in range(n) if d[i] < 0))
    mu = n * (n + 1) / 4.0
    sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if sd == 0:
        return None
    z = (w - mu + 0.5) / sd
    return min(1.0, 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2)))))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-root", default="revision/results")
    p.add_argument("--out", default="revision/results/G2_SUMMARY.md")
    args = p.parse_args(argv)

    root = PROJECT_ROOT / args.results_root
    data = load(root)
    lens = fragment_lengths(root)
    for rows in data.values():
        for r in rows:
            r["length"] = lens.get(r["task"])
    if not data:
        print("no endpoint records found", file=sys.stderr)
        return 2

    lines: List[str] = []
    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    emit("# Input-matched local comparators")
    emit()
    emit("Every arm below receives the same solved receptor, the same")
    emit("fragment definition, and the same anchors, and every one is scored")
    emit("by the endpoints frozen before any of them was read.")
    emit()
    emit("| arm | method | starts from | cells | selected in-frame | pool best | median pool |")
    emit("|---|---|---|---:|---:|---:|---:|")
    for arm, (label, _sub, from_native) in ARMS.items():
        rows = data.get(arm, [])
        if not rows:
            emit(f"| {arm} | {label} | — | 0 | — | — | — |")
            continue
        start = "deposited pose" if from_native else "no native conformation"
        sel = med([r["inframe"] for r in rows])
        orc = med([r["oracle"] for r in rows])
        pool = med([r["n_pool"] for r in rows if r["n_pool"]])
        emit(f"| {arm} | {label} | {start} | {len(rows)} | {sel:.3f} | "
             f"{orc:.3f} | {int(pool) if pool else 0:,} |")
    emit()
    emit("The start column is the whole reading of this table. An arm handed")
    emit("the deposited conformation and asked to move locally will sit close")
    emit("to it, and that closeness is not a reconstruction result.")
    emit()
    emit("## By fragment length")
    emit()
    emit("Pooling the lengths hides the result. A short segment pinned at")
    emit("both ends has little freedom left and any method that closes it")
    emit("lands near the answer; a longer one is a genuine search. The arms")
    emit("do not scale the same way, and the pooled median is dominated by")
    emit("whichever lengths are most numerous.")
    emit()
    all_len = sorted({r["length"] for rows in data.values() for r in rows
                      if r["length"]})
    header = "| length | tasks | " + " | ".join(ARMS) + " |"
    emit(header)
    emit("|---|---:|" + "---:|" * len(ARMS))
    for L in all_len:
        cells = []
        ntask = len({r["task"] for r in data.get("Q", [])
                     if r["length"] == L})
        for arm in ARMS:
            v = med([r["inframe"] for r in data.get(arm, [])
                     if r["length"] == L])
            cells.append(f"{v:.2f}" if v is not None else "—")
        emit(f"| {L} | {ntask} | " + " | ".join(cells) + " |")
    emit()
    emit("Selected in-frame RMSD, median over cells. The same table for the")
    emit("pool minimum:")
    emit()
    emit(header)
    emit("|---|---:|" + "---:|" * len(ARMS))
    for L in all_len:
        cells = []
        ntask = len({r["task"] for r in data.get("Q", [])
                     if r["length"] == L})
        for arm in ARMS:
            v = med([r["oracle"] for r in data.get(arm, [])
                     if r["length"] == L])
            cells.append(f"{v:.2f}" if v is not None else "—")
        emit(f"| {L} | {ntask} | " + " | ".join(cells) + " |")
    emit()

    # Task-level pairing. Repeats collapse to their median so that one task
    # contributes one observation, as in the pilot.
    tv: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    ov: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for arm, rows in data.items():
        for r in rows:
            tv[(arm, r["task"])].append(r["inframe"])
            if r["oracle"] is not None:
                ov[(arm, r["task"])].append(r["oracle"])

    emit("## Against kinematic closure")
    emit()
    emit("KIC is the one comparator that is not started at the answer, so it")
    emit("is the only one a superiority claim could rest on either way.")
    emit()
    for other in ("Q", "P", "M"):
        pairs = []
        for (arm, task), vals in tv.items():
            if arm != other:
                continue
            kic = tv.get(("R", task))
            if kic:
                pairs.append((task, med(vals), med(kic)))
        if not pairs:
            continue
        diffs = [a - b for _t, a, b in pairs]
        pv = wilcoxon(diffs)
        wins = sum(1 for d in diffs if d < 0)
        emit(f"**{other} vs KIC** — {len(pairs)} tasks, median difference "
             f"{med(diffs):+.3f} A, {other} closer on {wins}/{len(pairs)}"
             + (f", Wilcoxon p = {pv:.4f}" if pv is not None else ""))
        # Coverage, which is what a generator is for.
        opairs = []
        for (arm, task), vals in ov.items():
            if arm != other:
                continue
            kic = ov.get(("R", task))
            if kic:
                opairs.append(med(vals) - med(kic))
        if opairs:
            emit(f"  pool best: median difference {med(opairs):+.3f} A, "
                 f"{other} covers better on "
                 f"{sum(1 for d in opairs if d < 0)}/{len(opairs)}")
        emit()

    emit("## Refinement arms, reported separately")
    emit()
    emit("| arm | selected in-frame | pool best |")
    emit("|---|---:|---:|")
    for arm in ("RB", "I"):
        rows = data.get(arm, [])
        if rows:
            emit(f"| {ARMS[arm][1] if False else arm} | "
                 f"{med([r['inframe'] for r in rows]):.3f} | "
                 f"{med([r['oracle'] for r in rows]):.3f} |")
    emit()
    emit("These start from the deposited conformation. Their numbers bound")
    emit("how little local refinement needs to move, not how well a")
    emit("conformation can be rebuilt without it.")
    emit()

    out = PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
