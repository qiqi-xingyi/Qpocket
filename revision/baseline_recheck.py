#!/usr/bin/env python3
# Author: Yuqi Zhang
"""Re-score the baseline comparison under one output convention.

The recorded comparison places `min_kabsch_rmsd` -- the lowest RMSD among
every structure the pipeline generated, identified by comparison with the
native fragment -- in the same column as one prediction from each
sequence-to-structure model. The two are different quantities. This
recomputes the comparison with the pipeline scored the way the baselines
are: a single structure, chosen without the native.

The single structure is the representative the frozen selector ranks
first, taken from the G1 pilot's Q arm, which carries the same hardware
samples the recorded comparison was built from.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASELINES = ("AF3", "ESMFold", "OmegaFold", "OpenFold")


def _f(x: Any) -> Optional[float]:
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--merged", default="result/RMSD/merged_rmsd_summary.csv")
    p.add_argument("--results-root", default="revision/results/g1_pilot")
    p.add_argument("--out",
                   default="revision/results/g1_pilot/BASELINE_RECHECK.md")
    args = p.parse_args(argv)

    root = PROJECT_ROOT
    selected: Dict[str, float] = {}
    for f in glob.glob(str(root / args.results_root
                           / "Q/*/repeat_0/endpoint_record.json")):
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        prim = ((d.get("results") or {}).get("endpoints") or {}).get("primary")
        if prim and prim.get("kabsch_rmsd") is not None:
            selected[d["task_id"]] = float(prim["kabsch_rmsd"])
    if not selected:
        print("no scored Q cells found", file=sys.stderr)
        return 2

    merged = root / args.merged
    if not merged.is_file():
        print(f"missing {merged}", file=sys.stderr)
        return 2

    rows: List[Dict[str, Any]] = []
    for r in csv.DictReader(merged.open(encoding="utf-8")):
        cid = r["case_id"]
        oracle = _f(r.get("min_kabsch_rmsd"))
        sel = selected.get(cid)
        if oracle is None or sel is None:
            continue
        rows.append({
            "case_id": cid,
            "pool": _f(r.get("n_generated_total")),
            "oracle": oracle,
            "selected": sel,
            **{b: _f(r.get(b)) for b in BASELINES},
            **{b + "_n": _f(r.get(b + "_n_predictions")) for b in BASELINES},
        })
    if not rows:
        print("no case matched between the two sources", file=sys.stderr)
        return 2

    lines: List[str] = []
    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    pools = [r["pool"] for r in rows if r["pool"]]
    emit("# Baseline comparison under one output convention")
    emit()
    emit("The recorded comparison scores this pipeline by the lowest RMSD")
    emit("among every structure it generated, found by comparing candidates")
    emit("with the native fragment, and scores each sequence-to-structure")
    emit("model by its prediction. Recovering the first requires the answer;")
    emit("the second does not. Placing them in one column compares a search")
    emit("over a pool against a prediction.")
    emit()
    emit(f"Cases: {len(rows)}. Median pool searched: "
         f"{int(statistics.median(pools)):,} structures.")
    emit()
    emit("The pipeline's single-prediction column below is the representative")
    emit("its own frozen selector ranks first, which is what a user without")
    emit("the native fragment obtains.")
    emit()
    emit("| method | structures per case | median Kabsch RMSD |")
    emit("|---|---:|---:|")
    emit(f"| this pipeline, pool minimum | {int(statistics.median(pools)):,} | "
         f"{statistics.median([r['oracle'] for r in rows]):.3f} |")
    emit(f"| this pipeline, frozen selector | 1 | "
         f"{statistics.median([r['selected'] for r in rows]):.3f} |")
    for b in BASELINES:
        vs = [r[b] for r in rows if r[b] is not None]
        ns = [r[b + "_n"] for r in rows if r[b + "_n"]]
        if not vs:
            continue
        emit(f"| {b} | {int(statistics.median(ns)) if ns else 1} | "
             f"{statistics.median(vs):.3f} |")
    emit()

    emit("## Cases won, by convention")
    emit()
    emit("| baseline | pool minimum wins | frozen selector wins |")
    emit("|---|---:|---:|")
    for b in BASELINES:
        pairs = [(r["oracle"], r["selected"], r[b])
                 for r in rows if r[b] is not None]
        if not pairs:
            continue
        w_o = sum(1 for o, _s, v in pairs if o < v)
        w_s = sum(1 for _o, s, v in pairs if s < v)
        emit(f"| {b} | {w_o}/{len(pairs)} | {w_s}/{len(pairs)} |")
    emit()
    emit("The reported advantage over AF3, ESMFold and OpenFold does not")
    emit("survive scoring this pipeline the way those models are scored. It")
    emit("holds against OmegaFold, whose recorded errors are the largest of")
    emit("the four.")
    emit()

    emit("## What this does and does not show")
    emit()
    emit("The pool minimum remains a real property of the sampler: a")
    emit("structure close to the native is present among the generated")
    emit("candidates, and on most cases it is closer than any baseline's")
    emit("prediction. That is coverage, and it is worth reporting as")
    emit("coverage.")
    emit()
    emit("What it is not is an accuracy the method can deliver. Selecting it")
    emit("requires the native structure, and when the pipeline's own")
    emit("selector chooses without that, the choice is several angstroms")
    emit("worse and the ordering against the baselines reverses.")
    emit()
    emit("Two limits belong with this. The selector here is the production")
    emit("basin ranking at its default; a different selector could recover")
    emit("some of the gap, and nothing here shows that none can. And the")
    emit("input asymmetry runs the other way: this pipeline receives the")
    emit("solved receptor, both flanking anchors, and the ligand position,")
    emit("while the baselines receive a sequence window. The reversal")
    emit("appears despite that advantage, not in its absence.")
    emit()

    emit("## Per case")
    emit()
    emit("| case | pool | pool min | selector | " + " | ".join(BASELINES) + " |")
    emit("|---|---:|---:|---:|" + "---:|" * len(BASELINES))
    for r in sorted(rows, key=lambda r: r["case_id"]):
        cells = [r["case_id"],
                 f"{int(r['pool']):,}" if r["pool"] else "--",
                 f"{r['oracle']:.3f}", f"{r['selected']:.3f}"]
        cells += [f"{r[b]:.3f}" if r[b] is not None else "--"
                  for b in BASELINES]
        emit("| " + " | ".join(cells) + " |")
    emit()

    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[written] {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
