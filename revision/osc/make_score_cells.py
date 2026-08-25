#!/usr/bin/env python3
# Author: Yuqi Zhang
"""Build the scoring manifest from the cells that actually exist.

The sampling manifest lists what was meant to run. This one lists what is
present on disk and carries candidates, so a cell that produced no output
is absent here rather than becoming a scoring job that dies on a missing
file.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", default="revision/results/g1_pilot")
    p.add_argument("--arms", nargs="+", default=["Q", "P", "M"])
    p.add_argument("--out", default="revision/configs/g1_score_cells.tsv")
    args = p.parse_args(argv)

    root = PROJECT_ROOT / args.results_root
    rows = []
    for arm in args.arms:
        arm_dir = root / arm
        if not arm_dir.is_dir():
            print("WARNING: no directory for arm " + arm, file=sys.stderr)
            continue
        for cand in sorted(arm_dir.glob("*/repeat_*/candidates.csv")):
            repeat = cand.parent.name.split("_", 1)[1]
            task_id = cand.parent.parent.name
            rows.append((arm, task_id, repeat, cand.stat().st_size))

    # Largest first: the long cells start early instead of trailing the
    # array and setting the wall-clock on their own.
    rows.sort(key=lambda r: -r[3])

    out = PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["index", "arm", "task_id", "repeat", "candidates_bytes"])
        for i, (arm, task_id, repeat, size) in enumerate(rows, start=1):
            w.writerow([i, arm, task_id, repeat, size])

    by_arm = {}
    total = 0
    for arm, _t, _r, size in rows:
        by_arm[arm] = by_arm.get(arm, 0) + 1
        total += size
    print("wrote {0} scoring cells -> {1}".format(len(rows), out))
    print("  by arm: {0}".format(by_arm))
    print("  total candidate bytes: {0:.1f} GB".format(total / 1e9))
    if rows:
        print("  largest: {0:.1f} MB ({1} {2} repeat_{3})".format(
            rows[0][3] / 1e6, rows[0][0], rows[0][1], rows[0][2]))
    print("  submit with: --array=1-{0}".format(len(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
