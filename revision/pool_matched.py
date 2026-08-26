#!/usr/bin/env python3
# Author: Yuqi Zhang
"""Coverage under a matched number of delivered candidates.

Arm Q reaches a lower pool minimum than P or M, but it also delivers
roughly eight times as many candidates to the downstream under the same
proposal budget, because raw measured bitstrings survive validation at a
higher rate than the prior's pre-filtered rollouts. Those two facts cannot
be separated at the pool sizes actually produced.

This subsamples every arm to a common number of candidates and recomputes
the pool minimum, repeatedly, so the comparison is between samplers rather
than between sample sizes. The proposal-matched figures stay in the
summary; this is the second matched definition the frozen endpoints ask
for, not a replacement for the first.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ras_folding.kras.task_loader import load_kras_tasks           # noqa: E402
from ras_folding.sampler.context import SamplingContext            # noqa: E402
from ras_folding.sampler.sample_types import CandidateSample       # noqa: E402
from ras_folding.sampler.validity import decode_and_validate       # noqa: E402

ARMS = ("Q", "P", "M")
N_DRAW = 30           # independent subsamples per arm and case
MAX_READ = 250_000


def rmsds_for(task, path: Path) -> Optional[np.ndarray]:
    ei = task.encoder_inputs
    ref = np.asarray(task.reference_coords, dtype=np.float64)
    ctx = SamplingContext(encoder_inputs=ei, sequence=task.sequence,
                          metadata={"case_id": task.task_id, **task.metadata})
    out = []
    with path.open(newline="", encoding="utf-8") as fh:
        for i, r in enumerate(csv.DictReader(fh)):
            if i >= MAX_READ:
                break
            bs = (r.get("bitstring") or "").strip()
            if not bs:
                continue
            s = CandidateSample(bitstring=bs)
            decode_and_validate(s, ctx)
            if s.valid and s.coords is not None:
                d = s.coords - ref
                out.append(float(np.sqrt(np.mean(np.sum(d * d, axis=1)))))
    return np.asarray(out) if out else None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-root", default="revision/results/g1_pilot")
    p.add_argument("--out",
                   default="revision/results/g1_pilot/POOL_MATCHED.md")
    args = p.parse_args(argv)

    tasks, _ = load_kras_tasks(PROJECT_ROOT / "inputs/kras_tasks.csv",
                               pdb_dir=PROJECT_ROOT / "kras_select_systems")
    by_id = {t.task_id: t for t in tasks}
    root = PROJECT_ROOT / args.results_root

    per_case: Dict[str, Dict[str, np.ndarray]] = defaultdict(dict)
    for arm in ARMS:
        for f in sorted((root / arm).glob("*/repeat_0/candidates.csv")):
            tid = f.parent.parent.name
            task = by_id.get(tid)
            if task is None or task.reference_coords is None:
                continue
            r = rmsds_for(task, f)
            if r is not None and len(r) >= 100:
                per_case[tid][arm] = r
                print(f"  {arm} {tid:26s} n={len(r):>7,} "
                      f"best={r.min():5.2f}", flush=True)

    usable = {t: d for t, d in per_case.items() if len(d) == len(ARMS)}
    rng = np.random.default_rng(20260825)
    rows = []
    for tid, d in sorted(usable.items()):
        n_match = min(len(d[a]) for a in ARMS)
        rec: Dict[str, Any] = {"case": tid, "n_matched": int(n_match)}
        for a in ARMS:
            rec[f"full_{a}"] = float(d[a].min())
            rec[f"n_full_{a}"] = int(len(d[a]))
            mins = [float(rng.choice(d[a], n_match, replace=False).min())
                    for _ in range(N_DRAW)]
            rec[f"matched_{a}"] = float(np.median(mins))
            rec[f"matched_sd_{a}"] = float(np.std(mins))
        rows.append(rec)

    lines: List[str] = []
    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    emit("# Coverage at a matched candidate count")
    emit()
    emit(f"Cases with all three arms: {len(rows)}. Each arm is subsampled "
         f"{N_DRAW} times to the smallest pool among the three for that "
         f"case, and the pool minimum is taken from each draw.")
    emit()
    emit("| arm | median pool size | pool best, as produced | "
         "pool best, matched count |")
    emit("|---|---:|---:|---:|")
    for a in ARMS:
        emit(f"| {a} | {int(statistics.median([r[f'n_full_{a}'] for r in rows])):,} | "
             f"{statistics.median([r[f'full_{a}'] for r in rows]):.3f} | "
             f"{statistics.median([r[f'matched_{a}'] for r in rows]):.3f} |")
    emit()
    emit(f"Matched count, median across cases: "
         f"{int(statistics.median([r['n_matched'] for r in rows])):,} "
         f"candidates per arm.")
    emit()

    dq = [r["matched_Q"] - r["matched_P"] for r in rows]
    emit("## Q against P at the same candidate count")
    emit()
    emit(f"- median difference (Q - P): {statistics.median(dq):+.3f} A")
    emit(f"- cases where Q covers better: "
         f"{sum(1 for x in dq if x < 0)}/{len(dq)}")
    emit()
    emit("A difference that survives the matched count is a property of the")
    emit("sampler. One that disappears was a property of how many")
    emit("candidates each arm happened to deliver.")
    emit()
    emit("## Per case")
    emit()
    emit("| case | matched n | Q | P | M |")
    emit("|---|---:|---:|---:|---:|")
    for r in rows:
        emit(f"| {r['case']} | {r['n_matched']:,} | {r['matched_Q']:.3f} | "
             f"{r['matched_P']:.3f} | {r['matched_M']:.3f} |")
    emit()

    out = PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json.dump(rows, (out.parent / "pool_matched.json").open("w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
