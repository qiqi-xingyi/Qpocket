#!/usr/bin/env python3
# Author: Yuqi Zhang
"""Generate the frozen cell manifest for an array job.

One line == one (arm, task_id, repeat) cell. The manifest is written once
and then treated as read-only: the SLURM array indexes into it, so the
mapping from array index to experimental cell is fixed and auditable
after the fact.

    python revision/osc/make_cells.py --arms P --repeats 3 \
        --out revision/configs/g1_cells.tsv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--arms", nargs="+", default=["P"])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--tasks", default="inputs/kras_tasks.csv")
    p.add_argument("--task-ids", nargs="*", default=None,
                   help="restrict to these case_ids; default = all in CSV")
    p.add_argument("--structure-root", default="kras_select_systems")
    p.add_argument("--budget-mode", default="length_adaptive",
                   choices=["length_adaptive", "fixed"],
                   help="proposal budget rule; length_adaptive mirrors the "
                        "quantum arm's own shot allocation")
    p.add_argument("--fixed-shots", type=int, default=None,
                   help="required when --budget-mode=fixed")
    p.add_argument("--shots-per-circuit", type=int, default=8192,
                   help="allocation granularity. The quantum arm was run at "
                        "8192, not the 4096 in SHOT_BUDGET_DEFAULTS; matching "
                        "it here reproduces that arm's delivered budget "
                        "exactly rather than a recomputed approximation")
    p.add_argument("--out", default="revision/configs/g1_cells.tsv")
    args = p.parse_args(argv)

    from ras_folding.kras.task_loader import load_kras_tasks
    from ras_folding.prior.shot_budget import allocate_shots

    tasks, _schema = load_kras_tasks(
        PROJECT_ROOT / args.tasks,
        pdb_dir=PROJECT_ROOT / args.structure_root,
    )
    task_ids = [t.task_id for t in tasks]
    by_id = {t.task_id: t for t in tasks}
    if args.task_ids:
        wanted = set(args.task_ids)
        missing = wanted - set(task_ids)
        if missing:
            print(f"ERROR: unknown case_ids: {sorted(missing)}", file=sys.stderr)
            return 2
        task_ids = [t for t in task_ids if t in wanted]

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    total_proposals = 0
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        # LF endings: this file is parsed by the submit script with
        # cut, and csv.writer would otherwise emit CRLF and leave a
        # carriage return on the last field of every line.
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["index", "arm", "task_id", "repeat", "n_residues",
                    "n_proposals"])
        for arm in args.arms:
            for tid in task_ids:
                ei = by_id[tid].encoder_inputs
                # Proposal-matched budget, PER TAU. The quantum arm spent
                # its full shot allocation once per tau, so this figure is
                # multiplied by the number of taus at run time rather than
                # being the whole cell budget. The allocation is also
                # length-adaptive, so a flat per-task number would
                # over-budget short fragments and under-budget long ones.
                alloc = allocate_shots(
                    n_res=int(ei.n_residues), n_bonds=int(ei.n_bonds),
                    budget_mode=args.budget_mode,
                    fixed_shots_per_task=args.fixed_shots,
                    config_override={
                        "shots_per_circuit": args.shots_per_circuit,
                    },
                )
                n_prop = int(alloc.allocated_total_shots)
                for r in range(args.repeats):
                    n += 1
                    total_proposals += n_prop
                    w.writerow([n, arm, tid, r, int(ei.n_residues), n_prop])
    print(f"wrote {n} cells -> {out_path}")
    print(f"  arms={args.arms} tasks={len(task_ids)} repeats={args.repeats}")
    print(f"  proposals per tau, summed over cells: {total_proposals:,}")
    print(f"  x3 taus at the default setting: {total_proposals * 3:,}")
    print(f"  submit with: --array=1-{n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
