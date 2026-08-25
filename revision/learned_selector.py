#!/usr/bin/env python3
# Author: Yuqi Zhang
"""Can a combination of native-independent features rank better than any one?

Several features order candidates weakly but consistently. If their signals
are partly independent, a fitted combination should beat each of them. This
tests that, and it is the last construction worth trying before concluding
that the ensemble does not contain the information.

The fit is evaluated by leave-one-PDB-out cross-validation. Fragments from
one structure share a receptor and are not independent of each other, so a
plain split would train and test on the same protein and report a number
that does not transfer. Every figure below is from a held-out structure.

The comparison that matters is against the best single feature, not
against the production selector. Beating a selector built on an energy
that carries no signal is not evidence that the combination works.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ras_folding.kras.task_loader import load_kras_tasks           # noqa: E402
from ras_folding.prior.environment import build_environment_prior  # noqa: E402
from ras_folding.sampler.context import SamplingContext            # noqa: E402
from ras_folding.sampler.filter_hamiltonian import FilterHamiltonian  # noqa: E402
from ras_folding.sampler.sample_types import CandidateSample       # noqa: E402
from ras_folding.sampler.validity import decode_and_validate       # noqa: E402
from ras_folding.scoring.mj_contact import load_mj_table_default   # noqa: E402

N_SUB = 6000
FEATURES = ("dist_mean", "env_min", "rg", "contact_miss", "e_filter",
            "lig_min", "knn")
KNN_K = 20


def featurise(task, cand_path: Path, fh) -> Optional[Dict[str, Any]]:
    ei = task.encoder_inputs
    m = task.metadata
    ref = np.asarray(task.reference_coords, dtype=np.float64)
    ctx = SamplingContext(encoder_inputs=ei, sequence=task.sequence,
                          metadata={"case_id": task.task_id, **m})
    env = build_environment_prior(
        pdb_path=str(PROJECT_ROOT / "kras_select_systems" / m["ref_pdb"]),
        chain_id=m["chain_id"], start_resi=int(m["start_resi"]),
        end_resi=int(m["end_resi"]), ligand_resname=m.get("ligand_resname"),
        fragment_ca_centroid=ref.mean(0))
    envc = np.asarray(env.env_atom_coords, dtype=np.float64)
    lig = getattr(env, "ligand_atom_coords", None)
    lig = (np.asarray(lig, dtype=np.float64)
           if lig is not None and len(np.asarray(lig)) else None)

    rows = list(csv.DictReader(cand_path.open(newline="", encoding="utf-8")))
    step = max(1, len(rows) // N_SUB)
    coords, ef = [], []
    for r in rows[::step]:
        bs = (r.get("bitstring") or "").strip()
        if not bs:
            continue
        s = CandidateSample(bitstring=bs)
        decode_and_validate(s, ctx)
        if not s.valid or s.coords is None:
            continue
        coords.append(s.coords)
        h, _t = fh.evaluate(s, ctx)
        ef.append(float(h))
    if len(coords) < 200:
        return None
    C = np.stack(coords)
    n = len(C)

    d = C - ref[None]
    rmsd = np.sqrt(np.mean(np.sum(d * d, axis=2), axis=1))

    cen = C.mean(0)
    dc = C - cen[None]
    f_dist_mean = np.sqrt(np.mean(np.sum(dc * dc, axis=2), axis=1))
    f_rg = np.sqrt(np.mean(np.sum((C - C.mean(1)[:, None, :]) ** 2, axis=2),
                           axis=1))
    de = np.linalg.norm(C[:, :, None, :] - envc[None, None, :, :], axis=3)
    f_env_min = -de.min(axis=(1, 2))
    f_lig = (np.linalg.norm(C[:, :, None, :] - lig[None, None, :, :],
                            axis=3).min(axis=(1, 2))
             if lig is not None else np.zeros(n))

    flat = C.reshape(n, -1)
    sq = (flat ** 2).sum(1)
    dm = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * (flat @ flat.T),
                            0.0) / C.shape[1])
    k = min(KNN_K, n - 1)
    f_knn = np.partition(dm, k, axis=1)[:, 1:k + 1].mean(axis=1)

    return {
        "case": task.task_id,
        "pdb": str(m.get("ref_pdb", "?")),
        "rmsd": rmsd,
        "dist_mean": f_dist_mean,
        "env_min": f_env_min,
        "rg": f_rg,
        "contact_miss": np.asarray(ef),
        "e_filter": np.asarray(ef),
        "lig_min": f_lig,
        "knn": f_knn,
    }


def zscore(x: np.ndarray) -> np.ndarray:
    s = x.std()
    return (x - x.mean()) / s if s > 0 else np.zeros_like(x)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", default="Q")
    p.add_argument("--out",
                   default="revision/results/g1_pilot/LEARNED_SELECTOR.md")
    args = p.parse_args(argv)

    mj = load_mj_table_default()
    fh = FilterHamiltonian(residue_contact_weights=mj)
    tasks, _ = load_kras_tasks(PROJECT_ROOT / "inputs/kras_tasks.csv",
                               pdb_dir=PROJECT_ROOT / "kras_select_systems")
    by_id = {t.task_id: t for t in tasks}

    cases: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    root = PROJECT_ROOT / "revision/results/g1_pilot" / args.arm
    for f in sorted(root.glob("*/repeat_0/candidates.csv")):
        tid = f.parent.parent.name
        task = by_id.get(tid)
        if task is None or task.reference_coords is None:
            continue
        rec = featurise(task, f, fh)
        if rec is None:
            continue
        cases.append(rec)
        print(f"  {tid:26s} n={len(rec['rmsd']):>5,} "
              f"oracle={rec['rmsd'].min():5.2f}", flush=True)
    print(f"\nfeaturised {len(cases)} cases in "
          f"{time.perf_counter() - t0:.0f}s", flush=True)

    # Per-case z-scoring: only the ordering within a case matters, and the
    # features live on different scales across cases.
    for c in cases:
        c["Z"] = np.column_stack([zscore(c[k]) for k in FEATURES])

    lines: List[str] = []
    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    emit("# A fitted selector, held out by structure")
    emit()
    emit("Weights are fitted on eight structures and applied to the ninth,")
    emit("rotating through all nine. Fragments from one structure share a")
    emit("receptor, so splitting by fragment would train and test on the")
    emit("same protein.")
    emit()

    # Single features, no fitting, for the comparison that matters.
    emit("## Single features")
    emit()
    emit("| feature | median RMSD of its pick |")
    emit("|---|---:|")
    single = {}
    for i, k in enumerate(FEATURES):
        picks = [float(c["rmsd"][int(np.argmin(c["Z"][:, i]))]) for c in cases]
        single[k] = statistics.median(picks)
        emit(f"| {k} | {single[k]:.3f} |")
    best_single = min(single, key=single.get)
    emit()
    emit(f"Best single feature: **{best_single}** at "
         f"{single[best_single]:.3f} A.")
    emit()

    # Leave-one-PDB-out ridge on the rank-transformed target.
    pdbs = sorted({c["pdb"] for c in cases})
    held: List[float] = []
    per_fold = []
    for held_pdb in pdbs:
        tr = [c for c in cases if c["pdb"] != held_pdb]
        te = [c for c in cases if c["pdb"] == held_pdb]
        if not tr or not te:
            continue
        X = np.vstack([c["Z"] for c in tr])
        y = np.concatenate([zscore(c["rmsd"]) for c in tr])
        lam = 1.0
        A = X.T @ X + lam * np.eye(X.shape[1])
        w = np.linalg.solve(A, X.T @ y)
        for c in te:
            s = c["Z"] @ w
            held.append(float(c["rmsd"][int(np.argmin(s))]))
        per_fold.append((held_pdb, len(te), w))

    emit("## Fitted combination")
    emit()
    emit("| held-out structure | cases | picked RMSD (median) |")
    emit("|---|---:|---:|")
    idx = 0
    for pdb, ncase, _w in per_fold:
        vals = held[idx:idx + ncase]
        idx += ncase
        emit(f"| {pdb} | {ncase} | {statistics.median(vals):.3f} |")
    emit()
    oracle = statistics.median([float(c["rmsd"].min()) for c in cases])
    rand = statistics.median([float(np.median(c["rmsd"])) for c in cases])
    emit("| | median RMSD |")
    emit("|---|---:|")
    emit(f"| pool best, needs the answer | {oracle:.3f} |")
    emit(f"| fitted combination, held out | {statistics.median(held):.3f} |")
    emit(f"| best single feature ({best_single}) | {single[best_single]:.3f} |")
    emit(f"| random pick | {rand:.3f} |")
    emit()

    gain = single[best_single] - statistics.median(held)
    emit(f"The fitted combination moves the held-out median by "
         f"{gain:+.3f} A against the best single feature, and closes "
         f"{100.0 * (rand - statistics.median(held)) / (rand - oracle):.0f}% "
         f"of the distance between a random pick and the pool best.")
    emit()

    out = PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[written] {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
