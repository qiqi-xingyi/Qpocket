#!/usr/bin/env python3
# Author: Yuqi Zhang
"""Selection by inference in the sampled subspace.

The refinement stage already builds an effective Hamiltonian over the
sampled candidates -- energies on the diagonal, couplings off it -- and
diagonalises it. That machinery is used here for selection rather than
refinement: a candidate's weight in the ground state says it is not merely
low in energy but connected to other low-energy candidates, which is a
different statement from either quantity alone.

Four variants separate where any signal comes from, because the diagonal
and the two couplings can each be removed independently:

``hybrid``   energy diagonal, Hamming-1 and RMSD couplings -- the production form
``pauli``    flat diagonal, Hamming-1 coupling only -- pure move-graph connectivity
``rmsd``     flat diagonal, RMSD kernel only -- pure geometric connectivity
``energy``   energy diagonal, RMSD kernel -- energy shaped by geometry

A flat diagonal removes the energy entirely. Since the energy orders these
candidates barely better than chance, a variant that keeps its signal
without it locates that signal in the coupling.

Each case also reports the participation ratio of the ground state, which
says whether the state concentrates on a few candidates or spreads over
many. It is not used to select; it is a per-case statement about how much
any subspace pick deserves to be trusted.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ras_folding.kras.task_loader import load_kras_tasks           # noqa: E402
from ras_folding.sampler.context import SamplingContext            # noqa: E402
from ras_folding.sampler.filter_hamiltonian import FilterHamiltonian  # noqa: E402
from ras_folding.sampler.sample_types import (                     # noqa: E402
    CandidateSample, bitstring_int_to_codes, bitstring_str_to_int,
)
from ras_folding.sampler.validity import decode_and_validate       # noqa: E402
from ras_folding.scoring.mj_contact import load_mj_table_default   # noqa: E402

N_SUB = 3000        # subspace size; the diagonalisation is dense
G_PAULI = 0.03      # production g_quantum
ALPHA_RMSD = 0.5


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3 or a.std() == 0:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = math.sqrt(float((ra ** 2).sum()) * float((rb ** 2).sum()))
    return float((ra * rb).sum() / d) if d else float("nan")


def ground_state(H: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Lowest eigenvector, spectral gap, and participation ratio."""
    w, v = np.linalg.eigh(H)
    psi = v[:, 0]
    p = psi ** 2
    p = p / p.sum()
    pr = 1.0 / float((p ** 2).sum())          # participation ratio
    gap = float(w[1] - w[0]) if len(w) > 1 else 0.0
    return p, gap, pr


def process(task, cand_path: Path, fh) -> Optional[Dict[str, Any]]:
    ei = task.encoder_inputs
    nb = int(ei.n_bonds)
    ref = np.asarray(task.reference_coords, dtype=np.float64)
    ctx = SamplingContext(encoder_inputs=ei, sequence=task.sequence,
                          metadata={"case_id": task.task_id, **task.metadata})
    rows = list(csv.DictReader(cand_path.open(newline="", encoding="utf-8")))
    step = max(1, len(rows) // N_SUB)
    coords, codes, ener = [], [], []
    for r in rows[::step]:
        bs = (r.get("bitstring") or "").strip()
        if not bs:
            continue
        s = CandidateSample(bitstring=bs)
        decode_and_validate(s, ctx)
        if not s.valid or s.coords is None:
            continue
        coords.append(s.coords)
        codes.append(bitstring_int_to_codes(bitstring_str_to_int(bs), nb))
        h, _t = fh.evaluate(s, ctx)
        ener.append(float(h))
    n = len(coords)
    if n < 200:
        return None
    C = np.stack(coords)
    K = np.stack(codes).astype(np.int64)
    E = np.asarray(ener, dtype=np.float64)
    d = C - ref[None]
    rmsd = np.sqrt(np.mean(np.sum(d * d, axis=2), axis=1))

    # Hamming-1 in code space: candidates one bond apart.
    diff = (K[:, None, :] != K[None, :, :]).sum(axis=2)
    A_pauli = (diff == 1).astype(np.float64)

    flat = C.reshape(n, -1)
    sq = (flat ** 2).sum(1)
    D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * (flat @ flat.T),
                           0.0) / C.shape[1])
    sigma = float(np.median(D)) or 1.0
    A_rmsd = np.exp(-(D ** 2) / (2 * sigma ** 2))
    np.fill_diagonal(A_rmsd, 0.0)

    Ez = (E - E.mean()) / (E.std() or 1.0)
    out: Dict[str, Any] = {"case": task.task_id, "n": n,
                           "oracle": float(rmsd.min()),
                           "random": float(np.median(rmsd))}
    variants = {
        "hybrid": (Ez, G_PAULI * A_pauli + ALPHA_RMSD * A_rmsd),
        "pauli": (np.zeros(n), G_PAULI * A_pauli),
        "rmsd": (np.zeros(n), ALPHA_RMSD * A_rmsd),
        "energy": (Ez, ALPHA_RMSD * A_rmsd),
    }
    for name, (diag, cpl) in variants.items():
        H = -cpl.copy()
        np.fill_diagonal(H, diag)
        p, gap, pr = ground_state(H)
        out[f"pick_{name}"] = float(rmsd[int(np.argmax(p))])
        out[f"rho_{name}"] = spearman(-p, rmsd)
        out[f"gap_{name}"] = gap
        out[f"pr_{name}"] = pr / n
    out["pick_energy_only"] = float(rmsd[int(np.argmin(E))])
    out["rho_energy_only"] = spearman(E, rmsd)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", default="Q")
    p.add_argument("--out",
                   default="revision/results/g1_pilot/SUBSPACE_SELECTOR.md")
    args = p.parse_args(argv)

    mj = load_mj_table_default()
    fh = FilterHamiltonian(residue_contact_weights=mj)
    tasks, _ = load_kras_tasks(PROJECT_ROOT / "inputs/kras_tasks.csv",
                               pdb_dir=PROJECT_ROOT / "kras_select_systems")
    by_id = {t.task_id: t for t in tasks}

    recs: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    root = PROJECT_ROOT / "revision/results/g1_pilot" / args.arm
    for f in sorted(root.glob("*/repeat_0/candidates.csv")):
        tid = f.parent.parent.name
        task = by_id.get(tid)
        if task is None or task.reference_coords is None:
            continue
        r = process(task, f, fh)
        if r is None:
            continue
        recs.append(r)
        print(f"  {tid:26s} n={r['n']:>5,} oracle={r['oracle']:5.2f} "
              f"hybrid={r['pick_hybrid']:5.2f} pauli={r['pick_pauli']:5.2f} "
              f"rmsd={r['pick_rmsd']:5.2f}", flush=True)

    lines: List[str] = []
    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    emit("# Selection by inference in the sampled subspace")
    emit()
    emit(f"Cases: {len(recs)}. Subspace: up to {N_SUB:,} candidates per case.")
    emit()
    emit("| variant | diagonal | coupling | median picked RMSD | median rho |")
    emit("|---|---|---|---:|---:|")
    rowspec = [("hybrid", "energy", "Hamming-1 + RMSD"),
               ("pauli", "flat", "Hamming-1"),
               ("rmsd", "flat", "RMSD kernel"),
               ("energy", "energy", "RMSD kernel")]
    for k, diag, cpl in rowspec:
        pk = [r[f"pick_{k}"] for r in recs]
        rh = [r[f"rho_{k}"] for r in recs if r[f"rho_{k}"] == r[f"rho_{k}"]]
        emit(f"| {k} | {diag} | {cpl} | {statistics.median(pk):.3f} | "
             f"{statistics.median(rh):+.3f} |")
    pk = [r["pick_energy_only"] for r in recs]
    emit(f"| energy alone, no subspace | energy | none | "
         f"{statistics.median(pk):.3f} | "
         f"{statistics.median([r['rho_energy_only'] for r in recs]):+.3f} |")
    emit()
    emit("| | median RMSD |")
    emit("|---|---:|")
    emit(f"| pool best, needs the answer | "
         f"{statistics.median([r['oracle'] for r in recs]):.3f} |")
    emit(f"| best hand-crafted feature | 5.271 |")
    emit(f"| random pick | "
         f"{statistics.median([r['random'] for r in recs]):.3f} |")
    emit()
    emit("## Ground-state localisation")
    emit()
    emit("The participation ratio is the fraction of the subspace the ground")
    emit("state actually occupies. A state spread across most of the")
    emit("subspace is not selecting anything.")
    emit()
    emit("| variant | median participation ratio |")
    emit("|---|---:|")
    for k, _d, _c in rowspec:
        emit(f"| {k} | {statistics.median([r[f'pr_{k}'] for r in recs]):.3f} |")
    emit()

    out = PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json.dump(recs, (out.parent / "subspace_selector.json").open("w"), indent=1)
    print(f"\nelapsed={time.perf_counter() - t0:.0f}s", file=sys.stderr)
    print(f"[written] {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
