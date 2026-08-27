#!/usr/bin/env python3
# Author: Yuqi Zhang
"""Ligand-associated response, measured against a background of fragments.

Reviewer 2's objection is that every fragment in the study was chosen for
its function, so a large response in Switch-II has nothing to be compared
with:

    "...it is unclear whether Frag-B/Frag-F's larger response reflects
    genuine ligand-associated biology or simply that these fragments have
    larger inherent conformational search spaces."

The comparison he asks for needs both sets measured the same way, not
measured on the original scale. The construction below is applied
identically to the study's fragments and to the neutral ones, so the
question -- does Switch-II stand out against fragments not chosen for
function -- is answerable even though the code that produced the
manuscript's own figures is not in this repository.

Construction, following the recorded settings of that analysis: pairwise
CA distances as features, one principal-component projection fitted per
fragment across the cases that share it, occupancy over a fixed number of
bins, and the response as the largest L1 difference in occupancy between
cases carrying different ligands. Fragment length governs the size of the
space being searched, so fragments are compared within length.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

N_BINS = 5
MAX_PER_CASE = 500


def ca_distance_features(C: np.ndarray) -> np.ndarray:
    """Pairwise CA distances, the feature set the original analysis used."""
    n, L, _ = C.shape
    iu = np.triu_indices(L, 1)
    d = np.linalg.norm(C[:, :, None, :] - C[:, None, :, :], axis=3)
    return d[:, iu[0], iu[1]]


def load_case_npz(path: Path, limit: int) -> Optional[np.ndarray]:
    """Conformations from an arm that emitted coordinates directly."""
    try:
        C = np.load(path)["coords"]
    except Exception:
        return None
    if C.shape[0] < 5:
        return None
    return C[:limit]


def load_case(path: Path, ei, ctx, limit: int) -> Optional[np.ndarray]:
    from ras_folding.sampler.sample_types import CandidateSample
    from ras_folding.sampler.validity import decode_and_validate
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    if not rows:
        return None
    step = max(1, len(rows) // limit)
    out = []
    for r in rows[::step]:
        bs = (r.get("bitstring") or "").strip()
        if not bs:
            continue
        s = CandidateSample(bitstring=bs)
        decode_and_validate(s, ctx)
        if s.valid and s.coords is not None:
            out.append(s.coords)
        if len(out) >= limit:
            break
    return np.stack(out) if len(out) >= 20 else None


def response_for_fragment(cases: List[Tuple[str, str, np.ndarray]]
                          ) -> Optional[Dict[str, Any]]:
    """Largest occupancy difference between cases with different ligands.

    One projection is fitted across all of a fragment's cases so that the
    occupancies being differenced live on the same axes; fitting per case
    would make the comparison meaningless.
    """
    if len(cases) < 2:
        return None
    feats = [ca_distance_features(C) for _lig, _cid, C in cases]
    allf = np.concatenate(feats, axis=0)
    mu = allf.mean(0)
    X = allf - mu
    # Leading component only: the response is a one-dimensional summary and
    # a second axis would need its own occupancy convention.
    _u, _s, vt = np.linalg.svd(X, full_matrices=False)
    pc1 = vt[0]
    proj_all = X @ pc1
    lo, hi = np.percentile(proj_all, [1, 99])
    edges = np.linspace(lo, hi, N_BINS + 1)

    occ: Dict[str, np.ndarray] = {}
    for (lig, cid, _C), f in zip(cases, feats):
        pr = (f - mu) @ pc1
        h, _ = np.histogram(pr, bins=edges)
        tot = h.sum()
        occ[cid] = h / tot if tot else h.astype(float)

    lig_of = {cid: lig for lig, cid, _C in cases}
    best = 0.0
    pair = None
    ids = list(occ)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            if lig_of[a] == lig_of[b]:
                continue
            l1 = float(np.abs(occ[a] - occ[b]).sum())
            if l1 > best:
                best, pair = l1, (a, b)
    # Spread of the projection is the "inherent search space" the reviewer
    # offers as the competing explanation; it is reported alongside.
    return {"max_l1": best, "pair": pair, "n_cases": len(cases),
            "proj_sd": float(np.std(proj_all))}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study-root", default="revision/results/g1_pilot/P")
    p.add_argument("--study-tasks", default="inputs/kras_tasks.csv")
    p.add_argument("--bg-root", default="revision/results/g2_neutral/P")
    p.add_argument("--bg-tasks", default="inputs/kras_neutral_tasks.csv")
    p.add_argument("--arm-root", default=None,
                   help="also measure a comparator arm, e.g. "
                        "revision/results/g2_local/I")
    p.add_argument("--arm-label", default=None)
    p.add_argument("--min-cases", type=int, default=2)
    p.add_argument("--out", default="revision/results/RESPONSE_BACKGROUND.md")
    args = p.parse_args(argv)

    from ras_folding.kras.task_loader import load_kras_tasks
    from ras_folding.sampler.context import SamplingContext

    def collect(root: str, tasks_csv: str, label: str, npz: bool = False
                ) -> Dict[str, Dict[str, Any]]:
        tasks, _ = load_kras_tasks(PROJECT_ROOT / tasks_csv,
                                   pdb_dir=PROJECT_ROOT / "kras_select_systems")
        by_frag: Dict[str, List[Tuple[str, str, np.ndarray]]] = defaultdict(list)
        length: Dict[str, int] = {}
        for t in tasks:
            frag = t.task_id.split("_")[-1]
            fname = "conformations.npz" if npz else "candidates.csv"
            f = PROJECT_ROOT / root / t.task_id / "repeat_0" / fname
            if not f.is_file():
                continue
            if npz:
                C = load_case_npz(f, MAX_PER_CASE)
            else:
                ctx = SamplingContext(encoder_inputs=t.encoder_inputs,
                                      sequence=t.sequence,
                                      metadata={"case_id": t.task_id,
                                                **t.metadata})
                C = load_case(f, t.encoder_inputs, ctx, MAX_PER_CASE)
            if C is None:
                continue
            by_frag[frag].append((str(t.metadata.get("ligand_resname")),
                                  t.task_id, C))
            length[frag] = int(t.encoder_inputs.n_residues)
        out: Dict[str, Dict[str, Any]] = {}
        for frag, cases in by_frag.items():
            r = response_for_fragment(cases)
            if r:
                r["length"] = length[frag]
                r["set"] = label
                out[frag] = r
            print(f"  {label:<10} {frag:<12} n_cases={len(cases):>2} "
                  f"L={length.get(frag)} "
                  f"max_l1={r['max_l1']:.3f}" if r else
                  f"  {label:<10} {frag:<12} skipped", flush=True)
        return out

    study = collect(args.study_root, args.study_tasks, "study")
    bg = collect(args.bg_root, args.bg_tasks, "background")
    comp: Dict[str, Dict[str, Any]] = {}
    if args.arm_root:
        comp = collect(args.arm_root, args.study_tasks,
                       args.arm_label or "comparator", npz=True)

    lines: List[str] = []
    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    emit("# Ligand-associated response against a neutral background")
    emit()
    emit("Both sets are measured by the same construction. Fragments are")
    emit("compared within length, because the size of the space searched is")
    emit("the competing explanation the objection names.")
    emit()
    emit("| fragment | set | length | cases | response | projection spread |")
    emit("|---|---|---:|---:|---:|---:|")
    for name, r in sorted(list(study.items()) + list(bg.items()),
                          key=lambda kv: (kv[1]["length"], kv[1]["set"],
                                          -kv[1]["max_l1"])):
        emit(f"| {name} | {r['set']} | {r['length']} | {r['n_cases']} | "
             f"{r['max_l1']:.3f} | {r['proj_sd']:.2f} |")
    emit()

    emit("## Where the study's fragments sit")
    emit()
    emit("| fragment | length | response | background median | percentile |")
    emit("|---|---:|---:|---:|---:|")
    for name, r in sorted(study.items(), key=lambda kv: kv[1]["length"]):
        peers = [b["max_l1"] for b in bg.values()
                 if b["length"] == r["length"]]
        if not peers:
            emit(f"| {name} | {r['length']} | {r['max_l1']:.3f} | "
                 f"no background at this length | — |")
            continue
        pct = 100.0 * sum(1 for x in peers if x < r["max_l1"]) / len(peers)
        emit(f"| {name} | {r['length']} | {r['max_l1']:.3f} | "
             f"{statistics.median(peers):.3f} | {pct:.0f} |")
    emit()
    emit("A study fragment whose response sits inside the background range")
    emit("is not evidence of ligand-associated biology, whatever its")
    emit("absolute value.")
    emit()

    if comp:
        emit("## The same measure on a comparator")
        emit()
        emit("If a comparator's ensembles show the same localisation, the")
        emit("pattern is a property of the system rather than of this")
        emit("method. Only a comparator that receives the ligand can show a")
        emit("ligand-associated response at all; one run on protein alone")
        emit("would return a null by construction and would not be evidence")
        emit("of anything.")
        emit()
        emit("| fragment | length | this method | comparator | ratio |")
        emit("|---|---:|---:|---:|---:|")
        for name, r in sorted(study.items(), key=lambda kv: kv[1]["length"]):
            c = comp.get(name)
            if not c:
                emit(f"| {name} | {r['length']} | {r['max_l1']:.3f} | — | — |")
                continue
            ratio = (r["max_l1"] / c["max_l1"]) if c["max_l1"] else float("inf")
            emit(f"| {name} | {r['length']} | {r['max_l1']:.3f} | "
                 f"{c['max_l1']:.3f} | {ratio:.2f} |")
        emit()

    out = PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json.dump({"study": study, "background": bg, "comparator": comp},
              (out.parent / "response_background.json").open("w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
