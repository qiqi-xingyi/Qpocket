#!/usr/bin/env python3
# Author: Yuqi Zhang
"""Native-independent selectors built from the ensemble's own statistics.

The production selector ranks on a physical energy. On this benchmark that
energy carries almost no ordering information -- its rank correlation with
accuracy is near zero -- so a selector built on it performs close to
picking at random. These selectors use the sampled ensemble instead of the
energy function: where the samples concentrate, and how the sampled
degrees of freedom co-vary.

Features, all computable without the native fragment:

``e_full``, ``e_filter``
    The production energies, kept as the reference point.

``dist_mean``
    Distance to the pool's mean coordinates. Crude, because the mean of a
    multimodal ensemble need not resemble any real structure, but it is
    the simplest consensus statistic and a fair baseline for the rest.

``knn_density``
    Mean distance to the k nearest neighbours. Seeks the mode rather than
    the mean, so a multimodal ensemble does not pull the estimate into
    empty space between basins.

``eig_cent``
    Eigenvector centrality of an RMSD similarity graph. Weights a
    candidate by how central it is to a densely connected region, taking
    the whole neighbourhood structure into account rather than a fixed
    count of neighbours.

``ll_indep``
    Log-likelihood of a candidate's codes under the per-bond marginals of
    the ensemble. This is the null for the next feature: it captures
    everything the ensemble says about each bond considered alone.

``mi_coupling``
    Summed pointwise mutual information over bond pairs. Measures how well
    a candidate agrees with the ensemble's *coupling* structure, over and
    above the marginals. Reported separately from ``ll_indep`` so that any
    gain is attributable to the couplings and not to the marginals it
    would otherwise absorb.

Single-feature selectors fit nothing, so they are reported directly. Any
combination of features is fitted, and is evaluated only under
leave-one-PDB-out cross-validation -- fragments from one structure are not
independent of each other, and a plain split would leak between them.
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

from ras_folding.encoder.lattice import N_DIRECTIONS               # noqa: E402
from ras_folding.kras.task_loader import load_kras_tasks           # noqa: E402
from ras_folding.sampler.context import SamplingContext            # noqa: E402
from ras_folding.sampler.sample_types import (                     # noqa: E402
    CandidateSample, bitstring_int_to_codes, bitstring_str_to_int,
)
from ras_folding.sampler.validity import decode_and_validate       # noqa: E402

# Candidates decoded per case. The ensemble model is fitted by counting, so
# a larger sample costs little there; the graph features are quadratic and
# are computed on a subset of this.
N_DECODE = 40_000
N_GRAPH = 2_500
KNN = 25
PSEUDOCOUNT = 0.5


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    d = math.sqrt(float((ra ** 2).sum()) * float((rb ** 2).sum()))
    return float((ra * rb).sum() / d) if d else float("nan")


def fit_ensemble_model(codes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-bond marginals and pairwise joints, with pseudocounts.

    With 64 directions per bond the pairwise table is far larger than any
    sample can fill, so the pseudocount is doing real work: without it
    every unobserved pair would read as impossible rather than unobserved.
    """
    n, nb = codes.shape
    marg = np.full((nb, N_DIRECTIONS), PSEUDOCOUNT, dtype=np.float64)
    for i in range(nb):
        np.add.at(marg[i], codes[:, i], 1.0)
    marg /= marg.sum(axis=1, keepdims=True)

    joint = np.full((nb, nb, N_DIRECTIONS, N_DIRECTIONS), PSEUDOCOUNT,
                    dtype=np.float64)
    for i in range(nb):
        for j in range(i + 1, nb):
            np.add.at(joint[i, j], (codes[:, i], codes[:, j]), 1.0)
            joint[i, j] /= joint[i, j].sum()
            joint[j, i] = joint[i, j].T
    return marg, joint


def score_ensemble_model(codes: np.ndarray, marg: np.ndarray,
                         joint: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Independent log-likelihood and summed pointwise mutual information."""
    n, nb = codes.shape
    ll = np.zeros(n)
    for i in range(nb):
        ll += np.log(marg[i][codes[:, i]])
    mi = np.zeros(n)
    for i in range(nb):
        for j in range(i + 1, nb):
            pij = joint[i, j][codes[:, i], codes[:, j]]
            pi = marg[i][codes[:, i]]
            pj = marg[j][codes[:, j]]
            mi += np.log(pij / (pi * pj))
    return ll, mi


def graph_features(coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """k-NN density, eigenvector centrality, and the spectral gap.

    The gap between the first two eigenvalues says whether the similarity
    graph has one dominant region or several competing ones, which is a
    per-case statement about how much any consensus pick can be trusted.
    """
    n = coords.shape[0]
    flat = coords.reshape(n, -1)
    sq = (flat ** 2).sum(1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * (flat @ flat.T), 0.0)
    d = np.sqrt(d2 / coords.shape[1])          # per-residue RMSD

    k = min(KNN, n - 1)
    part = np.partition(d, k, axis=1)[:, 1:k + 1]
    knn = part.mean(axis=1)

    sigma = float(np.median(knn)) or 1.0
    w = np.exp(-(d ** 2) / (2.0 * sigma ** 2))
    np.fill_diagonal(w, 0.0)

    # Power iteration: the leading eigenvector of a non-negative similarity
    # matrix is non-negative, so it is directly usable as a centrality.
    v = np.full(n, 1.0 / math.sqrt(n))
    lam = 0.0
    for _ in range(200):
        nv = w @ v
        lam = float(np.linalg.norm(nv))
        if lam == 0:
            break
        nv /= lam
        if float(np.abs(nv - v).max()) < 1e-9:
            v = nv
            break
        v = nv
    # Deflate once for the second eigenvalue.
    w2 = w - lam * np.outer(v, v)
    u = np.random.default_rng(0).standard_normal(n)
    u /= np.linalg.norm(u)
    lam2 = 0.0
    for _ in range(200):
        nu = w2 @ u
        lam2 = float(np.linalg.norm(nu))
        if lam2 == 0:
            break
        nu /= lam2
        u = nu
    gap = (lam - lam2) / lam if lam else 0.0
    return knn, np.abs(v), float(gap)


def process_case(task, cand_path: Path) -> Optional[Dict[str, Any]]:
    ei = task.encoder_inputs
    ref = np.asarray(task.reference_coords, dtype=np.float64)
    ctx = SamplingContext(encoder_inputs=ei, sequence=task.sequence,
                          metadata={"case_id": task.task_id, **task.metadata})
    rows = list(csv.DictReader(cand_path.open(newline="", encoding="utf-8")))
    if not rows:
        return None
    step = max(1, len(rows) // N_DECODE)

    coords: List[np.ndarray] = []
    codes: List[np.ndarray] = []
    e_filter: List[float] = []
    for r in rows[::step]:
        bs = (r.get("bitstring") or "").strip()
        if not bs:
            continue
        s = CandidateSample(bitstring=bs)
        decode_and_validate(s, ctx)
        if not s.valid or s.coords is None:
            continue
        coords.append(s.coords)
        codes.append(bitstring_int_to_codes(bitstring_str_to_int(bs),
                                            int(ei.n_bonds)))
        try:
            e_filter.append(float(r.get("filter_energy")))
        except (TypeError, ValueError):
            e_filter.append(float("nan"))
    if len(coords) < 100:
        return None

    C = np.stack(coords)
    K = np.stack(codes).astype(np.int64)
    EF = np.asarray(e_filter)
    d = C - ref[None]
    rmsd = np.sqrt(np.mean(np.sum(d * d, axis=2), axis=1))

    marg, joint = fit_ensemble_model(K)
    ll, mi = score_ensemble_model(K, marg, joint)

    cen = C.mean(0)
    dc = C - cen[None]
    dist_mean = np.sqrt(np.mean(np.sum(dc * dc, axis=2), axis=1))

    # Graph features on a subset; the rest inherit their nearest member's
    # value only for reporting, never for selection.
    idx = np.arange(len(C))
    if len(C) > N_GRAPH:
        idx = np.random.default_rng(20260825).choice(len(C), N_GRAPH,
                                                     replace=False)
    knn_sub, eig_sub, gap = graph_features(C[idx])

    def pick(score: np.ndarray, subset: Optional[np.ndarray] = None,
             largest: bool = False) -> float:
        ii = subset if subset is not None else np.arange(len(rmsd))
        j = int(np.argmax(score) if largest else np.argmin(score))
        return float(rmsd[ii[j]])

    out = {
        "case": task.task_id,
        "pdb": task.metadata.get("ref_pdb", "?"),
        "n": int(len(C)),
        "oracle": float(rmsd.min()),
        "random": float(np.median(rmsd)),
        "pick_e_filter": pick(EF) if np.isfinite(EF).any() else None,
        "pick_dist_mean": pick(dist_mean),
        "pick_ll_indep": pick(-ll),
        "pick_mi": pick(-(ll + mi)),
        "pick_mi_only": pick(-mi),
        "pick_knn": pick(knn_sub, idx),
        "pick_eig": pick(eig_sub, idx, largest=True),
        "spectral_gap": gap,
        "rho_e_filter": spearman(EF, rmsd) if np.isfinite(EF).any() else None,
        "rho_dist_mean": spearman(dist_mean, rmsd),
        "rho_ll_indep": spearman(-ll, rmsd),
        "rho_mi": spearman(-(ll + mi), rmsd),
        "rho_mi_only": spearman(-mi, rmsd),
        "rho_knn": spearman(knn_sub, rmsd[idx]),
        "rho_eig": spearman(-eig_sub, rmsd[idx]),
    }
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", default="Q")
    p.add_argument("--results-root", default="revision/results/g1_pilot")
    p.add_argument("--out",
                   default="revision/results/g1_pilot/ensemble_selectors.json")
    args = p.parse_args(argv)

    tasks, _ = load_kras_tasks(PROJECT_ROOT / "inputs/kras_tasks.csv",
                               pdb_dir=PROJECT_ROOT / "kras_select_systems")
    by_id = {t.task_id: t for t in tasks}

    root = PROJECT_ROOT / args.results_root / args.arm
    out: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    for f in sorted(root.glob("*/repeat_0/candidates.csv")):
        tid = f.parent.parent.name
        task = by_id.get(tid)
        if task is None or task.reference_coords is None:
            continue
        rec = process_case(task, f)
        if rec is None:
            continue
        out.append(rec)
        print(f"  {tid:26s} n={rec['n']:>6,} oracle={rec['oracle']:6.3f} "
              f"eig={rec['pick_eig']:6.3f} mi={rec['pick_mi']:6.3f} "
              f"gap={rec['spectral_gap']:.3f}", flush=True)

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1), encoding="utf-8")

    print(f"\ncases={len(out)}  elapsed={time.perf_counter() - t0:.0f}s")
    print("\nRMSD of the candidate each selector picks (median, A):")
    for k, lbl in (("oracle", "pool best (needs the answer)"),
                   ("pick_eig", "eigenvector centrality"),
                   ("pick_knn", "k-NN density"),
                   ("pick_dist_mean", "distance to mean"),
                   ("pick_mi", "marginals + mutual information"),
                   ("pick_ll_indep", "marginals alone"),
                   ("pick_mi_only", "mutual information alone"),
                   ("pick_e_filter", "filter energy"),
                   ("random", "pool median (random pick)")):
        v = [r[k] for r in out if r.get(k) is not None]
        if v:
            print(f"   {lbl:34s} {statistics.median(v):6.3f}")
    print("\nSpearman rho(feature, RMSD) -- higher means better ordering:")
    for k, lbl in (("rho_eig", "eigenvector centrality"),
                   ("rho_knn", "k-NN density"),
                   ("rho_dist_mean", "distance to mean"),
                   ("rho_mi", "marginals + MI"),
                   ("rho_ll_indep", "marginals alone"),
                   ("rho_mi_only", "MI alone"),
                   ("rho_e_filter", "filter energy")):
        v = [r[k] for r in out if r.get(k) is not None and r[k] == r[k]]
        if v:
            print(f"   {lbl:26s} median={statistics.median(v):+.3f}  "
                  f"positive {sum(1 for x in v if x > 0)}/{len(v)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
