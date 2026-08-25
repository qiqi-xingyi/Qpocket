#!/usr/bin/env python3
# Author: Yuqi Zhang
"""A learned ranker over candidate conformations, held out by structure.

Hand-crafted scalars order these candidates weakly, and a linear
combination of them adds almost nothing. That does not settle whether the
information is absent: a scalar summary may simply discard it. This model
reads the structure instead -- per-residue geometry in the frame the
anchors define, together with what surrounds each residue in the receptor
and the ligand -- and learns the ordering directly.

What the setup can and cannot establish is fixed by the data. Candidates
are plentiful, but the independent unit is the structure and there are
nine. Training therefore sees eight, and only the held-out structure's
numbers are reported. A model that separates conformations well on the
proteins it was trained on and poorly on a new one has learned the
protein, and this split is what exposes that.

The comparison is against the best hand-crafted feature. Beating a
selector built on an energy that carries no signal proves nothing.
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
from ras_folding.prior.environment import build_environment_prior  # noqa: E402
from ras_folding.sampler.context import SamplingContext            # noqa: E402
from ras_folding.sampler.sample_types import CandidateSample       # noqa: E402
from ras_folding.sampler.validity import decode_and_validate       # noqa: E402

N_SUB = 6000          # candidates per case
MAX_RES = 15          # longest fragment; shorter ones are masked
SHELLS = (5.0, 8.0, 12.0)
CHUNK = 400           # candidates per distance-matrix chunk
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX = {c: i for i, c in enumerate(AA)}


def anchor_frame(a_left: np.ndarray, a_right: np.ndarray) -> np.ndarray:
    """Orthonormal frame from the two fixed anchors.

    Expressing coordinates here makes the representation invariant to how
    the receptor happens to sit in the file, while keeping the geometry the
    anchors impose -- which is the part of the input that carries signal.
    """
    x = a_right - a_left
    nx = np.linalg.norm(x)
    x = x / nx if nx > 0 else np.array([1.0, 0.0, 0.0])
    t = np.array([0.0, 0.0, 1.0])
    if abs(float(x @ t)) > 0.9:
        t = np.array([0.0, 1.0, 0.0])
    y = t - float(t @ x) * x
    ny = np.linalg.norm(y)
    y = y / ny if ny > 0 else np.array([0.0, 1.0, 0.0])
    return np.stack([x, y, np.cross(x, y)], axis=0)


def pseudo_angles(C: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """CA pseudo bond angles and dihedrals, padded to the residue axis."""
    n, L, _ = C.shape
    v = C[:, 1:] - C[:, :-1]
    vn = v / np.clip(np.linalg.norm(v, axis=2, keepdims=True), 1e-8, None)
    cosang = np.sum(-vn[:, :-1] * vn[:, 1:], axis=2)
    ang = np.zeros((n, L))
    ang[:, 1:L - 1] = np.arccos(np.clip(cosang, -1.0, 1.0))
    b1, b2, b3 = vn[:, :-2], vn[:, 1:-1], vn[:, 2:]
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    m = np.cross(n1, b2)
    x = np.sum(n1 * n2, axis=2)
    y = np.sum(m * n2, axis=2)
    dih = np.zeros((n, L))
    if x.shape[1]:
        dih[:, 1:1 + x.shape[1]] = np.arctan2(y, x)
    return ang, dih


def shell_counts(C: np.ndarray, pts: Optional[np.ndarray]) -> np.ndarray:
    """Per-residue neighbour counts and nearest distance, in chunks."""
    n, L, _ = C.shape
    out = np.zeros((n, L, len(SHELLS) + 1), dtype=np.float32)
    if pts is None or len(pts) == 0:
        out[..., -1] = 99.0
        return out
    for s in range(0, n, CHUNK):
        e = min(n, s + CHUNK)
        d = np.linalg.norm(C[s:e, :, None, :] - pts[None, None, :, :], axis=3)
        for k, r in enumerate(SHELLS):
            out[s:e, :, k] = (d < r).sum(axis=2)
        out[s:e, :, -1] = d.min(axis=2)
    return out


def featurise(task, cand_path: Path) -> Optional[Dict[str, Any]]:
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
    coords = []
    for r in rows[::step]:
        bs = (r.get("bitstring") or "").strip()
        if not bs:
            continue
        s = CandidateSample(bitstring=bs)
        decode_and_validate(s, ctx)
        if s.valid and s.coords is not None:
            coords.append(s.coords)
    if len(coords) < 200:
        return None
    C = np.stack(coords)
    n, L, _ = C.shape

    d = C - ref[None]
    rmsd = np.sqrt(np.mean(np.sum(d * d, axis=2), axis=1))

    R = anchor_frame(np.asarray(ei.anchor_left, dtype=np.float64),
                     np.asarray(ei.anchor_right, dtype=np.float64))
    local = (C - np.asarray(ei.anchor_left, dtype=np.float64)[None, None, :])
    local = local @ R.T

    ang, dih = pseudo_angles(C)
    envf = shell_counts(C, envc)
    ligf = shell_counts(C, lig)

    seq = task.sequence or ""
    sid = np.zeros((L,), dtype=np.int64)
    for i in range(min(L, len(seq))):
        sid[i] = AA_IDX.get(seq[i], 0)

    F = np.concatenate([
        local.astype(np.float32),
        np.cos(ang)[..., None].astype(np.float32),
        np.sin(ang)[..., None].astype(np.float32),
        np.cos(dih)[..., None].astype(np.float32),
        np.sin(dih)[..., None].astype(np.float32),
        envf, ligf,
    ], axis=2)

    pad = MAX_RES - L
    if pad > 0:
        F = np.pad(F, ((0, 0), (0, pad), (0, 0)))
        sid = np.pad(sid, (0, pad))
    mask = np.zeros((MAX_RES,), dtype=np.float32)
    mask[:L] = 1.0

    return {"case": task.task_id, "pdb": str(m.get("ref_pdb", "?")),
            "F": F.astype(np.float32), "sid": sid, "mask": mask,
            "rmsd": rmsd.astype(np.float32), "n_res": L}


def build_model(n_feat: int, torch):
    nn = torch.nn
    class Ranker(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(len(AA), 8)
            d = n_feat + 8
            self.inp = nn.Sequential(nn.Linear(d, 64), nn.GELU(),
                                     nn.Linear(64, 64), nn.GELU())
            # Convolution over the residue axis: what makes a conformation
            # wrong is local backbone geometry in context, not any single
            # residue on its own.
            self.conv = nn.Sequential(
                nn.Conv1d(64, 64, 3, padding=1), nn.GELU(),
                nn.Conv1d(64, 64, 3, padding=1), nn.GELU())
            self.head = nn.Sequential(nn.Linear(128, 64), nn.GELU(),
                                      nn.Dropout(0.1), nn.Linear(64, 1))

        def forward(self, F, sid, mask):
            h = torch.cat([F, self.emb(sid)], dim=-1)
            h = self.inp(h)
            h = self.conv(h.transpose(1, 2)).transpose(1, 2)
            m = mask.unsqueeze(-1)
            h = h * m
            pooled = torch.cat([h.sum(1) / m.sum(1).clamp(min=1),
                                h.max(1).values], dim=-1)
            return self.head(pooled).squeeze(-1)
    return Ranker()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", default="Q")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--pairs", type=int, default=20000,
                   help="ranking pairs sampled per epoch per training case")
    p.add_argument("--cache",
                   default="revision/results/g1_pilot/dl_features.npz")
    p.add_argument("--out", default="revision/results/g1_pilot/DL_RANKER.md")
    args = p.parse_args(argv)

    import torch
    torch.manual_seed(20260825)

    cache = PROJECT_ROOT / args.cache
    if cache.is_file():
        z = np.load(cache, allow_pickle=True)
        cases = list(z["cases"])
        print(f"loaded {len(cases)} cases from cache")
    else:
        tasks, _ = load_kras_tasks(PROJECT_ROOT / "inputs/kras_tasks.csv",
                                   pdb_dir=PROJECT_ROOT / "kras_select_systems")
        by_id = {t.task_id: t for t in tasks}
        cases = []
        t0 = time.perf_counter()
        root = PROJECT_ROOT / "revision/results/g1_pilot" / args.arm
        for f in sorted(root.glob("*/repeat_0/candidates.csv")):
            tid = f.parent.parent.name
            task = by_id.get(tid)
            if task is None or task.reference_coords is None:
                continue
            rec = featurise(task, f)
            if rec is None:
                continue
            cases.append(rec)
            print(f"  {tid:26s} n={len(rec['rmsd']):>5,} "
                  f"oracle={rec['rmsd'].min():5.2f}", flush=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, cases=np.array(cases, dtype=object))
        print(f"featurised in {time.perf_counter() - t0:.0f}s")

    n_feat = cases[0]["F"].shape[2]
    pdbs = sorted({c["pdb"] for c in cases})
    print(f"\ncases={len(cases)}  structures={len(pdbs)}  n_feat={n_feat}")

    lines: List[str] = []
    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    emit("# Learned ranker, held out by structure")
    emit()
    emit("Trained on eight structures and applied to the ninth, rotating")
    emit("through all nine. Only held-out numbers appear here.")
    emit()

    held: Dict[str, float] = {}
    train_seen: Dict[str, float] = {}
    for fold in pdbs:
        tr = [c for c in cases if c["pdb"] != fold]
        te = [c for c in cases if c["pdb"] == fold]
        if not tr or not te:
            continue
        model = build_model(n_feat, torch)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3,
                                weight_decay=1e-4)
        rng = np.random.default_rng(0)
        model.train()
        for ep in range(args.epochs):
            tot = 0.0
            for c in tr:
                nn_ = len(c["rmsd"])
                i = rng.integers(0, nn_, args.pairs)
                j = rng.integers(0, nn_, args.pairs)
                keep = c["rmsd"][i] != c["rmsd"][j]
                i, j = i[keep], j[keep]
                if len(i) < 2:
                    continue
                F = torch.from_numpy(c["F"])
                sid = torch.from_numpy(c["sid"]).unsqueeze(0)
                mask = torch.from_numpy(c["mask"]).unsqueeze(0)
                idx = np.concatenate([i, j])
                out = model(F[idx], sid.expand(len(idx), -1),
                            mask.expand(len(idx), -1))
                si, sj = out[:len(i)], out[len(i):]
                # Lower score should mean lower RMSD.
                target = torch.from_numpy(
                    np.where(c["rmsd"][i] < c["rmsd"][j], -1.0, 1.0
                             ).astype(np.float32))
                loss = torch.nn.functional.margin_ranking_loss(
                    si, sj, -target, margin=0.5)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(loss)
            print(f"  [{fold}] epoch {ep + 1}/{args.epochs} "
                  f"loss={tot / max(len(tr), 1):.4f}", flush=True)

        model.eval()
        with torch.no_grad():
            for c in te:
                s = model(torch.from_numpy(c["F"]),
                          torch.from_numpy(c["sid"]).unsqueeze(0).expand(
                              len(c["rmsd"]), -1),
                          torch.from_numpy(c["mask"]).unsqueeze(0).expand(
                              len(c["rmsd"]), -1)).numpy()
                held[c["case"]] = float(c["rmsd"][int(np.argmin(s))])
            # One training case, to show the gap between fit and transfer.
            c = tr[0]
            s = model(torch.from_numpy(c["F"]),
                      torch.from_numpy(c["sid"]).unsqueeze(0).expand(
                          len(c["rmsd"]), -1),
                      torch.from_numpy(c["mask"]).unsqueeze(0).expand(
                          len(c["rmsd"]), -1)).numpy()
            train_seen[f"{fold}:{c['case']}"] = float(
                c["rmsd"][int(np.argmin(s))])

    oracle = statistics.median([float(c["rmsd"].min()) for c in cases])
    rand = statistics.median([float(np.median(c["rmsd"])) for c in cases])
    emit("| | median RMSD (A) |")
    emit("|---|---:|")
    emit(f"| pool best, needs the answer | {oracle:.3f} |")
    emit(f"| learned ranker, held out | {statistics.median(held.values()):.3f} |")
    emit(f"| learned ranker, on training structures | "
         f"{statistics.median(train_seen.values()):.3f} |")
    emit(f"| best hand-crafted feature | 5.271 |")
    emit(f"| random pick | {rand:.3f} |")
    emit()
    emit("The training-structure row is diagnostic, not a result. A large")
    emit("gap between it and the held-out row means the model separated the")
    emit("proteins it saw rather than the conformations.")
    emit()
    emit("## Per held-out case")
    emit()
    emit("| case | picked RMSD |")
    emit("|---|---:|")
    for k in sorted(held):
        emit(f"| {k} | {held[k]:.3f} |")
    emit()

    out = PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[written] {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
