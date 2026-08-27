#!/usr/bin/env python3
# Author: Yuqi Zhang
"""Arm B -- co-folding with Boltz, as an openly-weighted stand-in for AF3.

AF3 cannot be run here: its weights are not published and the web server
has no batch interface, so the recorded AF3 numbers cannot be extended to
a new analysis. Boltz is the same class of model with public weights, and
unlike a minimiser it samples with a diffusion head, so several seeds give
a genuinely spread ensemble rather than repeated copies of one answer.
That spread is what a response measure needs; an induced-fit run supplies
none, and its response measure saturates for that reason.

The input matches what the sequence-to-structure baselines were given --
the fragment with twenty flanking residues on each side -- plus the
ligand, which is the point of the comparison: whether a co-folding model
handed the ligand shows conformational response to it.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FLANK = 20
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U", "PYL": "O",
}


def chain_sequence(pdb: Path, chain: str) -> Dict[int, str]:
    """One-letter sequence by residue number, from CA records."""
    out: Dict[int, str] = {}
    for line in pdb.read_text(errors="ignore").splitlines():
        if line[:6] != "ATOM  " or line[21] != chain:
            continue
        if line[12:16].strip() != "CA" or line[16] not in (" ", "A"):
            continue
        try:
            out[int(line[22:26])] = THREE_TO_ONE.get(line[17:20].strip(), "X")
        except ValueError:
            pass
    return out


def window_sequence(seq: Dict[int, str], lo: int, hi: int,
                    flank: int) -> Tuple[str, int, int]:
    """Flanked window, clipped to the contiguous run holding the fragment.

    Unmodelled residues are common in these structures -- 7RPZ is missing
    residue 2 alone -- and a sequence spliced across such a break is not a
    chain the model would ever be given. The window is therefore trimmed
    to the uninterrupted run containing the fragment rather than abandoned,
    which costs flanking context on one side and keeps the case. The flanks
    actually used are recorded, since they are no longer symmetric.
    """
    if any(r not in seq for r in range(lo, hi + 1)):
        return "", lo, hi
    run_lo, run_hi = lo, hi
    while (run_lo - 1) in seq:
        run_lo -= 1
    while (run_hi + 1) in seq:
        run_hi += 1
    start = max(lo - flank, run_lo)
    end = min(hi + flank, run_hi)
    return "".join(seq[r] for r in range(start, end + 1)), start, end


def smiles_for(ligand: str, params_root: Path) -> Optional[str]:
    sdf = params_root / ligand / f"{ligand}.sdf"
    if not sdf.is_file():
        return None
    obabel = shutil.which("obabel")
    if not obabel:
        return None
    p = subprocess.run([obabel, "-isdf", str(sdf), "-osmi"],
                       capture_output=True, text=True)
    line = (p.stdout or "").strip().splitlines()
    if not line:
        return None
    return line[0].split("\t")[0].strip() or None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task-id", required=True)
    p.add_argument("--repeat", type=int, default=0)
    p.add_argument("--samples", type=int, default=20,
                   help="diffusion samples; this is the ensemble")
    p.add_argument("--recycling", type=int, default=3)
    p.add_argument("--no-ligand", action="store_true",
                   help="withhold the ligand, to separate response to the "
                        "ligand from variation between structures")
    p.add_argument("--tasks", default="inputs/kras_tasks.csv")
    p.add_argument("--structure-root", default="kras_select_systems")
    p.add_argument("--params-root", default="revision/results/ligand_params")
    p.add_argument("--output-root", default="revision/results/g2_boltz")
    p.add_argument("--boltz", default=None)
    p.add_argument("--no-kernels", action="store_true",
                   help="disable the accelerated triangular kernels; slower "
                        "but independent of the cuEquivariance stack")
    args = p.parse_args(argv)

    import os
    from ras_folding.kras.task_loader import load_kras_tasks
    from revision.run_arm import derive_seed
    from revision.runrecord import RunRecord, write_run_record

    root = PROJECT_ROOT
    boltz = args.boltz or (f"/fs/scratch/PGS0423/{os.environ.get('USER','')}"
                           f"/envs/boltz/bin/boltz")
    tasks, _ = load_kras_tasks(root / args.tasks,
                               pdb_dir=root / args.structure_root)
    task = next((t for t in tasks if t.task_id == args.task_id), None)
    if task is None:
        print(f"task {args.task_id} not found", file=sys.stderr)
        return 2
    m = task.metadata
    chain = str(m["chain_id"])
    lo, hi = int(m["start_resi"]), int(m["end_resi"])
    arm = "B" if not args.no_ligand else "Bapo"
    seed = derive_seed(arm, args.task_id, args.repeat)

    cell = root / args.output_root / arm / args.task_id / f"repeat_{args.repeat}"
    work = cell / "work"
    work.mkdir(parents=True, exist_ok=True)

    record = RunRecord(
        experiment="g2_cofolding", arm=arm, task_id=args.task_id,
        config={"repeat": args.repeat, "seed": seed,
                "diffusion_samples": args.samples,
                "recycling_steps": args.recycling,
                "flank_residues": FLANK,
                "ligand_included": not args.no_ligand,
                "model": "boltz",
                "no_kernels": bool(args.no_kernels),
                "af3_note": "AF3 weights are not public and its server has "
                            "no batch interface; Boltz is used as an "
                            "openly-weighted model of the same class"},
    ).start(root)

    pdb = root / args.structure_root / m["ref_pdb"]
    seq_map = chain_sequence(pdb, chain)
    seq, wstart, wend = window_sequence(seq_map, lo, hi, FLANK)
    if not seq:
        record.note("Sequence window crosses a gap in the model; this case "
                    "cannot be given to a co-folding model as one chain.")
        record.finish()
        write_run_record(record, cell / "run_record.json")
        print(f"[arm {arm}] {args.task_id} sequence gap — skipped")
        return 3

    smiles = None
    if not args.no_ligand:
        smiles = smiles_for(str(m.get("ligand_resname")),
                            root / args.params_root)
        if smiles is None:
            record.note("No SMILES available for the ligand; the run would "
                        "not be the comparison intended.")
            record.finish()
            write_run_record(record, cell / "run_record.json")
            return 4

    yaml_lines = ["version: 1", "sequences:",
                  "  - protein:", "      id: A", f"      sequence: {seq}"]
    if smiles:
        yaml_lines += ["  - ligand:", "      id: L",
                       f"      smiles: '{smiles}'"]
    inp = work / f"{args.task_id}.yaml"
    inp.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    out_dir = work / "out"
    cmd = [boltz, "predict", str(inp),
           "--out_dir", str(out_dir),
           "--diffusion_samples", str(args.samples),
           "--recycling_steps", str(args.recycling),
           "--seed", str(seed % (2 ** 31 - 1)),
           "--output_format", "pdb",
           "--use_msa_server",
           "--override"]
    if args.no_kernels:
        cmd.append("--no_kernels")

    record.inputs = {
        "ref_pdb": m["ref_pdb"], "chain_id": chain,
        "fragment": [lo, hi], "window": [wstart, wend],
        "window_length": len(seq),
        "flank_left": lo - wstart, "flank_right": wend - hi,
        "ligand_resname": m.get("ligand_resname"),
        "smiles": smiles,
        "command": " ".join(cmd),
    }

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(work))
    wall = time.perf_counter() - t0
    (work / "boltz.log").write_text(
        (proc.stdout or "") + "\n----- stderr -----\n" + (proc.stderr or ""),
        encoding="utf-8")

    # Fragment CA coordinates, indexed within the window that was folded.
    from revision.arms.rosetta_arm import extract_fragment
    produced = sorted(out_dir.rglob("*.pdb"))
    coords: List[np.ndarray] = []
    off = lo - wstart
    n_res = hi - lo + 1
    for f in produced:
        ca: Dict[int, np.ndarray] = {}
        for line in f.read_text(errors="ignore").splitlines():
            if line[:6] != "ATOM  " or line[12:16].strip() != "CA":
                continue
            try:
                ca[int(line[22:26])] = np.array(
                    [float(line[30:38]), float(line[38:46]),
                     float(line[46:54])], dtype=np.float64)
            except ValueError:
                pass
        idx = sorted(ca)
        if len(idx) < off + n_res:
            continue
        coords.append(np.stack([ca[r] for r in idx[off:off + n_res]]))

    if coords:
        C = np.stack(coords)
        np.savez_compressed(cell / "conformations.npz", coords=C,
                            ligand_xyz=np.zeros((0, 3)))
        n = C.shape[0]
        flat = C.reshape(n, -1)
        sq = (flat ** 2).sum(1)
        pw = np.sqrt(np.maximum(sq[:, None] + sq[None, :]
                                - 2 * (flat @ flat.T), 0) / C.shape[1])
        iu = np.triu_indices(n, 1)
        spread = float(np.median(pw[iu])) if n > 1 else 0.0
    else:
        spread = None

    record.results = {
        "returncode": proc.returncode,
        "n_pdbs": len(produced),
        "n_conformations": len(coords),
        "wall_seconds": round(wall, 1),
        # The measure of whether this ensemble can carry a response at all.
        "ensemble_spread_A": spread,
    }
    record.finish()
    if proc.returncode != 0:
        record.note(f"boltz exited {proc.returncode}; see work/boltz.log.")
    write_run_record(record, cell / "run_record.json")

    print(f"[arm {arm}] {args.task_id} rc={proc.returncode} "
          f"structures={len(coords)}/{args.samples} "
          f"spread={spread if spread is None else round(spread, 3)}A "
          f"wall={wall:.0f}s")
    return 0 if coords else 3


if __name__ == "__main__":
    raise SystemExit(main())
