#!/usr/bin/env python3
# Author: Yuqi Zhang
"""Arm R -- loop remodelling with Rosetta, on the same inputs.

Reviewer 2 asks for comparators that receive the structural context this
method receives, rather than sequence-to-structure predictors solving a
harder problem from less. Kinematic closure is the closest match
available: it rebuilds a backbone segment with both endpoints held fixed,
which is the problem this pipeline solves, stated in another program's
terms.

The correspondence is exact where it matters. The fragment's first and
last residues are this pipeline's anchors -- verified directly, the
anchor coordinates equal the first and last reference positions -- so
they become the loop's fixed stems and the residues between them are what
KIC rebuilds. The receptor is the same file, held fixed.

Ligand handling is where the arms cannot be matched for free. This
pipeline biases sampling toward the ligand centroid; Rosetta needs a
parameter file per ligand before it can see one at all. Both settings are
therefore produced: the remodelling itself runs on protein alone, and a
ligand-aware variant is obtained by filtering the same conformations for
ligand overlap. Which of the two is the fair comparison is stated in the
output rather than decided here.
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

ROSETTA_ROOT = "/apps/rosetta/intel/2021.10/3.12"


def read_pdb_ca(path: Path, chain: str) -> Dict[int, np.ndarray]:
    """CA coordinates by residue number, for one chain."""
    out: Dict[int, np.ndarray] = {}
    for line in path.read_text(errors="ignore").splitlines():
        if not line.startswith("ATOM"):
            continue
        if line[21] != chain or line[12:16].strip() != "CA":
            continue
        alt = line[16]
        if alt not in (" ", "A"):
            continue
        try:
            resi = int(line[22:26])
            out[resi] = np.array([float(line[30:38]), float(line[38:46]),
                                  float(line[46:54])], dtype=np.float64)
        except ValueError:
            continue
    return out


def write_receptor_pdb(src: Path, dst: Path, chain: str,
                       ligand_resname: Optional[str]) -> Dict[str, Any]:
    """Protein-only PDB for one chain, plus the ligand kept aside.

    Rosetta reads the protein; the ligand coordinates are returned so the
    ligand-aware variant can be filtered afterwards without a parameter
    file.
    """
    keep: List[str] = []
    lig: List[np.ndarray] = []
    for line in src.read_text(errors="ignore").splitlines():
        rec = line[:6]
        if rec == "ATOM  " and line[21] == chain:
            if line[16] in (" ", "A"):
                keep.append(line)
        elif rec == "HETATM" and ligand_resname:
            if line[17:20].strip() == ligand_resname:
                el = line[76:78].strip().upper()
                if el != "H":
                    try:
                        lig.append(np.array(
                            [float(line[30:38]), float(line[38:46]),
                             float(line[46:54])], dtype=np.float64))
                    except ValueError:
                        pass
    keep.append("TER")
    keep.append("END")
    dst.write_text("\n".join(keep) + "\n", encoding="utf-8")
    return {"n_atom_lines": len(keep) - 2,
            "n_ligand_heavy": len(lig),
            "ligand_xyz": np.stack(lig) if lig else None}


def write_loop_file(dst: Path, start_resi: int, end_resi: int) -> Dict[str, Any]:
    """Loop definition with the fragment's own end residues as stems.

    The interior residues are rebuilt and the two ends are not, matching
    which positions this pipeline samples and which it holds fixed.
    """
    lo, hi = start_resi + 1, end_resi - 1
    if hi < lo:
        raise ValueError(
            f"fragment {start_resi}-{end_resi} has no interior residue to "
            f"rebuild once both anchors are held fixed")
    cut = (lo + hi) // 2
    dst.write_text(f"LOOP {lo} {hi} {cut} 0 1\n", encoding="utf-8")
    return {"loop_start": lo, "loop_end": hi, "cut": cut,
            "n_rebuilt": hi - lo + 1}


# Rosetta parses -run:jran as a signed 32-bit integer. The shared seed
# derivation yields a uint32, whose upper half overflows that and is
# rejected with an empty value in the error, so it is folded into range
# here rather than changing the derivation the other arms share.
JRAN_MAX = 2_147_483_647


def build_command(exe: str, pdb: Path, loops: Path, out_dir: Path,
                  nstruct: int, seed: int) -> List[str]:
    jran = (int(seed) % (JRAN_MAX - 1)) + 1
    return [
        exe,
        "-in:file:s", str(pdb),
        "-loops:loop_file", str(loops),
        "-loops:remodel", "perturb_kic",
        "-loops:refine", "refine_kic",
        "-nstruct", str(nstruct),
        "-out:path:pdb", str(out_dir),
        "-out:pdb_gz",
        "-in:file:fullatom",
        "-constant_seed", "-jran", str(jran),
        "-mute", "all",
        "-overwrite",
    ]


def extract_fragment(pdb_gz: Path, chain: str, start_resi: int,
                     end_resi: int, n_expected: int) -> Optional[np.ndarray]:
    import gzip
    try:
        txt = gzip.open(pdb_gz, "rt", errors="ignore").read()
    except OSError:
        try:
            txt = pdb_gz.read_text(errors="ignore")
        except OSError:
            return None
    ca: Dict[int, np.ndarray] = {}
    for line in txt.splitlines():
        if not line.startswith("ATOM"):
            continue
        if line[21] != chain or line[12:16].strip() != "CA":
            continue
        try:
            resi = int(line[22:26])
        except ValueError:
            continue
        if start_resi <= resi <= end_resi:
            try:
                ca[resi] = np.array([float(line[30:38]), float(line[38:46]),
                                     float(line[46:54])], dtype=np.float64)
            except ValueError:
                pass
    if len(ca) != n_expected:
        return None
    return np.stack([ca[r] for r in sorted(ca)])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task-id", required=True)
    p.add_argument("--repeat", type=int, default=0)
    p.add_argument("--nstruct", type=int, default=200)
    p.add_argument("--tasks", default="inputs/kras_tasks.csv")
    p.add_argument("--structure-root", default="kras_select_systems")
    p.add_argument("--output-root", default="revision/results/g2_local")
    p.add_argument("--rosetta-root", default=ROSETTA_ROOT)
    p.add_argument("--exe", default="loopmodel")
    p.add_argument("--keep-pdbs", action="store_true")
    args = p.parse_args(argv)

    from ras_folding.kras.task_loader import load_kras_tasks
    from revision.run_arm import derive_seed
    from revision.runrecord import RunRecord, write_run_record

    root = PROJECT_ROOT
    tasks, _ = load_kras_tasks(root / args.tasks,
                               pdb_dir=root / args.structure_root)
    task = next((t for t in tasks if t.task_id == args.task_id), None)
    if task is None:
        print(f"task {args.task_id} not found", file=sys.stderr)
        return 2
    m = task.metadata
    ei = task.encoder_inputs
    chain = str(m["chain_id"])
    start_resi, end_resi = int(m["start_resi"]), int(m["end_resi"])
    seed = derive_seed("R", args.task_id, args.repeat)

    cell = (Path(args.output_root) if Path(args.output_root).is_absolute()
            else root / args.output_root) / "R" / args.task_id / f"repeat_{args.repeat}"
    work = cell / "work"
    work.mkdir(parents=True, exist_ok=True)

    record = RunRecord(
        experiment="g2_local_comparators", arm="R", task_id=args.task_id,
        config={"repeat": args.repeat, "seed": seed, "nstruct": args.nstruct,
                "method": "rosetta_kic", "exe": args.exe,
                "rosetta_root": args.rosetta_root},
    ).start(root)

    src = root / args.structure_root / m["ref_pdb"]
    rec_pdb = work / "receptor.pdb"
    prep = write_receptor_pdb(src, rec_pdb, chain, m.get("ligand_resname"))
    loops = work / "loops.txt"
    loop_meta = write_loop_file(loops, start_resi, end_resi)

    exe = Path(args.rosetta_root) / "bin" / args.exe
    if not exe.exists():
        exe = Path(shutil.which(args.exe) or args.exe)
    out_pdbs = work / "pdbs"
    out_pdbs.mkdir(exist_ok=True)
    cmd = build_command(str(exe), rec_pdb, loops, out_pdbs,
                        args.nstruct, seed)

    record.inputs = {
        "jran": (int(seed) % (JRAN_MAX - 1)) + 1,
        "ref_pdb": m["ref_pdb"], "chain_id": chain,
        "start_resi": start_resi, "end_resi": end_resi,
        "n_residues": int(ei.n_residues),
        "anchors_fixed": [start_resi, end_resi],
        **loop_meta,
        "n_ligand_heavy": prep["n_ligand_heavy"],
        "ligand_resname": m.get("ligand_resname"),
        "command": " ".join(cmd),
    }

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          cwd=str(work))
    wall = time.perf_counter() - t0
    (work / "rosetta.log").write_text(
        (proc.stdout or "") + "\n----- stderr -----\n" + (proc.stderr or ""),
        encoding="utf-8")

    produced = sorted(out_pdbs.glob("*.pdb*"))
    coords: List[np.ndarray] = []
    for f in produced:
        c = extract_fragment(f, chain, start_resi, end_resi,
                             int(ei.n_residues))
        if c is not None:
            coords.append(c)

    lig = prep["ligand_xyz"]
    if coords:
        C = np.stack(coords)
        np.savez_compressed(cell / "conformations.npz", coords=C,
                            ligand_xyz=lig if lig is not None
                            else np.zeros((0, 3)))
    record.results = {
        "returncode": proc.returncode,
        "n_pdbs_written": len(produced),
        "n_conformations_extracted": len(coords),
        "wall_seconds_rosetta": round(wall, 1),
        "conformations_path": str(cell / "conformations.npz"),
        "ligand_available": lig is not None,
    }
    record.finish()
    if proc.returncode != 0:
        record.note(f"Rosetta exited {proc.returncode}; see work/rosetta.log.")
    if len(coords) < args.nstruct:
        record.note(
            f"Extracted {len(coords)} of {args.nstruct} requested structures. "
            f"The shortfall is reported, not padded.")
    write_run_record(record, cell / "run_record.json")

    if not args.keep_pdbs and coords:
        shutil.rmtree(out_pdbs, ignore_errors=True)

    print(f"[arm R] {args.task_id} repeat={args.repeat} rc={proc.returncode} "
          f"structures={len(coords)}/{args.nstruct} wall={wall:.0f}s")
    return 0 if coords else 3


if __name__ == "__main__":
    raise SystemExit(main())
