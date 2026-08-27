#!/usr/bin/env python3
# Author: Yuqi Zhang
"""Arm I -- induced-fit docking with RosettaLigand.

Reviewer 2 asks for a comparator that refines the local backbone and side
chains while re-docking the ligand. Schrodinger IFD is his example, but it
is access-restricted at this facility, so the comparison uses the method
he named first in the same list -- RosettaLigand -- driven by a protocol
that lets the pocket backbone move. Rigid-receptor ligand docking would
not be the same experiment.

The receptor and fragment definition are the ones every other arm
receives. The ligand is the copy the pipeline itself selects, so the
comparator and the sampling arms cannot disagree about which molecule is
in the pocket.
"""
from __future__ import annotations

import argparse
import gzip
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from revision.arms.rosetta_arm import (                            # noqa: E402
    JRAN_MAX, extract_fragment, read_pdb_ca,
)

ROSETTA_ROOT = Path("/apps/rosetta/intel/2021.10/3.12")
LIGAND_CHAIN = "X"


def build_complex(src: Path, dst: Path, chain: str, resname: str,
                  resseq: int, lig_chain: str) -> Dict[str, Any]:
    """Receptor plus the selected ligand copy, ligand moved to its own chain.

    RosettaLigand addresses the ligand by chain, and the protocol names
    chain X, so the copy is relabelled rather than left where it sits.
    """
    prot: List[str] = []
    lig: List[str] = []
    lig_xyz: List[np.ndarray] = []
    for line in src.read_text(errors="ignore").splitlines():
        rec = line[:6]
        if rec == "ATOM  " and line[21] == chain and line[16] in (" ", "A"):
            prot.append(line)
        elif rec == "HETATM" and line[17:20].strip() == resname:
            try:
                if int(line[22:26]) != resseq:
                    continue
            except ValueError:
                continue
            # Chain and residue number are rewritten; coordinates are not.
            lig.append(line[:21] + lig_chain + f"{1:>4}" + line[26:])
            try:
                lig_xyz.append(np.array(
                    [float(line[30:38]), float(line[38:46]),
                     float(line[46:54])], dtype=np.float64))
            except ValueError:
                pass
    body = prot + ["TER"] + lig + ["TER", "END"]
    dst.write_text("\n".join(body) + "\n", encoding="utf-8")
    return {"n_protein_atoms": len(prot), "n_ligand_atoms": len(lig),
            "ligand_xyz": np.stack(lig_xyz) if lig_xyz else None}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task-id", required=True)
    p.add_argument("--repeat", type=int, default=0)
    p.add_argument("--nstruct", type=int, default=50)
    p.add_argument("--tasks", default="inputs/kras_tasks.csv")
    p.add_argument("--structure-root", default="kras_select_systems")
    p.add_argument("--output-root", default="revision/results/g2_local")
    p.add_argument("--params-root", default="revision/results/ligand_params")
    p.add_argument("--xml", default="revision/tools/induced_fit.xml")
    p.add_argument("--rosetta-root", default=str(ROSETTA_ROOT))
    p.add_argument("--keep-pdbs", action="store_true")
    args = p.parse_args(argv)

    from ras_folding.kras.task_loader import load_kras_tasks
    from ras_folding.prior.environment import build_environment_prior
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
    resname = str(m.get("ligand_resname") or "")
    seed = derive_seed("I", args.task_id, args.repeat)
    jran = (int(seed) % (JRAN_MAX - 1)) + 1

    cell = (Path(args.output_root) if Path(args.output_root).is_absolute()
            else root / args.output_root) / "I" / args.task_id / f"repeat_{args.repeat}"
    work = cell / "work"
    work.mkdir(parents=True, exist_ok=True)

    record = RunRecord(
        experiment="g2_local_comparators", arm="I", task_id=args.task_id,
        config={"repeat": args.repeat, "seed": seed, "jran": jran,
                "nstruct": args.nstruct, "method": "rosettaligand_induced_fit",
                "xml": args.xml, "ligand_chain": LIGAND_CHAIN,
                "schrodinger_ifd": "unavailable: access-restricted at this "
                                   "facility"},
    ).start(root)

    # The ligand copy is the pipeline's own choice, not an independent one.
    src = root / args.structure_root / m["ref_pdb"]
    env = build_environment_prior(
        pdb_path=str(src), chain_id=chain, start_resi=start_resi,
        end_resi=end_resi, ligand_resname=resname or None,
        fragment_ca_centroid=np.asarray(task.reference_coords,
                                        dtype=np.float64).mean(0))
    resseq = env.selected_ligand_resseq
    if resseq is None:
        record.note("No ligand copy selected for this system; the "
                    "induced-fit arm is undefined here and did not run.")
        record.finish()
        write_run_record(record, cell / "run_record.json")
        print(f"[arm I] {args.task_id} no ligand — skipped")
        return 3

    cplx = work / "complex.pdb"
    prep = build_complex(src, cplx, chain, resname, int(resseq),
                         LIGAND_CHAIN)

    params = (root / args.params_root / resname / f"{resname}.params")
    if not params.is_file():
        record.note(f"Missing parameter file {params}; run "
                    f"revision/tools/make_ligand_params.py first.")
        record.finish()
        write_run_record(record, cell / "run_record.json")
        print(f"[arm I] {args.task_id} missing params — skipped",
              file=sys.stderr)
        return 4

    exe = Path(args.rosetta_root) / "bin" / "rosetta_scripts"
    if not exe.exists():
        exe = Path(shutil.which("rosetta_scripts") or "rosetta_scripts")
    out_pdbs = work / "pdbs"
    out_pdbs.mkdir(exist_ok=True)

    cmd = [
        str(exe),
        "-in:file:s", str(cplx),
        "-in:file:extra_res_fa", str(params),
        "-parser:protocol", str(root / args.xml),
        "-nstruct", str(args.nstruct),
        "-out:path:pdb", str(out_pdbs),
        "-out:pdb_gz",
        "-in:file:fullatom",
        "-constant_seed", "-jran", str(jran),
        "-mute", "all",
        "-overwrite",
    ]
    record.inputs = {
        "ref_pdb": m["ref_pdb"], "chain_id": chain,
        "start_resi": start_resi, "end_resi": end_resi,
        "n_residues": int(ei.n_residues),
        "ligand_resname": resname,
        "ligand_resseq_selected": int(resseq),
        "ligand_selection_mode": env.ligand_selection_mode,
        "n_ligand_copies_found": int(env.n_ligand_copies_found),
        "params_file": str(params.relative_to(root))
        if params.is_relative_to(root) else str(params),
        **{k: v for k, v in prep.items() if k != "ligand_xyz"},
        "command": " ".join(cmd),
    }

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(work))
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

    drift = None
    if coords:
        C = np.stack(coords)
        native = read_pdb_ca(src, chain)
        a_lo, a_hi = native.get(start_resi), native.get(end_resi)
        if a_lo is not None and a_hi is not None:
            d_lo = np.linalg.norm(C[:, 0, :] - a_lo[None, :], axis=1)
            d_hi = np.linalg.norm(C[:, -1, :] - a_hi[None, :], axis=1)
            drift = {"anchor_left_mean": float(d_lo.mean()),
                     "anchor_left_max": float(d_lo.max()),
                     "anchor_right_mean": float(d_hi.mean()),
                     "anchor_right_max": float(d_hi.max())}
        np.savez_compressed(
            cell / "conformations.npz", coords=C,
            ligand_xyz=prep["ligand_xyz"] if prep["ligand_xyz"] is not None
            else np.zeros((0, 3)))

    record.results = {
        "returncode": proc.returncode,
        "n_pdbs_written": len(produced),
        "n_conformations_extracted": len(coords),
        "wall_seconds_rosetta": round(wall, 1),
        "conformations_path": str(cell / "conformations.npz"),
        # This protocol restrains the backbone rather than fixing it, so
        # how far the fragment ends move is measured, not assumed.
        "anchor_drift_A": drift,
    }
    record.finish()
    if proc.returncode != 0:
        record.note(f"rosetta_scripts exited {proc.returncode}; see "
                    f"work/rosetta.log.")
    if len(coords) < args.nstruct:
        record.note(f"Extracted {len(coords)} of {args.nstruct} requested; "
                    f"the shortfall is reported, not padded.")
    write_run_record(record, cell / "run_record.json")

    if not args.keep_pdbs and coords:
        shutil.rmtree(out_pdbs, ignore_errors=True)

    dmsg = (f" drift={drift['anchor_left_mean']:.2f}/"
            f"{drift['anchor_right_mean']:.2f}A" if drift else "")
    print(f"[arm I] {args.task_id} repeat={args.repeat} rc={proc.returncode} "
          f"structures={len(coords)}/{args.nstruct} wall={wall:.0f}s{dmsg}")
    return 0 if coords else 3


if __name__ == "__main__":
    raise SystemExit(main())
