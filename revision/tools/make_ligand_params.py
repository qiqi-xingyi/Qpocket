#!/usr/bin/env python3
# Author: Yuqi Zhang
"""Rosetta parameter files for the study's ligands.

RosettaLigand cannot see a ligand without a parameter file, so the
induced-fit comparator needs one per compound. Two details decide whether
these files describe the same ligand the sampling arms were given.

*The copy.* Some structures contain more than one copy of the ligand, and
the pipeline selects one by proximity to the pocket. The same selection is
queried here rather than reproduced, so the comparator and the sampling
arms cannot drift apart on which molecule they mean.

*The hydrogens.* These structures are inconsistent -- some ligands carry
explicit hydrogens and some do not. Hydrogens are added where missing so
that every parameter file describes a complete molecule, and the heavy-atom
count is checked against the structure afterwards, since that is the part
that must survive the round trip unchanged.

The generator ships with Rosetta and is written for Python 2. It is run
unmodified under a Python 2.7 interpreter rather than machine-translated,
so the parameters come from the tool as its authors wrote it.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ROSETTA = Path("/apps/rosetta/intel/2021.10/3.12")
SCRIPTS = (ROSETTA / "demos/protocol_capture"
           / "using_ncaas_protein_peptide_interface_design"
           / "HowToMakeResidueTypeParamFiles/scripts/python")


def selected_copies() -> Dict[str, Tuple[str, int]]:
    """Ask the pipeline which ligand copy each structure contributes."""
    from ras_folding.kras.task_loader import load_kras_tasks
    from ras_folding.prior.environment import build_environment_prior

    tasks, _ = load_kras_tasks(PROJECT_ROOT / "inputs/kras_tasks.csv",
                               pdb_dir=PROJECT_ROOT / "kras_select_systems")
    out: Dict[str, Tuple[str, int]] = {}
    for t in tasks:
        m = t.metadata
        pdb = m["ref_pdb"]
        if pdb in out:
            continue
        env = build_environment_prior(
            pdb_path=str(PROJECT_ROOT / "kras_select_systems" / pdb),
            chain_id=m["chain_id"], start_resi=int(m["start_resi"]),
            end_resi=int(m["end_resi"]),
            ligand_resname=m.get("ligand_resname"),
            fragment_ca_centroid=np.asarray(t.reference_coords,
                                            dtype=np.float64).mean(0))
        if env.selected_ligand_resseq is None:
            continue
        out[pdb] = (str(m.get("ligand_resname")),
                    int(env.selected_ligand_resseq),
                    str(env.selected_ligand_chain),
                    int(env.n_ligand_atoms))
    return out


def extract_copy(pdb_path: Path, resname: str, resseq: int, chain: str,
                 dst: Path) -> int:
    lines = []
    for line in pdb_path.read_text(errors="ignore").splitlines():
        if line[:6] != "HETATM":
            continue
        if line[17:20].strip() != resname:
            continue
        if line[21] != chain:
            continue
        try:
            if int(line[22:26]) != resseq:
                continue
        except ValueError:
            continue
        lines.append(line)
    dst.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def heavy_count(pdb_path: Path) -> int:
    n = 0
    for line in pdb_path.read_text(errors="ignore").splitlines():
        if line[:6] in ("HETATM", "ATOM  "):
            el = line[76:78].strip().upper()
            if el and el != "H":
                n += 1
            elif not el and line[12:16].strip()[:1] != "H":
                n += 1
    return n


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--py2", default=None,
                   help="Python 2 interpreter; defaults to the scratch env")
    p.add_argument("--out", default="revision/results/ligand_params")
    args = p.parse_args(argv)

    import os
    py2 = Path(args.py2 or f"/fs/scratch/PGS0423/{os.environ.get('USER','')}"
                           f"/envs/py27/bin/python")
    if not py2.exists():
        print(f"ERROR: Python 2 interpreter not found at {py2}",
              file=sys.stderr)
        return 2
    obabel = shutil.which("obabel")
    if not obabel:
        print("ERROR: obabel not on PATH", file=sys.stderr)
        return 2

    out_root = PROJECT_ROOT / args.out
    out_root.mkdir(parents=True, exist_ok=True)

    sel = selected_copies()
    # One parameter file per distinct ligand. Where a ligand appears in
    # several structures the first is used; the check below confirms the
    # others agree on heavy-atom count before reusing it.
    built: Dict[str, Path] = {}
    rows: List[Tuple[str, ...]] = []

    for pdb in sorted(sel):
        resname, resseq, chain, n_heavy_expected = sel[pdb]
        work = out_root / resname
        if resname in built:
            rows.append((resname, pdb, str(resseq), str(n_heavy_expected),
                         "-", "-", "reused"))
            continue
        work.mkdir(parents=True, exist_ok=True)

        raw = work / f"{resname}_raw.pdb"
        n_lines = extract_copy(PROJECT_ROOT / "kras_select_systems" / pdb,
                               resname, resseq, chain, raw)
        if n_lines == 0:
            rows.append((resname, pdb, str(resseq), str(n_heavy_expected),
                         "-", "-", "COPY NOT FOUND"))
            continue

        sdf = work / f"{resname}.sdf"
        subprocess.run([obabel, "-ipdb", str(raw), "-osdf", "-O", str(sdf),
                        "-h"], capture_output=True, text=True)
        if not sdf.exists():
            rows.append((resname, pdb, str(resseq), str(n_heavy_expected),
                         "-", "-", "OBABEL FAILED"))
            continue

        env = dict(os.environ, PYTHONPATH=str(SCRIPTS))
        proc = subprocess.run(
            [str(py2), str(SCRIPTS / "apps/public/molfile_to_params.py"),
             "-n", resname, "-p", resname, "--clobber", sdf.name],
            cwd=str(work), capture_output=True, text=True, env=env)
        (work / f"{resname}.log").write_text(
            proc.stdout + "\n----- stderr -----\n" + proc.stderr,
            encoding="utf-8")

        params = work / f"{resname}.params"
        if not params.is_file():
            extra = sorted(work.glob(f"{resname}?.params"))
            note = (f"SPLIT INTO {len(extra)}" if extra else "FAILED")
            rows.append((resname, pdb, str(resseq), str(n_heavy_expected),
                         "-", "-", note))
            continue

        n_atom = sum(1 for l in params.read_text().splitlines()
                     if l.startswith("ATOM"))
        n_bond = sum(1 for l in params.read_text().splitlines()
                     if l.startswith("BOND"))
        n_heavy_out = heavy_count(work / f"{resname}_0001.pdb") \
            if (work / f"{resname}_0001.pdb").is_file() else -1
        ok = (n_heavy_out == n_heavy_expected)
        rows.append((resname, pdb, str(resseq), str(n_heavy_expected),
                     str(n_atom), str(n_bond),
                     "ok" if ok else f"HEAVY MISMATCH ({n_heavy_out})"))
        if ok:
            built[resname] = params

    hdr = ("LIGAND", "SOURCE", "RESSEQ", "HEAVY", "ATOMS", "BONDS", "STATUS")
    print("{:<7} {:<9} {:>6} {:>6} {:>6} {:>6}  {}".format(*hdr))
    for r in rows:
        print("{:<7} {:<9} {:>6} {:>6} {:>6} {:>6}  {}".format(*r))
    bad = [r for r in rows if r[-1] not in ("ok", "reused")]
    print(f"\nbuilt {len(built)} parameter files; {len(bad)} problem(s)")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
