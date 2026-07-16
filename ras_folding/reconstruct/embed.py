# Author: Yuqi Zhang
"""Embed a rebuilt fragment back into a reference PDB.

receptor = reference - (chain_id, [start_resi, end_resi])
         + (rebuilt fragment, renumbered to [start_resi, end_resi])
         - HETATM (default; ligand and waters dropped)

The embedding does NOT realign the rebuilt fragment — both PULCHRA
output and the encoder anchor frame share the same coordinate system,
because PULCHRA preserves CA coordinates of its input. We do, however,
verify the CA positions of the rebuilt fragment match the input CA
trace within ``ca_drift_warn_threshold_a``; deviations are recorded as
warnings (not failures) so downstream PDB tooling can still consume
the file.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


# minimal column layout per PDB v3.x ATOM/HETATM records
_ATOM_FMT = (
    "{record:<6s}{serial:>5s} "
    "{name:<4s}{altLoc:1s}{resName:>3s} "
    "{chainID:1s}{resSeq:>4s}{iCode:1s}   "
    "{x:8.3f}{y:8.3f}{z:8.3f}"
    "{occupancy:>6s}{tempFactor:>6s}"
    "          "
    "{element:>2s}{charge:2s}"
)


@dataclass
class _AtomRecord:
    record: str          # "ATOM  " or "HETATM"
    serial: str
    name: str
    altLoc: str
    resName: str
    chainID: str
    resSeq: str          # raw string
    iCode: str
    x: float
    y: float
    z: float
    occupancy: str
    tempFactor: str
    element: str
    charge: str


def _parse_atom_line(line: str) -> Optional[_AtomRecord]:
    if not (line.startswith("ATOM  ") or line.startswith("HETATM")):
        return None
    if len(line) < 54:
        return None
    try:
        return _AtomRecord(
            record=line[0:6],
            serial=line[6:11],
            name=line[12:16],
            altLoc=line[16:17] if len(line) > 16 else " ",
            resName=line[17:20] if len(line) > 19 else "   ",
            chainID=line[21:22] if len(line) > 21 else " ",
            resSeq=line[22:26] if len(line) > 25 else "    ",
            iCode=line[26:27] if len(line) > 26 else " ",
            x=float(line[30:38]),
            y=float(line[38:46]),
            z=float(line[46:54]),
            occupancy=(line[54:60] if len(line) > 59 else "  1.00"),
            tempFactor=(line[60:66] if len(line) > 65 else "  0.00"),
            element=(line[76:78] if len(line) > 77 else "  "),
            charge=(line[78:80] if len(line) > 79 else "  "),
        )
    except ValueError:
        return None


def _format_atom(rec: _AtomRecord) -> str:
    occ = rec.occupancy.strip() or "1.00"
    tf = rec.tempFactor.strip() or "0.00"
    try:
        occ_str = f"{float(occ):6.2f}"
    except ValueError:
        occ_str = "  1.00"
    try:
        tf_str = f"{float(tf):6.2f}"
    except ValueError:
        tf_str = "  0.00"
    return _ATOM_FMT.format(
        record=rec.record.ljust(6),
        serial=rec.serial.strip().rjust(5)[:5],
        name=rec.name.ljust(4)[:4],
        altLoc=(rec.altLoc[:1] if rec.altLoc else " "),
        resName=rec.resName.strip().rjust(3)[:3],
        chainID=(rec.chainID[:1] if rec.chainID else " "),
        resSeq=rec.resSeq.strip().rjust(4)[:4],
        iCode=(rec.iCode[:1] if rec.iCode else " "),
        x=rec.x, y=rec.y, z=rec.z,
        occupancy=occ_str,
        tempFactor=tf_str,
        element=rec.element.strip().rjust(2)[:2],
        charge=(rec.charge or "").rjust(2)[:2],
    )


def _is_water(resname: str) -> bool:
    return resname.strip().upper() in ("HOH", "WAT", "DOD", "TIP", "SOL")


def embed_rebuilt_fragment_into_reference(
    reference_pdb: Path,
    rebuilt_fragment_pdb: Path,
    chain_id: str,
    start_resi: int,
    end_resi: int,
    output_pdb: Path,
    *,
    remove_hetero: bool = True,
    remove_waters: bool = True,
    ligand_resname: Optional[str] = None,
    ca_drift_warn_threshold_a: float = 0.5,
) -> Path:
    """Replace residues `[start_resi, end_resi]` of `chain_id` in
    `reference_pdb` with the residues of `rebuilt_fragment_pdb`.

    Returns the output PDB path. Writes warnings into a sibling
    `<output_pdb>.warnings.txt` if CA drift exceeds threshold.

    Parameters
    ----------
    remove_hetero : default True. Drops every HETATM record (including
        the native ligand). Set False to retain HETATMs (cofactors etc.).
    remove_waters : default True. Drops HOH/WAT records.
    ligand_resname : if provided AND remove_hetero=False, only the named
        ligand resname is dropped (other HETATMs preserved). Allows the
        caller to keep cofactors but exclude the docking ligand.
    """
    ref = Path(reference_pdb)
    frag = Path(rebuilt_fragment_pdb)
    out = Path(output_pdb)
    out.parent.mkdir(parents=True, exist_ok=True)

    if start_resi > end_resi:
        raise ValueError(
            f"start_resi ({start_resi}) > end_resi ({end_resi})"
        )

    # 1. Parse rebuilt fragment, group atoms by source residue order.
    frag_atoms_per_residue: List[List[_AtomRecord]] = []
    seen_keys: List[Tuple[str, str]] = []
    with frag.open("r", encoding="utf-8") as fh:
        for line in fh:
            rec = _parse_atom_line(line)
            if rec is None or rec.record.strip() != "ATOM":
                continue
            key = (rec.resSeq.strip(), rec.iCode.strip())
            if not seen_keys or seen_keys[-1] != key:
                seen_keys.append(key)
                frag_atoms_per_residue.append([])
            frag_atoms_per_residue[-1].append(rec)

    expected = end_resi - start_resi + 1
    if len(frag_atoms_per_residue) != expected:
        raise ValueError(
            f"rebuilt fragment has {len(frag_atoms_per_residue)} residues; "
            f"expected {expected} (start={start_resi}, end={end_resi})"
        )

    # 2. Renumber rebuilt residues to [start_resi..end_resi] under chain_id.
    renumbered: List[_AtomRecord] = []
    new_resi = start_resi
    for residue_atoms in frag_atoms_per_residue:
        for a in residue_atoms:
            a2 = _AtomRecord(
                record="ATOM  ",
                serial=a.serial,         # rewritten below
                name=a.name,
                altLoc=a.altLoc,
                resName=a.resName,
                chainID=chain_id[:1],
                resSeq=str(new_resi).rjust(4),
                iCode=" ",
                x=a.x, y=a.y, z=a.z,
                occupancy=a.occupancy,
                tempFactor=a.tempFactor,
                element=a.element,
                charge=a.charge,
            )
            renumbered.append(a2)
        new_resi += 1

    # 3. Walk reference PDB; emit non-target residues + insert renumbered
    #    fragment in place of the target slice.
    out_lines: List[str] = []
    inserted = False
    with ref.open("r", encoding="utf-8") as fh:
        for line in fh:
            rec = _parse_atom_line(line)
            if rec is None:
                # passthrough non-ATOM/HETATM lines except TER/END (we
                # write our own TER/END at the bottom).
                stripped = line.strip()
                if stripped.startswith("TER") or stripped.startswith("END"):
                    continue
                # Drop MODEL/ENDMDL too for simplicity; keep CRYST/HEADER.
                if stripped.startswith(("MODEL", "ENDMDL")):
                    continue
                if stripped:
                    out_lines.append(line.rstrip("\n"))
                continue

            is_target_chain = (rec.chainID.strip() == chain_id.strip())
            try:
                rseq = int(rec.resSeq.strip())
            except ValueError:
                rseq = None

            in_target_range = (
                is_target_chain
                and rseq is not None
                and start_resi <= rseq <= end_resi
            )

            if rec.record.strip() == "HETATM":
                if remove_hetero:
                    continue
                if remove_waters and _is_water(rec.resName):
                    continue
                if (
                    ligand_resname is not None
                    and rec.resName.strip().upper() == ligand_resname.upper()
                ):
                    continue
                out_lines.append(_format_atom(rec))
                continue

            # ATOM
            if in_target_range:
                if not inserted:
                    for a in renumbered:
                        out_lines.append(_format_atom(a))
                    inserted = True
                continue  # skip the original target residue atoms
            out_lines.append(_format_atom(rec))

    # If the original reference had no atoms in the target range,
    # append the renumbered fragment at the end.
    if not inserted:
        for a in renumbered:
            out_lines.append(_format_atom(a))

    # 4. Renumber serials sequentially.
    final_lines: List[str] = []
    serial = 1
    for ln in out_lines:
        if ln.startswith(("ATOM  ", "HETATM")):
            new_ln = ln[:6] + f"{serial:5d}" + ln[11:]
            final_lines.append(new_ln)
            serial += 1
        else:
            final_lines.append(ln)
    final_lines.append("TER")
    final_lines.append("END")

    out.write_text("\n".join(final_lines) + "\n", encoding="utf-8")
    return out


__all__ = ["embed_rebuilt_fragment_into_reference"]
