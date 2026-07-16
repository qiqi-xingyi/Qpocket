# Author: Yuqi Zhang
"""Load KRAS fragment task definitions and build the geometric context that
ras_folding.encoder.EncoderInputs.from_fragment_context(ctx) consumes.

Input artifacts
---------------
1) Tasks CSV (e.g. inputs/kras_tasks.csv) with columns:
     case_id, ref_pdb, chain_id, start_resi, end_resi,
     ligand_resname, scale_factor, scale_mode, has_native_ref

2) Reference structure directory (e.g. kras_select_systems/) containing the
   PDB file referenced by `ref_pdb` for each row.

Output (consumed by encoder)
----------------------------
`FragmentContext` — a frozen dataclass exposing exactly the attributes
EncoderInputs.from_fragment_context reads:
  - n_residues          : int
  - fragment_ca_ref     : (n_residues, 3) float64 ndarray
  - flank_before_ca     : (k_before, 3)   float64 ndarray (may be (0, 3))
  - flank_after_ca      : (k_after, 3)    float64 ndarray (may be (0, 3))
plus task metadata copied verbatim from the CSV row (case_id, ref_pdb,
chain_id, start_resi, end_resi, ligand_resname, scale_factor, scale_mode,
has_native_ref) and the one-letter `sequence` recovered from the PDB.

Encoder consumption
-------------------
    ctx = build_fragment_context(task_row, pdb_dir=PDB_DIR, flank_size=1)
    inputs = EncoderInputs.from_fragment_context(ctx)
    # inputs is now ready for decode_bitstring(...)

Design notes (surfaced, not hidden)
-----------------------------------
- `flank_size` default = 1.  EncoderInputs.from_fragment_context only consumes
  `flank_before_ca[-1]` and `flank_after_ca[0]` (single residue each), so 1 is
  the minimum that lets the encoder avoid its native-direction fallback
  branch (an information leak, see inputs.py:86-88). Larger flanks are
  preserved verbatim and ignored by the current encoder; downstream stages
  may use them.
- The PDB parser is a minimal column-based reader per PDB v3.x spec for ATOM
  records. It does NOT use Biopython. Behavior at boundaries:
    * Only ATOM records are accepted (HETATM ignored — ligand atoms are
      surfaced separately via `ligand_resname`, not as residues).
    * Only chain == `chain_id` is kept.
    * Only `atom_name == "CA"` rows are kept.
    * altLoc filter: blank or 'A'. First match wins per residue.
    * Multi-MODEL: only MODEL 1 is read (parser stops at first ENDMDL).
    * Insertion codes (iCode != ' ') in the requested range cause an error.
      The CSV uses plain integer resSeq, so any iCode in-range is unhandled
      and we surface it instead of silently skipping.
- 3-letter → 1-letter mapping covers the standard 20 amino acids only.
  Non-standard residues in the requested range raise — we do NOT map to 'X'
  silently. Modified residues (MSE, etc.) must be handled by the caller.

This module does NOT mutate state and writes no files.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------- #
# Constants                                                              #
# ---------------------------------------------------------------------- #

# Standard 20-AA three-letter to one-letter map. Anything else raises.
_THREE_TO_ONE: Dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


# ---------------------------------------------------------------------- #
# Data classes                                                            #
# ---------------------------------------------------------------------- #

@dataclass(frozen=True)
class TaskRow:
    """One row of the tasks CSV, typed."""
    case_id: str
    ref_pdb: str
    chain_id: str
    start_resi: int
    end_resi: int
    ligand_resname: Optional[str]
    scale_factor: float
    scale_mode: str
    has_native_ref: bool

    @property
    def n_residues_expected(self) -> int:
        return self.end_resi - self.start_resi + 1


@dataclass(frozen=True)
class FragmentContext:
    """Geometric + metadata context consumed by EncoderInputs.from_fragment_context.

    The four attributes EncoderInputs reads are:
        n_residues, fragment_ca_ref, flank_before_ca, flank_after_ca
    All other fields are pass-through metadata for downstream stages.
    """
    # encoder-consumed
    fragment_ca_ref: np.ndarray              # (n_residues, 3) float64
    flank_before_ca: np.ndarray              # (k_before, 3) float64
    flank_after_ca: np.ndarray               # (k_after, 3) float64
    sequence: str                            # one-letter, length == n_residues
    # task metadata (verbatim from CSV)
    case_id: str
    ref_pdb: str
    chain_id: str
    start_resi: int
    end_resi: int
    ligand_resname: Optional[str]
    scale_factor: float
    scale_mode: str
    has_native_ref: bool
    # actual residue numbers of the flank CAs (for traceability)
    flank_before_resi: Tuple[int, ...] = field(default_factory=tuple)
    flank_after_resi: Tuple[int, ...] = field(default_factory=tuple)

    @property
    def n_residues(self) -> int:
        return len(self.sequence)


# ---------------------------------------------------------------------- #
# CSV loader                                                             #
# ---------------------------------------------------------------------- #

def _parse_bool(val: str) -> bool:
    s = val.strip().lower()
    if s in ("true", "1", "yes", "y", "t"):
        return True
    if s in ("false", "0", "no", "n", "f"):
        return False
    raise ValueError(f"unrecognized boolean value: {val!r}")


def load_tasks_csv(csv_path: str) -> List[TaskRow]:
    """Parse the tasks CSV into a list of TaskRow.

    Required columns:
        case_id, ref_pdb, chain_id, start_resi, end_resi,
        ligand_resname, scale_factor, scale_mode, has_native_ref
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"tasks CSV not found: {csv_path}")
    out: List[TaskRow] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        required = {
            "case_id", "ref_pdb", "chain_id", "start_resi", "end_resi",
            "ligand_resname", "scale_factor", "scale_mode", "has_native_ref",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"tasks CSV missing required columns: {sorted(missing)}; "
                f"found: {reader.fieldnames}"
            )
        for i, row in enumerate(reader, start=2):  # header is line 1
            try:
                lig = row["ligand_resname"].strip()
                out.append(TaskRow(
                    case_id=row["case_id"].strip(),
                    ref_pdb=row["ref_pdb"].strip(),
                    chain_id=row["chain_id"].strip(),
                    start_resi=int(row["start_resi"]),
                    end_resi=int(row["end_resi"]),
                    ligand_resname=(lig if lig else None),
                    scale_factor=float(row["scale_factor"]),
                    scale_mode=row["scale_mode"].strip(),
                    has_native_ref=_parse_bool(row["has_native_ref"]),
                ))
            except (KeyError, ValueError) as e:
                raise ValueError(f"malformed CSV row at line {i}: {e}") from e
    return out


# ---------------------------------------------------------------------- #
# PDB CA parser                                                          #
# ---------------------------------------------------------------------- #

@dataclass(frozen=True)
class _CARecord:
    """One CA atom keyed by integer resSeq."""
    resi: int
    icode: str        # raw insertion code, ' ' if none
    res_three: str    # 3-letter residue name
    xyz: np.ndarray   # (3,) float64


def parse_pdb_ca(pdb_path: str, chain_id: str) -> Dict[int, _CARecord]:
    """Read CA atoms from MODEL 1 of `pdb_path` on chain `chain_id`.

    Returns
    -------
    dict mapping resSeq (int) -> _CARecord. First seen altLoc per residue
    is kept (altLoc must be ' ' or 'A').
    """
    if not os.path.isfile(pdb_path):
        raise FileNotFoundError(f"PDB not found: {pdb_path}")

    out: Dict[int, _CARecord] = {}
    in_model = True  # PDBs without MODEL records are treated as a single model
    with open(pdb_path, "r", encoding="utf-8") as fh:
        for line in fh:
            rec = line[0:6].strip()
            if rec == "MODEL":
                # Reset: treat the first MODEL as the active one.
                if out:
                    # We've already finished MODEL 1 (out has data) and a
                    # new MODEL is starting. Stop reading.
                    break
                in_model = True
                continue
            if rec == "ENDMDL":
                # End of MODEL 1. Stop after first MODEL block.
                break
            if rec != "ATOM":
                continue
            if not in_model:
                continue

            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue

            altloc = line[16:17]
            if altloc not in (" ", "A"):
                continue

            ch = line[21:22]
            if ch != chain_id:
                continue

            res_three = line[17:20].strip()
            try:
                resi = int(line[22:26])
            except ValueError:
                # malformed residue number column — skip this line silently
                # would be a fake fallback; raise instead.
                raise ValueError(
                    f"non-integer resSeq in {pdb_path}: {line.rstrip()!r}"
                )
            icode = line[26:27]
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError as e:
                raise ValueError(
                    f"malformed coordinate in {pdb_path}: {line.rstrip()!r}"
                ) from e

            if resi in out:
                # First altLoc wins; subsequent altLocs ignored. (We already
                # filtered to ' ' or 'A' above, so this only triggers if a
                # PDB has duplicated CA records.)
                continue
            out[resi] = _CARecord(
                resi=resi,
                icode=icode,
                res_three=res_three,
                xyz=np.array([x, y, z], dtype=np.float64),
            )
    return out


def _three_to_one(res_three: str, *, where: str) -> str:
    aa = _THREE_TO_ONE.get(res_three.upper())
    if aa is None:
        raise ValueError(
            f"unsupported residue {res_three!r} in {where}; "
            f"only standard 20 AAs are mapped"
        )
    return aa


# ---------------------------------------------------------------------- #
# PDB SEQRES parser (used only by the lenient context builder)           #
# ---------------------------------------------------------------------- #

def parse_pdb_seqres(pdb_path: str, chain_id: str) -> List[str]:
    """Read SEQRES records for a chain. Returns the ordered list of
    3-letter residue codes for `chain_id`. Empty list if no SEQRES.

    PDB SEQRES format:
        SEQRES <serNum> <chainID> <numRes>  <resName1> <resName2> ...
    Columns 12 = chain id; columns 19+ = whitespace-separated 3-letter codes.
    """
    if not os.path.isfile(pdb_path):
        raise FileNotFoundError(f"PDB not found: {pdb_path}")
    out: List[str] = []
    with open(pdb_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.startswith("SEQRES"):
                continue
            # column 11 is the chain id (0-indexed)
            if len(line) < 12 or line[11] != chain_id:
                continue
            # res names start at column 19
            tail = line[19:].strip()
            for tok in tail.split():
                if len(tok) == 3 and tok.isalpha():
                    out.append(tok.upper())
    return out


def _calibrate_seqres_offset(
    seqres: List[str],
    ca_records: Dict[int, _CARecord],
) -> Optional[int]:
    """Find the offset such that seqres[resSeq - offset - 1] equals the
    observed three-letter code for every CA-record. Returns the offset
    if a unique consistent offset exists with at least 5 confirming
    matches; otherwise None.

    Convention: SEQRES is 1-indexed (position 1 = seqres[0]). The offset
    is `resSeq - SEQRES_position`, i.e. position(1-indexed) = resSeq - offset.
    """
    if not seqres or not ca_records:
        return None
    # candidate offsets are resSeq - position_of_match in seqres for each
    # observed res_three. Tally votes across all CA records.
    from collections import Counter
    votes: Counter = Counter()
    for resi, rec in ca_records.items():
        three = rec.res_three.upper()
        for pos1, s in enumerate(seqres, start=1):
            if s == three:
                votes[resi - pos1] += 1
    if not votes:
        return None
    best_offset, best_count = votes.most_common(1)[0]
    if best_count < 5:
        return None
    # double-check consistency: this offset must agree with every CA-record
    # that lies inside the SEQRES range
    n_checked = 0
    for resi, rec in ca_records.items():
        pos1 = resi - best_offset
        idx = pos1 - 1
        if 0 <= idx < len(seqres):
            if seqres[idx] != rec.res_three.upper():
                return None
            n_checked += 1
    if n_checked < 5:
        return None
    return int(best_offset)


# ---------------------------------------------------------------------- #
# Context builder                                                        #
# ---------------------------------------------------------------------- #

def build_fragment_context(
    task: TaskRow,
    *,
    pdb_dir: str,
    flank_size: int = 1,
) -> FragmentContext:
    """Build a FragmentContext from one TaskRow + the PDB it references.

    Parameters
    ----------
    task : TaskRow
    pdb_dir : str — directory holding `task.ref_pdb`
    flank_size : int — residues to take on each side of the fragment.
        Default 1 (the minimum the encoder consumes; see module docstring).
        Pass 0 to omit flanks; the encoder will then fall back to native
        first/last bond direction (information leak).

    Returns
    -------
    FragmentContext ready for EncoderInputs.from_fragment_context(...).
    """
    if flank_size < 0:
        raise ValueError(f"flank_size must be >= 0, got {flank_size}")
    if task.start_resi > task.end_resi:
        raise ValueError(
            f"{task.case_id}: start_resi ({task.start_resi}) > "
            f"end_resi ({task.end_resi})"
        )

    pdb_path = os.path.join(pdb_dir, task.ref_pdb)
    ca_records = parse_pdb_ca(pdb_path, task.chain_id)
    if not ca_records:
        raise ValueError(
            f"{task.case_id}: no CA atoms found on chain {task.chain_id!r} "
            f"in {pdb_path}"
        )

    # --- fragment range --------------------------------------------------
    frag_xyz: List[np.ndarray] = []
    seq_chars: List[str] = []
    missing: List[int] = []
    for r in range(task.start_resi, task.end_resi + 1):
        rec = ca_records.get(r)
        if rec is None:
            missing.append(r)
            continue
        if rec.icode != " ":
            raise ValueError(
                f"{task.case_id}: residue {r} has insertion code "
                f"{rec.icode!r} (unsupported)"
            )
        frag_xyz.append(rec.xyz)
        seq_chars.append(_three_to_one(
            rec.res_three,
            where=f"{task.case_id} resi {r}",
        ))
    if missing:
        raise ValueError(
            f"{task.case_id}: missing CA at residues {missing} in "
            f"{pdb_path} (chain {task.chain_id})"
        )

    fragment_ca_ref = np.stack(frag_xyz, axis=0).astype(np.float64, copy=False)
    sequence = "".join(seq_chars)

    if fragment_ca_ref.shape[0] != task.n_residues_expected:
        # Should be unreachable given the missing-check above; surfaced as
        # a defensive invariant rather than swallowed.
        raise AssertionError(
            f"{task.case_id}: built fragment length "
            f"{fragment_ca_ref.shape[0]} != expected "
            f"{task.n_residues_expected}"
        )

    # --- flanks ----------------------------------------------------------
    flank_before_xyz: List[np.ndarray] = []
    flank_before_resi: List[int] = []
    for r in range(task.start_resi - flank_size, task.start_resi):
        rec = ca_records.get(r)
        if rec is None:
            # Truncated flank (e.g. fragment starts at residue 3 with
            # flank_size=2 but residue 1 has no CA). We surface this by
            # returning what we have; the encoder's native-direction
            # fallback covers the empty case. No silent padding.
            continue
        if rec.icode != " ":
            continue
        flank_before_xyz.append(rec.xyz)
        flank_before_resi.append(r)

    flank_after_xyz: List[np.ndarray] = []
    flank_after_resi: List[int] = []
    for r in range(task.end_resi + 1, task.end_resi + 1 + flank_size):
        rec = ca_records.get(r)
        if rec is None:
            continue
        if rec.icode != " ":
            continue
        flank_after_xyz.append(rec.xyz)
        flank_after_resi.append(r)

    flank_before_ca = (
        np.stack(flank_before_xyz, axis=0).astype(np.float64, copy=False)
        if flank_before_xyz else np.zeros((0, 3), dtype=np.float64)
    )
    flank_after_ca = (
        np.stack(flank_after_xyz, axis=0).astype(np.float64, copy=False)
        if flank_after_xyz else np.zeros((0, 3), dtype=np.float64)
    )

    return FragmentContext(
        fragment_ca_ref=fragment_ca_ref,
        flank_before_ca=flank_before_ca,
        flank_after_ca=flank_after_ca,
        sequence=sequence,
        case_id=task.case_id,
        ref_pdb=task.ref_pdb,
        chain_id=task.chain_id,
        start_resi=task.start_resi,
        end_resi=task.end_resi,
        ligand_resname=task.ligand_resname,
        scale_factor=task.scale_factor,
        scale_mode=task.scale_mode,
        has_native_ref=task.has_native_ref,
        flank_before_resi=tuple(flank_before_resi),
        flank_after_resi=tuple(flank_after_resi),
    )


# ---------------------------------------------------------------------- #
# Lenient context builder (no-native / partial reference)                #
# ---------------------------------------------------------------------- #

@dataclass(frozen=True)
class LenientBuildReport:
    """Side-band record of WHY the lenient builder was used.

    Returned alongside FragmentContext so the loader can populate
    KrasTask.metadata with native_reference_status and
    missing_native_residues.
    """
    case_id: str
    missing_native_residues: Tuple[int, ...]
    interior_missing: Tuple[int, ...]
    endpoint_missing: Tuple[int, ...]
    sequence_source: str  # "atoms_only" | "atoms+seqres" | "seqres_only"
    seqres_offset: Optional[int]
    anchor_left_present: bool
    anchor_right_present: bool


def build_fragment_context_lenient(
    task: TaskRow,
    *,
    pdb_dir: str,
    flank_size: int = 1,
) -> Tuple[FragmentContext, LenientBuildReport]:
    """Lenient counterpart of `build_fragment_context()`.

    Tolerates missing INTERIOR CA atoms (e.g. disordered loops in a
    crystal structure) by:
      - Filling fragment_ca_ref interior positions with NaN
      - Recovering missing residue identity from the chain's SEQRES
        record (calibrated to ATOM resSeq via observed CA records)
      - Setting has_native_ref=False on the returned FragmentContext

    Hard requirements (still raise ValueError if violated):
      - Anchor positions (start_resi, end_resi) MUST have CA atoms.
        Without anchors the encoder cannot construct EncoderInputs,
        so sampling is impossible.
      - Sequence at every fragment position must be resolvable from
        either the ATOM CA records or from SEQRES.

    The returned FragmentContext is shape-compatible with the strict
    one — the caller can pass it to EncoderInputs.from_fragment_context
    unchanged. RMSD evaluation on this context will produce NaN at
    interior positions, which downstream code MUST detect via
    `has_native_ref=False` (do not compute RMSD on this context).
    """
    if flank_size < 0:
        raise ValueError(f"flank_size must be >= 0, got {flank_size}")
    if task.start_resi > task.end_resi:
        raise ValueError(
            f"{task.case_id}: start_resi ({task.start_resi}) > "
            f"end_resi ({task.end_resi})"
        )

    pdb_path = os.path.join(pdb_dir, task.ref_pdb)
    ca_records = parse_pdb_ca(pdb_path, task.chain_id)
    if not ca_records:
        raise ValueError(
            f"{task.case_id}: no CA atoms found on chain {task.chain_id!r} "
            f"in {pdb_path}"
        )

    n_residues = task.n_residues_expected

    # --- anchors are non-negotiable -------------------------------------
    anchor_left_rec = ca_records.get(task.start_resi)
    anchor_right_rec = ca_records.get(task.end_resi)
    if anchor_left_rec is None:
        raise ValueError(
            f"{task.case_id}: anchor_left CA at residue "
            f"{task.start_resi} missing in {pdb_path} (chain "
            f"{task.chain_id}); cannot construct EncoderInputs"
        )
    if anchor_right_rec is None:
        raise ValueError(
            f"{task.case_id}: anchor_right CA at residue "
            f"{task.end_resi} missing in {pdb_path} (chain "
            f"{task.chain_id}); cannot construct EncoderInputs"
        )
    if anchor_left_rec.icode != " ":
        raise ValueError(
            f"{task.case_id}: anchor_left has insertion code "
            f"{anchor_left_rec.icode!r} (unsupported)"
        )
    if anchor_right_rec.icode != " ":
        raise ValueError(
            f"{task.case_id}: anchor_right has insertion code "
            f"{anchor_right_rec.icode!r} (unsupported)"
        )

    # --- SEQRES calibration (used to fill missing interior identities) -
    seqres = parse_pdb_seqres(pdb_path, task.chain_id)
    seqres_offset = _calibrate_seqres_offset(seqres, ca_records) \
        if seqres else None

    def seqres_lookup(resi: int) -> Optional[str]:
        if seqres and seqres_offset is not None:
            idx = resi - seqres_offset - 1  # 1-indexed -> 0-indexed
            if 0 <= idx < len(seqres):
                return seqres[idx]
        return None

    # --- fragment range with NaN-padding for missing CAs -----------------
    NAN3 = np.full(3, np.nan, dtype=np.float64)
    frag_xyz: List[np.ndarray] = []
    seq_chars: List[str] = []
    missing_resi: List[int] = []
    interior_missing: List[int] = []
    used_seqres = False
    for r in range(task.start_resi, task.end_resi + 1):
        rec = ca_records.get(r)
        if rec is None:
            missing_resi.append(r)
            if r != task.start_resi and r != task.end_resi:
                interior_missing.append(r)
            three = seqres_lookup(r)
            if three is None:
                raise ValueError(
                    f"{task.case_id}: residue {r} has no CA in "
                    f"{pdb_path} and SEQRES has no resolvable identity "
                    f"for it (offset_calibrated="
                    f"{seqres_offset is not None}); "
                    "cannot construct sequence"
                )
            try:
                aa = _three_to_one(three, where=f"{task.case_id} resi {r} (seqres)")
            except ValueError as exc:
                raise ValueError(
                    f"{task.case_id}: SEQRES residue {three!r} at "
                    f"position {r} cannot be mapped to one-letter AA: "
                    f"{exc}"
                ) from exc
            seq_chars.append(aa)
            frag_xyz.append(NAN3.copy())
            used_seqres = True
            continue
        if rec.icode != " ":
            raise ValueError(
                f"{task.case_id}: residue {r} has insertion code "
                f"{rec.icode!r} (unsupported)"
            )
        frag_xyz.append(rec.xyz)
        seq_chars.append(_three_to_one(
            rec.res_three, where=f"{task.case_id} resi {r}",
        ))

    fragment_ca_ref = np.stack(frag_xyz, axis=0).astype(np.float64, copy=False)
    sequence = "".join(seq_chars)
    if fragment_ca_ref.shape[0] != n_residues or len(sequence) != n_residues:
        raise AssertionError(
            f"{task.case_id}: lenient builder produced length "
            f"{fragment_ca_ref.shape[0]} (seq={len(sequence)}) "
            f"!= expected {n_residues}"
        )

    # --- flanks (same logic as strict builder; no NaN padding here) -----
    flank_before_xyz: List[np.ndarray] = []
    flank_before_resi: List[int] = []
    for r in range(task.start_resi - flank_size, task.start_resi):
        rec = ca_records.get(r)
        if rec is None or rec.icode != " ":
            continue
        flank_before_xyz.append(rec.xyz)
        flank_before_resi.append(r)

    flank_after_xyz: List[np.ndarray] = []
    flank_after_resi: List[int] = []
    for r in range(task.end_resi + 1, task.end_resi + 1 + flank_size):
        rec = ca_records.get(r)
        if rec is None or rec.icode != " ":
            continue
        flank_after_xyz.append(rec.xyz)
        flank_after_resi.append(r)

    flank_before_ca = (
        np.stack(flank_before_xyz, axis=0).astype(np.float64, copy=False)
        if flank_before_xyz else np.zeros((0, 3), dtype=np.float64)
    )
    flank_after_ca = (
        np.stack(flank_after_xyz, axis=0).astype(np.float64, copy=False)
        if flank_after_xyz else np.zeros((0, 3), dtype=np.float64)
    )

    ctx = FragmentContext(
        fragment_ca_ref=fragment_ca_ref,
        flank_before_ca=flank_before_ca,
        flank_after_ca=flank_after_ca,
        sequence=sequence,
        case_id=task.case_id,
        ref_pdb=task.ref_pdb,
        chain_id=task.chain_id,
        start_resi=task.start_resi,
        end_resi=task.end_resi,
        ligand_resname=task.ligand_resname,
        scale_factor=task.scale_factor,
        scale_mode=task.scale_mode,
        # Lenient builder ALWAYS reports has_native_ref=False because
        # interior NaNs make the reference unusable for RMSD.
        has_native_ref=False,
        flank_before_resi=tuple(flank_before_resi),
        flank_after_resi=tuple(flank_after_resi),
    )
    if used_seqres and any(c == c for c in fragment_ca_ref.flat):  # mixed
        seq_source = "atoms+seqres"
    elif used_seqres:
        seq_source = "seqres_only"
    else:
        seq_source = "atoms_only"
    report = LenientBuildReport(
        case_id=task.case_id,
        missing_native_residues=tuple(missing_resi),
        interior_missing=tuple(interior_missing),
        endpoint_missing=tuple(
            r for r in missing_resi
            if r in (task.start_resi, task.end_resi)
        ),
        sequence_source=seq_source,
        seqres_offset=seqres_offset,
        anchor_left_present=True,
        anchor_right_present=True,
    )
    return ctx, report


def load_fragment_contexts(
    csv_path: str,
    *,
    pdb_dir: str,
    flank_size: int = 1,
    skip_missing_pdb: bool = False,
) -> List[FragmentContext]:
    """Load all rows from `csv_path` and build FragmentContexts.

    Parameters
    ----------
    csv_path : str — tasks CSV
    pdb_dir : str — directory of reference PDBs
    flank_size : int — see build_fragment_context
    skip_missing_pdb : bool — if True, rows whose PDB file is missing are
        skipped with a printed warning. If False (default), the first
        missing PDB raises. We do NOT default to skipping because that
        masks data-availability bugs.
    """
    tasks = load_tasks_csv(csv_path)
    out: List[FragmentContext] = []
    for t in tasks:
        pdb_path = os.path.join(pdb_dir, t.ref_pdb)
        if not os.path.isfile(pdb_path):
            if skip_missing_pdb:
                print(f"[data_loader] skipping {t.case_id}: missing {pdb_path}")
                continue
            raise FileNotFoundError(
                f"{t.case_id}: PDB not found at {pdb_path}"
            )
        out.append(build_fragment_context(
            t, pdb_dir=pdb_dir, flank_size=flank_size,
        ))
    return out


# ---------------------------------------------------------------------- #
# CLI / smoke entrypoint                                                 #
# ---------------------------------------------------------------------- #

def _demo() -> None:
    """End-to-end demo: CSV + PDBs → FragmentContext → EncoderInputs.

    Run from project root:
        python ras_folding/utils/data_loader.py
    """
    import sys

    # Allow running as a script without installing the package.
    _here = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(os.path.dirname(_here))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

    from ras_folding.encoder import EncoderInputs

    csv_path = os.path.join(_project_root, "inputs", "kras_tasks.csv")
    pdb_dir = os.path.join(_project_root, "kras_select_systems")

    print(f"[data_loader] CSV : {csv_path}")
    print(f"[data_loader] PDBs: {pdb_dir}")
    print()

    contexts = load_fragment_contexts(csv_path, pdb_dir=pdb_dir, flank_size=1)
    print(f"loaded {len(contexts)} fragment contexts.")
    print()

    for ctx in contexts:
        inputs = EncoderInputs.from_fragment_context(ctx)
        anchor_dist = float(np.linalg.norm(
            inputs.anchor_right - inputs.anchor_left
        ))
        print(
            f"  {ctx.case_id:24s} "
            f"chain={ctx.chain_id} "
            f"resi=[{ctx.start_resi:>3d},{ctx.end_resi:>3d}] "
            f"n={ctx.n_residues:2d} "
            f"seq={ctx.sequence} "
            f"flanks=({len(ctx.flank_before_resi)},{len(ctx.flank_after_resi)}) "
            f"||A_R-A_L||={anchor_dist:7.3f}"
        )


if __name__ == "__main__":
    _demo()


__all__ = [
    "TaskRow",
    "FragmentContext",
    "LenientBuildReport",
    "load_tasks_csv",
    "parse_pdb_ca",
    "parse_pdb_seqres",
    "build_fragment_context",
    "build_fragment_context_lenient",
    "load_fragment_contexts",
]
