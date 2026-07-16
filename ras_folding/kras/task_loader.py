# Author: Yuqi Zhang
"""KRAS task loader — adapts inputs/kras_tasks.csv → list[KrasTask].

Schema auto-detection
---------------------
The project's canonical CSV
(``inputs/kras_tasks.csv``) has columns:

    case_id, ref_pdb, chain_id, start_resi, end_resi,
    ligand_resname, scale_factor, scale_mode, has_native_ref

There is NO ``sequence`` column — sequence is derived from the PDB at
``ref_pdb`` over residues [start_resi, end_resi] via
``ras_folding.utils.data_loader.load_fragment_contexts``.

The loader is tolerant of column-name aliases: ``case_id`` /
``task_id`` / ``name`` / ``id`` are all accepted as the task identifier.
Any unrecognised columns are preserved verbatim in
``KrasTask.metadata`` so downstream tooling (reports, summaries) can
surface them.

Output
------
``load_kras_tasks(csv_path, *, pdb_dir=..., flank_size=1)`` returns a
list[KrasTask] PLUS a side-effect: it builds a ``schema_summary`` dict
the caller can read via ``last_schema_summary()``. Callers that want
only the schema (e.g., the global report writer) can call
``inspect_csv_schema(csv_path)``.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ras_folding.encoder.inputs import EncoderInputs
from ras_folding.utils.data_loader import (
    FragmentContext,
    LenientBuildReport,
    build_fragment_context,
    build_fragment_context_lenient,
    load_fragment_contexts,
    load_tasks_csv,
)


_DEFAULT_PDB_DIR_NAMES = ("kras_select_systems", "kras_systems", "pdb")

_TASK_ID_ALIASES = ("task_id", "case_id", "name", "id")


@dataclass
class KrasTask:
    """A KRAS pocket conformational-sampling task ready for the full-batch runner."""
    task_id: str
    sequence: str
    encoder_inputs: EncoderInputs
    reference_coords: Optional[np.ndarray]
    reference_pdb: Optional[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------- #
# schema inspection                                                      #
# ---------------------------------------------------------------------- #

def inspect_csv_schema(csv_path: Path) -> Dict[str, Any]:
    """Read header row and a few sample values; return a schema summary.

    Never raises on a row that fails to parse — only the header is
    inspected here. Used by the runner before it commits to loading the
    full task list.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"tasks CSV not found: {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    n_rows = len(rows)

    detected = {
        "task_id_column": _resolve_alias(fieldnames, _TASK_ID_ALIASES),
        "ref_pdb_column": _resolve_alias(
            fieldnames, ("ref_pdb", "reference_pdb", "pdb"),
        ),
        "chain_id_column": _resolve_alias(fieldnames, ("chain_id", "chain")),
        "start_resi_column": _resolve_alias(
            fieldnames, ("start_resi", "start", "start_residue"),
        ),
        "end_resi_column": _resolve_alias(
            fieldnames, ("end_resi", "end", "end_residue"),
        ),
        "sequence_column": _resolve_alias(
            fieldnames, ("sequence", "seq", "fragment_sequence"),
        ),
        "ligand_column": _resolve_alias(
            fieldnames, ("ligand_resname", "ligand"),
        ),
        "scale_factor_column": _resolve_alias(
            fieldnames, ("scale_factor",),
        ),
        "scale_mode_column": _resolve_alias(fieldnames, ("scale_mode",)),
        "has_native_ref_column": _resolve_alias(
            fieldnames, ("has_native_ref",),
        ),
    }
    summary: Dict[str, Any] = {
        "csv_path": str(csv_path),
        "n_rows": n_rows,
        "header": fieldnames,
        "detected_columns": detected,
        "missing_optional_columns": [
            k for k, v in detected.items() if v is None
        ],
    }
    return summary


def _resolve_alias(
    header: List[str], aliases: Tuple[str, ...],
) -> Optional[str]:
    """Return the first column name in `header` matching any alias (case-insensitive)."""
    h_lower = {c.lower(): c for c in header}
    for a in aliases:
        if a.lower() in h_lower:
            return h_lower[a.lower()]
    return None


# ---------------------------------------------------------------------- #
# loader                                                                 #
# ---------------------------------------------------------------------- #

def _resolve_pdb_dir(
    csv_path: Path,
    pdb_dir: Optional[Path],
) -> Path:
    """Locate the PDB directory containing reference structures.

    Resolution order:
      1. Explicit `pdb_dir` argument (if provided).
      2. Any of the conventional sister directories at the project root.
      3. fallback: csv_path.parent.parent / "kras_select_systems".
    """
    if pdb_dir is not None:
        p = Path(pdb_dir)
        if not p.is_dir():
            raise FileNotFoundError(f"pdb_dir does not exist: {p}")
        return p
    project_root = csv_path.parent.parent
    for name in _DEFAULT_PDB_DIR_NAMES:
        cand = project_root / name
        if cand.is_dir():
            return cand
    raise FileNotFoundError(
        f"could not locate a KRAS PDB directory under {project_root}; "
        f"pass pdb_dir explicitly"
    )


def load_kras_tasks(
    csv_path: Path,
    *,
    pdb_dir: Optional[Path] = None,
    flank_size: int = 1,
    skip_missing_pdb: bool = True,
) -> Tuple[List[KrasTask], Dict[str, Any]]:
    """Load tasks from the CSV.

    Returns
    -------
    (tasks, schema_summary) — schema_summary is the inspect_csv_schema
    output extended with load-time statistics (n_loaded, n_skipped,
    n_with_reference, sequence_length_distribution).

    Tasks whose PDB cannot be opened are skipped (logged in schema_summary)
    when ``skip_missing_pdb=True`` (default).
    """
    csv_path = Path(csv_path)
    schema = inspect_csv_schema(csv_path)
    pdb_dir_resolved = _resolve_pdb_dir(csv_path, pdb_dir)
    schema["pdb_dir"] = str(pdb_dir_resolved)

    # ALSO read raw CSV rows so that ALL columns (including new
    # mutation_group / ligand_family / pocket_module / analysis_role
    # and any future extras) survive into KrasTask.metadata. The typed
    # TaskRow loader below only consumes a fixed schema — extras are
    # otherwise discarded.
    #
    # Fail-loud on duplicate case_id (would otherwise silently overwrite
    # the dict entry and present as "loaded N-1 unique tasks").
    raw_csv_rows: Dict[str, Dict[str, str]] = {}
    raw_case_id_order: List[str] = []
    duplicate_case_ids: List[str] = []
    blank_case_id_rows: List[int] = []
    n_raw_rows = 0
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row_idx, raw in enumerate(reader):
            n_raw_rows += 1
            cid = (raw.get("case_id") or raw.get("task_id")
                   or raw.get("name") or raw.get("id") or "").strip()
            if not cid:
                blank_case_id_rows.append(row_idx)
                continue
            if cid in raw_csv_rows:
                duplicate_case_ids.append(cid)
                continue
            raw_csv_rows[cid] = {
                k: (v if v is not None else "") for k, v in raw.items()
            }
            raw_case_id_order.append(cid)
    schema["n_raw_rows"] = n_raw_rows
    schema["raw_case_id_order"] = list(raw_case_id_order)
    if duplicate_case_ids:
        schema["duplicate_case_ids"] = list(duplicate_case_ids)
        raise ValueError(
            "duplicate case_id detected in tasks CSV "
            f"({csv_path}): {sorted(set(duplicate_case_ids))}. "
            "Resolve duplicates before running."
        )
    if blank_case_id_rows:
        schema["blank_case_id_rows"] = list(blank_case_id_rows)

    # Load TaskRow list first; then build FragmentContext per row inside
    # try/except so a single missing-CA row does NOT lose the rest of the
    # batch. (`load_fragment_contexts` raises on the first failure.)
    #
    # Two failure classes exist:
    #   (a) HARD-DROP — PDB missing, anchors missing, sequence cannot be
    #       built. The row cannot run sampling. -> `skipped`.
    #   (b) NO-NATIVE — interior CA atoms missing but anchors+sequence
    #       still resolvable (typical: disordered Switch-II loops in
    #       inactive-state KRAS structures). The row can still run
    #       sampling/landscape/docking — only RMSD evaluation is
    #       impossible. -> use lenient builder; mark in schema.
    raw_rows = load_tasks_csv(str(csv_path))
    contexts: List[FragmentContext] = []
    skipped: List[Dict[str, Any]] = []
    no_native_reports: Dict[str, Dict[str, Any]] = {}
    for row in raw_rows:
        pdb_path = os.path.join(str(pdb_dir_resolved), row.ref_pdb)
        if not os.path.isfile(pdb_path):
            if skip_missing_pdb:
                skipped.append({
                    "case_id": row.case_id,
                    "reason": f"PDB not found: {pdb_path}",
                    "drop_class": "hard_drop",
                })
                continue
            raise FileNotFoundError(
                f"{row.case_id}: PDB not found at {pdb_path}"
            )
        try:
            ctx = build_fragment_context(
                row, pdb_dir=str(pdb_dir_resolved), flank_size=flank_size,
            )
            contexts.append(ctx)
        except ValueError as strict_exc:
            # Strict builder raised — only some classes of failure are
            # recoverable via the lenient builder. The "missing CA at
            # residues" case is the canonical recoverable failure. We
            # try lenient unconditionally on ValueError; if the lenient
            # builder ALSO raises (e.g. anchor missing), that's a
            # genuine hard-drop.
            try:
                ctx, report = build_fragment_context_lenient(
                    row, pdb_dir=str(pdb_dir_resolved),
                    flank_size=flank_size,
                )
                contexts.append(ctx)
                no_native_reports[row.case_id] = {
                    "case_id": row.case_id,
                    "reason": (
                        "interior CA atoms missing in reference PDB; "
                        "lenient builder used (sequence recovered from "
                        "SEQRES, fragment_ca_ref padded with NaN); "
                        "RMSD-based metrics disabled for this task"
                    ),
                    "missing_native_residues": list(
                        report.missing_native_residues
                    ),
                    "interior_missing": list(report.interior_missing),
                    "endpoint_missing": list(report.endpoint_missing),
                    "sequence_source": report.sequence_source,
                    "seqres_offset": report.seqres_offset,
                    "strict_builder_error": repr(strict_exc),
                }
            except Exception as lenient_exc:
                skipped.append({
                    "case_id": row.case_id,
                    "reason": (
                        f"strict={strict_exc!r}; "
                        f"lenient={lenient_exc!r}"
                    ),
                    "drop_class": "hard_drop",
                })
        except Exception as e:
            skipped.append({
                "case_id": row.case_id,
                "reason": repr(e),
                "drop_class": "hard_drop",
            })
    if skipped:
        schema["rows_skipped_in_loader"] = skipped

    tasks: List[KrasTask] = []
    seq_lens: List[int] = []
    n_with_ref = 0

    for i, ctx in enumerate(contexts):
        try:
            inputs = EncoderInputs.from_fragment_context(ctx)
        except Exception as e:
            # Surface but skip — runner-level resume will retry next time.
            schema.setdefault("rows_skipped_in_loader", []).append({
                "case_id": getattr(ctx, "case_id", f"row_{i}"),
                "reason": repr(e),
                "drop_class": "hard_drop",
            })
            continue

        task_id = (
            getattr(ctx, "case_id", None)
            or f"kras_task_{i + 1:04d}"
        )

        ref_coords: Optional[np.ndarray] = None
        if ctx.has_native_ref and ctx.fragment_ca_ref is not None:
            ref_coords = np.asarray(ctx.fragment_ca_ref, dtype=np.float64)
            if ref_coords.shape == (ctx.n_residues, 3):
                # Final guard: even when has_native_ref says True, the
                # array must contain no NaN to be usable for RMSD.
                if not np.isfinite(ref_coords).all():
                    ref_coords = None
                else:
                    n_with_ref += 1
            else:
                ref_coords = None

        # Verbatim raw CSV row (preserves any extra/future columns).
        raw_row = raw_csv_rows.get(ctx.case_id, {}) or {}
        # Convenience surface fields for analysis grouping. Empty string
        # values are normalised to None so report writers can use truthy
        # checks (`if task.metadata["mutation_group"]`).
        def _opt(v):
            if v is None:
                return None
            s = str(v).strip()
            return s if s else None

        # Look up no-native report (set by lenient retry above).
        nn_report = no_native_reports.get(ctx.case_id)
        rmsd_available = bool(ref_coords is not None) and nn_report is None
        meta = {
            "case_id": ctx.case_id,
            "ref_pdb": ctx.ref_pdb,
            "chain_id": ctx.chain_id,
            "start_resi": ctx.start_resi,
            "end_resi": ctx.end_resi,
            "ligand_resname": ctx.ligand_resname,
            "scale_factor": ctx.scale_factor,
            "scale_mode": ctx.scale_mode,
            "has_native_ref": ctx.has_native_ref,
            "flank_before_resi": list(ctx.flank_before_resi),
            "flank_after_resi": list(ctx.flank_after_resi),
            # No-native / RMSD-availability surface (always present).
            "rmsd_available": rmsd_available,
            "oracle_anchor_available": rmsd_available,
            "native_reference_status": (
                "complete" if rmsd_available
                else (
                    "missing_ca_atoms" if nn_report is not None
                    else "csv_disabled"
                    if not ctx.has_native_ref
                    else "unknown"
                )
            ),
            "missing_native_residues": (
                list(nn_report["missing_native_residues"])
                if nn_report else []
            ),
            "loader_warning": (
                nn_report["strict_builder_error"]
                if nn_report else None
            ),
            # Top-level convenience surface for analysis grouping.
            "mutation_group": _opt(raw_row.get("mutation_group")),
            "ligand_family": _opt(raw_row.get("ligand_family")),
            "pocket_module": _opt(raw_row.get("pocket_module")),
            "analysis_role": _opt(raw_row.get("analysis_role")),
            # Verbatim raw CSV row — survives any extra/future columns
            # (custom analysis flags, additional grouping tags, etc.)
            "csv_row": dict(raw_row),
        }

        tasks.append(KrasTask(
            task_id=task_id,
            sequence=ctx.sequence,
            encoder_inputs=inputs,
            reference_coords=ref_coords,
            reference_pdb=ctx.ref_pdb,
            metadata=meta,
        ))
        seq_lens.append(int(ctx.n_residues))

    if seq_lens:
        seq_arr = np.asarray(seq_lens, dtype=np.int64)
        seq_dist = {
            "min": int(seq_arr.min()),
            "max": int(seq_arr.max()),
            "mean": float(seq_arr.mean()),
            "histogram": {
                int(k): int(v)
                for k, v in zip(*np.unique(seq_arr, return_counts=True))
            },
        }
    else:
        seq_dist = {}

    schema["n_loaded"] = len(tasks)
    schema["n_with_reference"] = n_with_ref
    schema["sequence_length_distribution"] = seq_dist
    # Always-present canonical keys for dropped vs no-native tasks.
    # `dropped_tasks` is HARD-DROP only (sampling impossible).
    # `no_native_reference_tasks` is loaded-but-RMSD-disabled.
    all_skipped = list(schema.get("rows_skipped_in_loader", []))
    hard = [s for s in all_skipped
            if s.get("drop_class", "hard_drop") == "hard_drop"]
    schema["dropped_tasks"] = hard
    schema["n_dropped"] = len(hard)
    schema["no_native_reference_tasks"] = list(no_native_reports.values())
    schema["n_no_native_reference"] = len(no_native_reports)
    # blank-case_id rows count toward "raw vs loaded" only via blank list
    return tasks, schema


__all__ = [
    "KrasTask",
    "load_kras_tasks",
    "inspect_csv_schema",
]
