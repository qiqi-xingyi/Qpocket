# Author: Yuqi Zhang
"""End-to-end PULCHRA → embed pipeline."""
from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from ras_folding.reconstruct.embed import (
    embed_rebuilt_fragment_into_reference,
)
from ras_folding.reconstruct.io import compute_ca_drift
from ras_folding.reconstruct.pulchra_adapter import PulchraAdapter
from ras_folding.reconstruct.types import (
    ReconstructionInput,
    ReconstructionResult,
)


class FullAtomReconstructionPipeline:
    def __init__(
        self,
        pulchra_adapter: Optional[PulchraAdapter] = None,
        remove_hetero: bool = True,
        remove_waters: bool = True,
        ca_drift_warn_threshold_a: float = 0.5,
        pulchra_overwrite: bool = True,
    ) -> None:
        self.pulchra = pulchra_adapter or PulchraAdapter()
        self.remove_hetero = bool(remove_hetero)
        self.remove_waters = bool(remove_waters)
        self.ca_drift_warn_threshold_a = float(ca_drift_warn_threshold_a)
        # Whether the PULCHRA wrapper should delete a stale
        # <stem>.rebuilt.pdb in output_dir before invocation. Default
        # True so reuse-existing / --overwrite reruns never serve up
        # the previous run's rebuild by mistake.
        self.pulchra_overwrite = bool(pulchra_overwrite)

    # ------------------------------------------------------------------ #
    def reconstruct(
        self,
        inp: ReconstructionInput,
        output_dir: Path,
        ligand_resname: Optional[str] = None,
    ) -> ReconstructionResult:
        out_dir = Path(output_dir) / "reconstruct"
        out_dir.mkdir(parents=True, exist_ok=True)
        # Stale-state hygiene: if a previous failed run left an
        # ERROR.txt or reconstruction_summary.json behind, remove them
        # *before* this attempt so the on-disk state never reflects two
        # different runs at once. Actual PDB outputs from PULCHRA are
        # only produced by this run, so we don't need to touch them
        # eagerly (purge_stage in reuse-existing already wiped the dir).
        for stale in ("ERROR.txt", "reconstruction_summary.json"):
            p = out_dir / stale
            if p.exists():
                try:
                    p.unlink()
                except OSError:
                    pass
        # Initialize ca_drift keys up-front so the summary always
        # carries the canonical fields, even when PULCHRA / embed fail
        # before drift is computed. The validator then never has to
        # tolerate "ca_drift fields missing".
        meta: Dict[str, Any] = {
            "task_id": inp.task_id,
            "predicted_ca_pdb": str(inp.predicted_ca_pdb),
            "reference_pdb": str(inp.reference_pdb),
            "chain_id": inp.chain_id,
            "start_resi": int(inp.start_resi),
            "end_resi": int(inp.end_resi),
            "sequence": inp.sequence,
            "ligand_resname": ligand_resname,
            "ca_drift_warn_threshold_a": float(self.ca_drift_warn_threshold_a),
            "ca_drift_rmsd": None,
            "ca_drift_max": None,
            "ca_drift_n_ca_original": None,
            "ca_drift_n_ca_rebuilt": None,
            "ca_drift_shape_mismatch": None,
            "ca_drift_warning": None,
            "pulchra_bin": None,
        }

        # 1) PULCHRA rebuild
        meta["pulchra_overwrite"] = self.pulchra_overwrite
        meta["pulchra_stdout_path"] = str(out_dir / "pulchra.stdout.txt")
        meta["pulchra_stderr_path"] = str(out_dir / "pulchra.stderr.txt")
        try:
            rebuilt = self.pulchra.rebuild(
                ca_pdb=Path(inp.predicted_ca_pdb),
                output_dir=out_dir,
                sequence=inp.sequence,
                overwrite=self.pulchra_overwrite,
            )
        except Exception as e:
            err = f"pulchra_failed: {e!r}"
            (out_dir / "ERROR.txt").write_text(err)
            res = ReconstructionResult(
                task_id=inp.task_id,
                predicted_ca_pdb=Path(inp.predicted_ca_pdb),
                rebuilt_fragment_pdb=None,
                embedded_receptor_pdb=None,
                status="failed",
                error=err,
                metadata=meta,
            )
            self._write_summary(out_dir, res)
            return res
        # Record the actual PULCHRA binary path that was used so
        # reconstruction_summary downstream readers (validator,
        # post-mortem reports) can audit which build of PULCHRA produced
        # this output.
        meta["pulchra_bin"] = getattr(self.pulchra, "_resolved", None)
        # Original PULCHRA output (before we copy it to the canonical
        # `rebuilt_fragment.pdb`). Keeping both paths in metadata makes
        # debugging "which file did PULCHRA actually emit" trivial.
        meta["pulchra_output_pdb"] = str(rebuilt)

        # canonicalize the rebuilt fragment path inside out_dir
        rebuilt_local = out_dir / "rebuilt_fragment.pdb"
        if Path(rebuilt).resolve() != rebuilt_local.resolve():
            shutil.copyfile(rebuilt, rebuilt_local)
        meta["rebuilt_fragment_pdb"] = str(rebuilt_local)

        # 1b) CA drift check (no Kabsch — encoder anchor frame is fixed).
        # Default behaviour: warn-only. Embed continues even when drift
        # exceeds threshold or CA counts mismatch; the embed step has its
        # own count check that will raise loudly if needed.
        try:
            drift = compute_ca_drift(
                original_ca_pdb=Path(inp.predicted_ca_pdb),
                rebuilt_fragment_pdb=rebuilt_local,
            )
        except Exception as e:
            drift = {
                "ca_drift_rmsd": None,
                "ca_drift_max": None,
                "ca_drift_n_ca_original": None,
                "ca_drift_n_ca_rebuilt": None,
                "ca_drift_shape_mismatch": None,
                "ca_drift_warning": f"ca_drift_compute_failed: {e!r}",
            }
        # Promote to "exceeds threshold" warning when applicable. Don't
        # overwrite the count-mismatch warning emitted by compute_ca_drift.
        if (
            drift.get("ca_drift_max") is not None
            and drift.get("ca_drift_warning") is None
            and drift["ca_drift_max"] > self.ca_drift_warn_threshold_a
        ):
            drift["ca_drift_warning"] = "pulchra_ca_drift_exceeds_threshold"
        meta.update(drift)

        # 2) embed
        embedded_path = out_dir / "embedded_receptor.pdb"
        try:
            embed_rebuilt_fragment_into_reference(
                reference_pdb=Path(inp.reference_pdb),
                rebuilt_fragment_pdb=rebuilt_local,
                chain_id=inp.chain_id,
                start_resi=int(inp.start_resi),
                end_resi=int(inp.end_resi),
                output_pdb=embedded_path,
                remove_hetero=self.remove_hetero,
                remove_waters=self.remove_waters,
                ligand_resname=ligand_resname,
            )
        except Exception as e:
            err = f"embed_failed: {e!r}"
            (out_dir / "ERROR.txt").write_text(err)
            res = ReconstructionResult(
                task_id=inp.task_id,
                predicted_ca_pdb=Path(inp.predicted_ca_pdb),
                rebuilt_fragment_pdb=rebuilt_local,
                embedded_receptor_pdb=None,
                status="failed",
                error=err,
                metadata=meta,
            )
            self._write_summary(out_dir, res)
            return res

        meta["embedded_receptor_pdb"] = str(embedded_path)
        res = ReconstructionResult(
            task_id=inp.task_id,
            predicted_ca_pdb=Path(inp.predicted_ca_pdb),
            rebuilt_fragment_pdb=rebuilt_local,
            embedded_receptor_pdb=embedded_path,
            status="done",
            error=None,
            metadata=meta,
        )
        self._write_summary(out_dir, res)
        return res

    # ------------------------------------------------------------------ #
    @staticmethod
    def _write_summary(
        out_dir: Path, res: ReconstructionResult,
    ) -> None:
        meta = dict(res.metadata or {})
        # Top-level mirror of the ca_drift fields makes downstream
        # readers (validator, post-hoc analysis) robust to either
        # location convention. This is additive — metadata still
        # carries the canonical copy.
        ca_drift_keys = (
            "ca_drift_rmsd",
            "ca_drift_max",
            "ca_drift_n_ca_original",
            "ca_drift_n_ca_rebuilt",
            "ca_drift_shape_mismatch",
            "ca_drift_warning",
            "ca_drift_warn_threshold_a",
        )
        payload: Dict[str, Any] = {
            "task_id": res.task_id,
            "status": res.status,
            "error": res.error,
            "predicted_ca_pdb": str(res.predicted_ca_pdb),
            "rebuilt_fragment_pdb": (
                None if res.rebuilt_fragment_pdb is None
                else str(res.rebuilt_fragment_pdb)
            ),
            "embedded_receptor_pdb": (
                None if res.embedded_receptor_pdb is None
                else str(res.embedded_receptor_pdb)
            ),
            "metadata": meta,
        }
        for k in ca_drift_keys:
            if k in meta:
                payload[k] = meta[k]
        (out_dir / "reconstruction_summary.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8",
        )


__all__ = ["FullAtomReconstructionPipeline"]
