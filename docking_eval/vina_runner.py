# Author: Yuqi Zhang
"""AutoDock Vina runner — single docking call, parses best affinity."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


# Patterns matching Vina's log lines:
#   "   1     -7.500       0.000      0.000"
# and the PDBQT REMARK form:
#   "REMARK VINA RESULT:    -7.500      0.000      0.000"
_LOG_TABLE_RE = re.compile(
    r"^\s*1\s+(-?\d+\.\d+)\s+\S+\s+\S+",
    re.MULTILINE,
)
_REMARK_RE = re.compile(
    r"REMARK\s+VINA\s+RESULT:\s*(-?\d+\.\d+)",
    re.IGNORECASE,
)


class VinaRunner:
    def __init__(
        self,
        vina_bin: Optional[Union[str, Path]] = None,
        timeout_sec: int = 600,
    ) -> None:
        self.vina_bin = (None if vina_bin is None else str(vina_bin))
        self.timeout_sec = int(timeout_sec)
        self._resolved: Optional[str] = None

    # ------------------------------------------------------------------ #
    def _resolve_bin(self) -> str:
        if self._resolved:
            return self._resolved
        candidates: List[Optional[str]] = [
            self.vina_bin,
            os.environ.get("VINA_BIN"),
            shutil.which("vina"),
        ]
        for c in candidates:
            if c and Path(c).is_file() and os.access(c, os.X_OK):
                self._resolved = c
                return c
        raise RuntimeError(
            "AutoDock Vina binary 'vina' not found. Install Vina and "
            "put it on PATH, or set VINA_BIN=/full/path/to/vina."
        )

    def check_available(self) -> bool:
        try:
            self._resolve_bin()
            return True
        except RuntimeError:
            return False

    # ------------------------------------------------------------------ #
    def run_once(
        self,
        receptor_pdbqt: Path,
        ligand_pdbqt: Path,
        box: Dict[str, Any],
        output_pdbqt: Path,
        log_path: Path,
        exhaustiveness: int = 8,
        num_modes: int = 9,
        seed: int = 2024,
    ) -> float:
        """Run a single Vina call. Returns the BEST (mode 1) affinity in
        kcal/mol. Raises RuntimeError on failure or unparseable output.
        """
        bin_ = self._resolve_bin()
        receptor_pdbqt = Path(receptor_pdbqt)
        ligand_pdbqt = Path(ligand_pdbqt)
        output_pdbqt = Path(output_pdbqt)
        log_path = Path(log_path)
        output_pdbqt.parent.mkdir(parents=True, exist_ok=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            bin_,
            "--receptor", str(receptor_pdbqt),
            "--ligand", str(ligand_pdbqt),
            "--center_x", f"{float(box['center_x']):.3f}",
            "--center_y", f"{float(box['center_y']):.3f}",
            "--center_z", f"{float(box['center_z']):.3f}",
            "--size_x", f"{float(box['size_x']):.3f}",
            "--size_y", f"{float(box['size_y']):.3f}",
            "--size_z", f"{float(box['size_z']):.3f}",
            "--exhaustiveness", str(int(exhaustiveness)),
            "--num_modes", str(int(num_modes)),
            "--seed", str(int(seed)),
            "--out", str(output_pdbqt),
        ]
        # `--log` was removed in Vina >= 1.2; we capture stdout instead.
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.timeout_sec,
        )
        log_path.write_text(
            f"$ {' '.join(cmd)}\n\n"
            f"--- stdout ---\n{proc.stdout or ''}\n"
            f"--- stderr ---\n{proc.stderr or ''}\n",
            encoding="utf-8",
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Vina exited with code {proc.returncode}; "
                f"see {log_path} for details."
            )

        # Parse best affinity. Try stdout first; then output PDBQT REMARK.
        aff = _parse_affinity(proc.stdout or "")
        if aff is None and output_pdbqt.is_file():
            aff = _parse_affinity(output_pdbqt.read_text(encoding="utf-8"))
        if aff is None:
            raise RuntimeError(
                "Could not parse Vina best affinity from log/output. "
                f"See {log_path}."
            )
        return float(aff)


def _parse_affinity(text: str) -> Optional[float]:
    """Parse mode-1 affinity in kcal/mol from Vina log/output text."""
    m = _LOG_TABLE_RE.search(text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    m = _REMARK_RE.search(text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


__all__ = ["VinaRunner", "_parse_affinity"]
