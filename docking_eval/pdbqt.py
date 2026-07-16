# Author: Yuqi Zhang
"""Receptor / ligand PDBQT preparation via OpenBabel."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Union


def count_pdbqt_root_blocks(path: Path) -> int:
    """Count occurrences of the ``ROOT`` keyword in a PDBQT file.

    Vina expects exactly one ROOT...ENDROOT block per ligand file. If a
    ligand PDB had multiple residue copies merged into one file, obabel
    will emit one ROOT block per copy, and Vina will fail with a parsing
    error. This helper lets the pipeline surface that condition before
    invoking Vina.
    """
    n = 0
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.rstrip("\r\n")
            if stripped == "ROOT" or stripped.startswith("ROOT "):
                n += 1
    return n


def assert_single_root_pdbqt(path: Path) -> int:
    """Raise RuntimeError if ``path`` does not contain exactly one ROOT.

    Returns the ROOT count so callers can record it in metadata.
    """
    n = count_pdbqt_root_blocks(path)
    if n != 1:
        raise RuntimeError(
            f"ligand PDBQT must contain exactly one ROOT block; "
            f"found {n} (likely multiple ligand copies were merged); "
            f"path={path}"
        )
    return n


class OpenBabelPDBQTPreparer:
    def __init__(
        self,
        obabel_bin: Optional[Union[str, Path]] = None,
        timeout_sec: int = 120,
    ) -> None:
        self.obabel_bin = (
            None if obabel_bin is None else str(obabel_bin)
        )
        self.timeout_sec = int(timeout_sec)
        self._resolved: Optional[str] = None

    # ------------------------------------------------------------------ #
    def _resolve_bin(self) -> str:
        if self._resolved:
            return self._resolved
        candidates: List[Optional[str]] = [
            self.obabel_bin,
            os.environ.get("OBABEL_BIN"),
            shutil.which("obabel"),
        ]
        for c in candidates:
            if c and Path(c).is_file() and os.access(c, os.X_OK):
                self._resolved = c
                return c
        raise RuntimeError(
            "OpenBabel binary 'obabel' not found. Install OpenBabel and "
            "put it on PATH, or set OBABEL_BIN=/full/path/to/obabel."
        )

    def check_available(self) -> bool:
        try:
            self._resolve_bin()
            return True
        except RuntimeError:
            return False

    # ------------------------------------------------------------------ #
    def prepare_receptor(
        self, receptor_pdb: Path, output_pdbqt: Path,
    ) -> Path:
        """Convert receptor PDB → PDBQT. Tries `-xr` first (rigid receptor);
        falls back to plain conversion if that flag isn't supported by
        the installed OpenBabel."""
        receptor_pdb = Path(receptor_pdb)
        output_pdbqt = Path(output_pdbqt)
        output_pdbqt.parent.mkdir(parents=True, exist_ok=True)
        bin_ = self._resolve_bin()

        cmd = [bin_, str(receptor_pdb), "-O", str(output_pdbqt), "-xr"]
        proc = self._run(cmd, output_pdbqt.parent / "obabel_receptor.log")
        if proc.returncode != 0 or not output_pdbqt.is_file():
            # fallback without -xr
            cmd2 = [bin_, str(receptor_pdb), "-O", str(output_pdbqt)]
            proc2 = self._run(cmd2, output_pdbqt.parent / "obabel_receptor.log")
            if proc2.returncode != 0 or not output_pdbqt.is_file():
                raise RuntimeError(
                    "OpenBabel receptor conversion failed; see "
                    f"{output_pdbqt.parent / 'obabel_receptor.log'} for details."
                )
        return output_pdbqt

    def prepare_ligand(
        self, ligand_pdb: Path, output_pdbqt: Path,
    ) -> Path:
        """Convert native ligand PDB → PDBQT. We do NOT regenerate 3D
        coordinates (--gen3d) because the ligand already has native
        coordinates from the reference structure. Hydrogens are added
        with `-h`.
        """
        ligand_pdb = Path(ligand_pdb)
        output_pdbqt = Path(output_pdbqt)
        output_pdbqt.parent.mkdir(parents=True, exist_ok=True)
        bin_ = self._resolve_bin()

        cmd = [bin_, str(ligand_pdb), "-O", str(output_pdbqt), "-h"]
        proc = self._run(cmd, output_pdbqt.parent / "obabel_ligand.log")
        if proc.returncode != 0 or not output_pdbqt.is_file():
            raise RuntimeError(
                "OpenBabel ligand conversion failed; see "
                f"{output_pdbqt.parent / 'obabel_ligand.log'} for details."
            )
        return output_pdbqt

    # ------------------------------------------------------------------ #
    def _run(self, cmd: List[str], log_path: Path):
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.timeout_sec,
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            f"$ {' '.join(cmd)}\n\n"
            f"--- stdout ---\n{proc.stdout or ''}\n"
            f"--- stderr ---\n{proc.stderr or ''}\n",
            encoding="utf-8",
        )
        return proc


__all__ = [
    "OpenBabelPDBQTPreparer",
    "count_pdbqt_root_blocks",
    "assert_single_root_pdbqt",
]
