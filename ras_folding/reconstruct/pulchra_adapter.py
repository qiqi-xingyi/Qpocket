# Author: Yuqi Zhang
"""Thin wrapper around the PULCHRA binary.

PULCHRA is a third-party Fortran tool that rebuilds full-atom backbone
+ side chains from CA-only PDBs. It is NOT a conda/pip package. We look
for the binary in this priority order so that an IDE-launched run does
not need a configured shell PATH:

    1. explicit ``pulchra_bin`` argument (DI / CLI)
    2. ``PULCHRA_BIN`` environment variable
    3. ``shutil.which("pulchra")`` (anything on PATH)
    4. project-local ``<project_root>/tools/pulchra/pulchra``

If none of those resolve, we raise loudly with the full list of paths
we tried. We NEVER fabricate an all-atom structure.

Output-file handling
--------------------
PULCHRA produces ``<stem>.rebuilt.pdb`` next to its input. Earlier
revisions of this adapter detected the new file via a directory diff
(``after - before``); when an old ``*.rebuilt.pdb`` was already in the
output dir from a prior run, the diff was empty and the adapter
incorrectly raised "produced no new PDB output" (or, worse, would have
silently picked up the stale file). We now:

* compute a deterministic output path before invocation,
* delete it if it exists when ``overwrite=True``,
* prefer the deterministic path after invocation,
* fall back to a *time-filtered* directory diff (only files whose mtime
  is at or after the command-start timestamp) if the deterministic path
  was not produced (e.g. PULCHRA build with a different naming convention),
* validate the final file: exists, non-empty, contains at least one
  ``ATOM`` record.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union


def _project_tools_pulchra_candidates() -> List[Path]:
    """Return the project-local ``tools/pulchra/pulchra`` candidate paths.

    We try (a) the path inferred from this file's location (the canonical
    answer when the package is installed in-place) and (b) ``cwd`` (the
    canonical answer when the user runs the entry script from the
    project root inside an IDE). Both paths are returned so the resolver
    will probe each — and so the error message can list every path
    that was attempted.
    """
    out: List[Path] = []
    # (a) inferred from package layout: this file is at
    #     <project_root>/ras_folding/reconstruct/pulchra_adapter.py
    try:
        pkg_root = Path(__file__).resolve().parents[2]
        out.append(pkg_root / "tools" / "pulchra" / "pulchra")
    except (IndexError, OSError):
        pass
    # (b) cwd-relative: covers IDE runs where the script's working dir
    #     is the project root.
    try:
        out.append(Path.cwd() / "tools" / "pulchra" / "pulchra")
    except OSError:
        pass
    # de-duplicate while preserving order
    seen: set = set()
    uniq: List[Path] = []
    for p in out:
        rp = p.resolve(strict=False)
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    return uniq


class PulchraAdapter:
    def __init__(
        self,
        pulchra_bin: Optional[Union[str, Path]] = None,
        timeout_sec: int = 120,
    ) -> None:
        self.pulchra_bin = (
            None if pulchra_bin is None else str(pulchra_bin)
        )
        self.timeout_sec = int(timeout_sec)
        self._resolved: Optional[str] = None

    # ------------------------------------------------------------------ #
    def _candidate_paths(self) -> List[Tuple[str, Optional[str]]]:
        """Ordered list of (source_label, candidate_path).

        Path may be None (e.g. PULCHRA_BIN not set) — those entries are
        kept for the error-message listing so the user can see every
        slot the resolver looked at.
        """
        cands: List[Tuple[str, Optional[str]]] = [
            ("explicit pulchra_bin argument", self.pulchra_bin),
            ("PULCHRA_BIN env var", os.environ.get("PULCHRA_BIN")),
            ("PATH (shutil.which 'pulchra')", shutil.which("pulchra")),
        ]
        for p in _project_tools_pulchra_candidates():
            cands.append(
                (f"project tools/pulchra/pulchra ({p})", str(p)),
            )
        return cands

    def _resolve_bin(self) -> str:
        if self._resolved:
            return self._resolved
        cands = self._candidate_paths()
        for _label, c in cands:
            if c and Path(c).is_file() and os.access(c, os.X_OK):
                self._resolved = c
                return c
        # None matched — build a descriptive error listing every slot
        # we probed and what we found there.
        lines = [
            "PULCHRA binary not found. Tried (in priority order):",
        ]
        for label, c in cands:
            if not c:
                lines.append(f"  - {label}: <not set>")
            else:
                if Path(c).exists():
                    if not Path(c).is_file():
                        why = "exists but is not a regular file"
                    elif not os.access(c, os.X_OK):
                        why = "exists but is NOT executable"
                    else:
                        why = "is the file we expected (resolver bug?)"
                else:
                    why = "does not exist"
                lines.append(f"  - {label}: {c}  [{why}]")
        lines.append(
            "Fix: pass --pulchra-bin=<path>, or build PULCHRA in "
            "<project_root>/tools/pulchra/pulchra (so IDE runs don't "
            "need PATH). Reconstruction refuses to fabricate all-atom "
            "output."
        )
        raise RuntimeError("\n".join(lines))

    def check_available(self) -> bool:
        try:
            self._resolve_bin()
            return True
        except RuntimeError:
            return False

    # ------------------------------------------------------------------ #
    @staticmethod
    def _expected_rebuilt_path(local_ca: Path) -> Path:
        """PULCHRA's canonical output name: ``<input>.rebuilt.pdb``.

        Concretely, given ``local_ca = .../top1_ca.pdb``, PULCHRA writes
        ``.../top1_ca.pdb.rebuilt`` on some builds and
        ``.../top1_ca.rebuilt.pdb`` on others. Both have been seen in
        practice, so we treat ``<stem>.rebuilt.pdb`` as the deterministic
        target and let the time-filtered fallback handle the alternate
        suffix.
        """
        return local_ca.with_suffix("").with_suffix(".rebuilt.pdb")

    def rebuild(
        self,
        ca_pdb: Path,
        output_dir: Path,
        sequence: Optional[str] = None,
        overwrite: bool = True,
    ) -> Path:
        """Run PULCHRA on ``ca_pdb``. Returns the path to the rebuilt PDB.

        Selection of the rebuilt file is *deterministic*:

        1. Compute the expected output path (``<stem>.rebuilt.pdb``).
        2. If ``overwrite=True`` and that file exists, delete it before
           invoking PULCHRA so a stale copy from a prior run cannot be
           returned by mistake.
        3. After PULCHRA exits successfully, prefer the deterministic
           path. If it does not exist, fall back to a *time-filtered*
           directory diff: only PDB files whose mtime is at or after
           the command-start timestamp are considered fresh outputs.
           The input copy is excluded.
        4. Validate: file exists, non-empty, contains at least one
           ``ATOM`` line. Otherwise raise RuntimeError pointing at
           pulchra.stdout.txt / pulchra.stderr.txt.

        stdout / stderr are saved as ``pulchra.stdout.txt`` /
        ``pulchra.stderr.txt`` for traceability.
        """
        ca_pdb = Path(ca_pdb)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if not ca_pdb.is_file():
            raise FileNotFoundError(f"CA PDB not found: {ca_pdb}")

        binary = self._resolve_bin()

        # PULCHRA reads from stdin/file and writes alongside the input
        # by default. We copy ca_pdb into output_dir so the rebuilt PDB
        # is contained.
        local_ca = output_dir / ca_pdb.name
        if local_ca.resolve() != ca_pdb.resolve():
            shutil.copyfile(ca_pdb, local_ca)

        # Deterministic expected output path for this invocation.
        expected_out = self._expected_rebuilt_path(local_ca)

        # If overwrite=True, delete the deterministic stale copy AND any
        # other PDB whose name suggests it is a previous PULCHRA output
        # (so the time-filtered fallback below cannot grab them). This
        # is intentionally narrow: we never delete the input copy or
        # files that don't end with the rebuilt-marker substrings.
        if overwrite:
            stale_candidates = []
            if expected_out.exists():
                stale_candidates.append(expected_out)
            for p in output_dir.glob("*.rebuilt.pdb"):
                if p.resolve() != local_ca.resolve():
                    stale_candidates.append(p)
            for p in output_dir.glob("*.pdb.rebuilt"):
                stale_candidates.append(p)
            for p in stale_candidates:
                try:
                    p.unlink()
                except OSError:
                    pass

        # Snapshot existing PDBs *before* PULCHRA runs. We use this to
        # build a fallback set of "files that already existed", then
        # combine with an mtime threshold to identify the new output.
        before: set = set()
        for d in (output_dir, ca_pdb.parent):
            if d.is_dir():
                before.update(p.resolve() for p in d.glob("*.pdb"))
                before.update(p.resolve() for p in d.glob("*.pdb.rebuilt"))

        # Record the wall-clock time *just before* the subprocess starts
        # so we can later filter out PDBs that pre-date this run.
        # Subtract a small tolerance to absorb filesystem mtime
        # quantisation (some FS only have 1 s resolution).
        t_start = time.time() - 1.0

        cmd = [binary, str(local_ca)]
        proc = subprocess.run(
            cmd,
            cwd=str(output_dir),
            capture_output=True,
            text=True,
            timeout=self.timeout_sec,
        )
        stdout_p = output_dir / "pulchra.stdout.txt"
        stderr_p = output_dir / "pulchra.stderr.txt"
        stdout_p.write_text(proc.stdout or "", encoding="utf-8")
        stderr_p.write_text(proc.stderr or "", encoding="utf-8")
        if proc.returncode != 0:
            raise RuntimeError(
                f"PULCHRA exited with code {proc.returncode}. "
                f"stderr: {proc.stderr.strip()[:500]}"
            )

        # 1) Deterministic path first.
        rebuilt: Optional[Path] = None
        if expected_out.is_file():
            rebuilt = expected_out

        # 2) Fallback: time-filtered diff. Only files newly produced by
        # this invocation. This covers PULCHRA builds that emit
        # `*.pdb.rebuilt` (no .pdb suffix) or any other naming variant.
        if rebuilt is None:
            after: set = set()
            for d in (output_dir, ca_pdb.parent):
                if d.is_dir():
                    after.update(p.resolve() for p in d.glob("*.pdb"))
                    after.update(p.resolve() for p in d.glob("*.pdb.rebuilt"))
            fresh = []
            for p in sorted(after - before, key=lambda p: -p.stat().st_mtime):
                if p == local_ca.resolve():
                    continue
                try:
                    if p.stat().st_mtime + 1.0 < t_start:
                        # Pre-dates this run despite being absent from
                        # ``before`` (race on parallel runs). Skip.
                        continue
                except OSError:
                    continue
                fresh.append(p)
            if fresh:
                rebuilt = Path(fresh[0])

        if rebuilt is None:
            raise RuntimeError(
                "PULCHRA completed but produced no fresh PDB output in "
                f"{output_dir}. Expected deterministic path: {expected_out}. "
                f"See pulchra.stdout.txt: {stdout_p}, "
                f"pulchra.stderr.txt: {stderr_p}."
            )

        # File-content validation: non-empty + at least one ATOM line.
        try:
            size = rebuilt.stat().st_size
        except OSError as e:
            raise RuntimeError(
                f"PULCHRA output is unreadable at {rebuilt}: {e!r}. "
                f"See stdout: {stdout_p}, stderr: {stderr_p}."
            ) from e
        if size <= 0:
            raise RuntimeError(
                f"PULCHRA output is empty (size=0) at {rebuilt}. "
                f"See stdout: {stdout_p}, stderr: {stderr_p}."
            )
        if not _pdb_has_atom_record(rebuilt):
            raise RuntimeError(
                "PULCHRA produced an empty/non-atomic PDB at "
                f"{rebuilt} (no ATOM records). Possible cause: input "
                "residues are UNK or unsupported residue names — "
                "PULCHRA cannot rebuild side chains for UNK. "
                f"See stdout: {stdout_p}, stderr: {stderr_p}."
            )

        return rebuilt


def _pdb_has_atom_record(pdb_path: Path) -> bool:
    """Return True iff `pdb_path` contains at least one ATOM line."""
    try:
        with Path(pdb_path).open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("ATOM"):
                    return True
    except OSError:
        return False
    return False


__all__ = ["PulchraAdapter"]
