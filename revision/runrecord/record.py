# Author: Yuqi Zhang
"""Run record — provenance captured at execution time, not reconstructed.

Every field written here is read from the live process or the scheduler.
Nothing is inferred: a value that cannot be observed is recorded as null
with a reason, so the SI can state the exact boundary of what is known
rather than presenting a plausible reconstruction.

The record is the unit the revision cites for hardware, resource, and
environment disclosure.
"""
from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# Packages whose versions are recorded when importable.
_TRACKED_PACKAGES = (
    "numpy", "scipy", "pandas", "qiskit", "qiskit_ibm_runtime",
    "qiskit_aer", "Bio", "sklearn",
)

# SLURM variables worth preserving verbatim.
_SLURM_VARS = (
    "SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_JOB_ACCOUNT",
    "SLURM_JOB_PARTITION", "SLURM_JOB_NODELIST", "SLURM_NNODES",
    "SLURM_CPUS_ON_NODE", "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE",
    "SLURM_SUBMIT_DIR", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID",
)


def _run_git(args: List[str], cwd: Path) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", *args], cwd=str(cwd), capture_output=True,
            text=True, timeout=10,
        )
        if out.returncode != 0:
            return None
        return out.stdout.strip() or None
    except Exception:
        return None


def capture_git(repo_root: Path) -> Dict[str, Any]:
    """Repository state at run time.

    ``dirty`` matters: a run made from a modified working tree is not
    identified by its commit alone, and the record says so instead of
    implying the commit describes the code that ran.
    """
    commit = _run_git(["rev-parse", "HEAD"], repo_root)
    if commit is not None:
        status = _run_git(["status", "--porcelain"], repo_root)
        return {
            "source": "observed_at_run_time",
            "commit": commit,
            "branch": _run_git(["rev-parse", "--abbrev-ref", "HEAD"],
                               repo_root),
            "dirty": bool(status),
            "dirty_files": (status.splitlines() if status else []),
            "unavailable_reason": None,
        }

    # No repository here — typically a compute host that received the code
    # without .git. A stamp written at transfer time can still identify the
    # source, but it is a weaker claim than reading the repository during
    # the run, so it is labelled differently rather than presented as the
    # same thing.
    stamp_path = Path(repo_root) / "revision" / "configs" / "source_commit.json"
    if stamp_path.is_file():
        try:
            stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
            return {
                "source": "stamped_at_sync_time",
                "commit": stamp.get("commit"),
                "branch": stamp.get("branch"),
                "dirty": stamp.get("dirty"),
                "stamped_at_utc": stamp.get("stamped_at_utc"),
                "stamped_from_host": stamp.get("stamped_from_host"),
                "unavailable_reason": (
                    "no git repository at run time; commit identifies the "
                    "source tree as of the transfer, not as observed here"
                ),
            }
        except Exception as e:
            return {
                "source": None, "commit": None, "dirty": None, "branch": None,
                "unavailable_reason": f"commit stamp unreadable: {e!r}",
            }

    return {
        "source": None, "commit": None, "dirty": None, "branch": None,
        "unavailable_reason": (
            "no git repository and no commit stamp at repo_root"
        ),
    }


def capture_environment(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    """Interpreter, packages, host, and repository state."""
    pkgs: Dict[str, Optional[str]] = {}
    for name in _TRACKED_PACKAGES:
        try:
            mod = __import__(name)
            pkgs[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            pkgs[name] = None          # not installed — recorded, not guessed
    env: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "hostname": socket.gethostname(),
        "cpu_count_os": os.cpu_count(),
        "cpu_count_affinity": (
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity") else None
        ),
        "packages": pkgs,
    }
    if repo_root is not None:
        env["git"] = capture_git(Path(repo_root))
    return env


def capture_slurm() -> Dict[str, Any]:
    """Scheduler context. Empty dict outside SLURM — never fabricated."""
    present = {k: os.environ[k] for k in _SLURM_VARS if k in os.environ}
    return {
        "under_slurm": bool(present),
        "variables": present,
    }


@dataclass
class RunRecord:
    """One experiment invocation."""
    experiment: str
    arm: str
    task_id: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    inputs: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, Any] = field(default_factory=dict)
    slurm: Dict[str, Any] = field(default_factory=dict)
    timing: Dict[str, Any] = field(default_factory=dict)
    quantum: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    started_utc: Optional[str] = None
    finished_utc: Optional[str] = None

    _t0: Optional[float] = field(default=None, repr=False, compare=False)
    _c0: Optional[float] = field(default=None, repr=False, compare=False)

    def start(self, repo_root: Optional[Path] = None) -> "RunRecord":
        self.environment = capture_environment(repo_root)
        self.slurm = capture_slurm()
        self.started_utc = time.strftime("%Y-%m-%dT%H:%M:%S+00:00",
                                         time.gmtime())
        self._t0 = time.perf_counter()
        self._c0 = time.process_time()
        return self

    def finish(self) -> "RunRecord":
        self.finished_utc = time.strftime("%Y-%m-%dT%H:%M:%S+00:00",
                                          time.gmtime())
        if self._t0 is not None:
            self.timing["wall_seconds"] = round(
                time.perf_counter() - self._t0, 3)
        if self._c0 is not None:
            self.timing["cpu_seconds"] = round(
                time.process_time() - self._c0, 3)
        return self

    def note(self, message: str) -> None:
        """Record something the reader must know — including shortfalls."""
        self.notes.append(message)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("_t0", None)
        d.pop("_c0", None)
        return d


def _json_default(o: Any) -> Any:
    try:
        import numpy as np
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass
    if isinstance(o, Path):
        return str(o)
    return str(o)


def write_run_record(record: RunRecord, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(record.to_dict(), indent=2, default=_json_default),
        encoding="utf-8",
    )
    return path


__all__ = ["RunRecord", "capture_git", "capture_environment",
           "capture_slurm", "write_run_record"]
