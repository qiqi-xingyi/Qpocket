# Author: Yuqi Zhang
"""V2 run-time logging utilities.

Provides a single RunLogger object that:
  - Configures stdlib `logging` with stream + file handlers.
  - Writes structured progress events to progress.jsonl.
  - Maintains run_status.json snapshot of all task states.
  - Per-task logging (task_run.log, error.log).

Intentionally lightweight — no external deps beyond stdlib.

Console output is INFO+ by default (summary-level only). The full
debug stream goes to the file handlers. We never log large coordinate
arrays or full bitstring counts to console; only summary metrics.
"""
from __future__ import annotations

import json
import logging
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


_LOGGER_NAME = "ras_folding.v2"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class RunLogger:
    """Top-level logger for one `run_v2.py` invocation."""

    def __init__(
        self,
        output_root: Path,
        *,
        log_level: str = "INFO",
        write_run_log: bool = True,
        progress_interval_sec: int = 30,
    ):
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.write_run_log = bool(write_run_log)
        self.progress_interval_sec = int(progress_interval_sec)
        self.run_log_path = self.output_root / "run.log"
        self.status_path = self.output_root / "run_status.json"
        self.progress_path = self.output_root / "progress.jsonl"
        self.start_time = _now_iso()
        # state
        self._status: Dict[str, Any] = {
            "start_time": self.start_time,
            "last_update_time": self.start_time,
            "current_stage": "preflight",
            "current_task": None,
            "total_tasks": 0,
            "completed_tasks": [],
            "failed_tasks": [],
            "task_status": {},  # task_id -> {status, stage, last_update}
        }
        self._task_handlers: Dict[str, logging.FileHandler] = {}
        # configure root v2 logger
        self.logger = logging.getLogger(_LOGGER_NAME)
        self.logger.setLevel(logging.DEBUG)
        # Reset handlers in case of re-init.
        self.logger.handlers = []
        self.logger.propagate = False
        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)-5s] %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        # stream handler at user level
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(getattr(logging, str(log_level).upper(), logging.INFO))
        sh.setFormatter(fmt)
        self.logger.addHandler(sh)
        if self.write_run_log:
            fh = logging.FileHandler(self.run_log_path, mode="w",
                                      encoding="utf-8")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(fmt)
            self.logger.addHandler(fh)
        # touch progress file
        if self.write_run_log:
            self.progress_path.write_text("")
        self._flush_status()

    # ---------- low-level emit ---------- #

    def _emit_progress(self, event: str, *,
                        task_id: Optional[str] = None,
                        stage: Optional[str] = None,
                        status: Optional[str] = None,
                        message: Optional[str] = None,
                        metrics: Optional[Dict[str, Any]] = None) -> None:
        if not self.write_run_log:
            return
        rec = {
            "time": _now_iso(),
            "event": event,
            "task_id": task_id,
            "stage": stage,
            "status": status,
            "message": message,
            "metrics": metrics or {},
        }
        try:
            with self.progress_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, default=str) + "\n")
        except Exception:
            # Logging must never crash the run.
            pass

    def _flush_status(self) -> None:
        if not self.write_run_log:
            return
        self._status["last_update_time"] = _now_iso()
        try:
            self.status_path.write_text(
                json.dumps(self._status, indent=2, default=str)
            )
        except Exception:
            pass

    # ---------- public API ---------- #

    def banner(self, args, n_tasks: int) -> None:
        """Print banner at start. `args` is argparse Namespace."""
        self._status["total_tasks"] = int(n_tasks)
        self._status["args"] = vars(args).copy() if args else {}
        log = self.logger
        log.info("[V2] ===== Starting V2 run =====")
        log.info(f"[V2] output_root = {self.output_root}")
        log.info(f"[V2] backend = {getattr(args, 'backend', '?')}, "
                  f"execution_mode = {getattr(args, 'ibm_execution_mode', '?')}, "
                  f"ibm_backend = {getattr(args, 'ibm_backend', '?')}")
        log.info(f"[V2] prior_mode = {getattr(args, 'prior_mode', '?')}, "
                  f"prior_n_samples = {getattr(args, 'prior_n_samples', '?')}, "
                  f"taus = {getattr(args, 'taus', '?')}")
        log.info(f"[V2] shot_budget_mode = "
                  f"{getattr(args, 'shot_budget_mode', '?')}, "
                  f"max_shots_per_task = "
                  f"{getattr(args, 'max_shots_per_task', '?')}")
        log.info(f"[V2] tasks = {n_tasks}, "
                  f"dry_run = {getattr(args, 'dry_run', False)}")
        log.info(f"[V2] docking = "
                  f"{'skipped' if getattr(args, 'skip_docking', False) else 'enabled'}, "
                  f"selection_mode = "
                  f"{getattr(args, 'docking_selection_mode', '?')}, "
                  f"max_generated_docking = "
                  f"{getattr(args, 'max_generated_docking_candidates', '?')}, "
                  f"include_anchor = "
                  f"{getattr(args, 'include_oracle_anchor_in_docking', True)}")
        log.info(f"[V2] safety: oracle_anchor=enabled, "
                  f"generated-only-metrics-exclude-anchor=True, "
                  f"ranking-untouched=True, V1-backward-compat=True")
        self._emit_progress(
            "run_start", stage="preflight",
            metrics={"n_tasks": int(n_tasks)},
        )
        self._flush_status()

    def shot_budget_summary(self, allocations: List[Dict[str, Any]]) -> None:
        """Print shot allocation across tasks before any execution."""
        if not allocations:
            return
        log = self.logger
        log.info("[V2] shot budget summary:")
        total_shots = 0
        total_circuits = 0
        for a in allocations:
            log.info(
                f"[V2]   {a.get('task_id', '?'):28s} "
                f"n_res={a.get('n_res', '?')}, "
                f"shots={a.get('allocated_total_shots', 0):>10d}, "
                f"n_circuits={a.get('n_circuits', 0):>5d}"
            )
            total_shots += int(a.get('allocated_total_shots', 0))
            total_circuits += int(a.get('n_circuits', 0))
        log.info(f"[V2] total_shots={total_shots:,}  "
                  f"total_circuits={total_circuits:,}")
        self._emit_progress(
            "shot_budget_summary", stage="preflight",
            metrics={
                "total_shots": int(total_shots),
                "total_circuits": int(total_circuits),
                "n_tasks": len(allocations),
            },
        )

    def task_start(self, task_id: str, idx: int, total: int,
                    n_res: int, n_bonds: int, n_qubits: int,
                    allocated_shots: int, shots_per_circuit: int,
                    n_circuits: int, prior_mode: str) -> None:
        self._status["current_task"] = task_id
        self._status["current_stage"] = "task_start"
        self._status["task_status"][task_id] = {
            "status": "running", "stage": "preflight",
            "started_at": _now_iso(),
        }
        log = self.logger
        log.info(f"[Task {idx:02d}/{total}] {task_id}")
        log.info(f"  n_res={n_res}, n_bonds={n_bonds}, n_qubits={n_qubits}")
        log.info(f"  allocated_shots={allocated_shots:,}, "
                  f"shots_per_circuit={shots_per_circuit:,}, "
                  f"n_circuits={n_circuits:,}")
        log.info(f"  prior_mode={prior_mode}")
        self._open_task_log(task_id)
        self._emit_progress(
            "task_start", task_id=task_id, stage="preflight",
            status="running",
            metrics={
                "n_res": n_res, "n_bonds": n_bonds,
                "n_qubits": n_qubits,
                "allocated_shots": allocated_shots,
                "n_circuits": n_circuits,
                "prior_mode": prior_mode,
            },
        )
        self._flush_status()

    def stage(self, task_id: Optional[str], stage: str, message: str,
               level: int = logging.INFO,
               **metrics: Any) -> None:
        prefix = f"[{stage}]" if stage else ""
        msg = f"  {prefix} {message}"
        self.logger.log(level, msg)
        if task_id:
            self._status["task_status"].setdefault(
                task_id, {}
            )["stage"] = stage
            self._status["current_stage"] = stage
        self._emit_progress(
            "stage", task_id=task_id, stage=stage,
            message=message, metrics=metrics,
        )

    def task_done(self, task_id: str, elapsed: float,
                   key_metrics: Dict[str, Any]) -> None:
        self._status["task_status"].setdefault(
            task_id, {}
        )["status"] = "done"
        self._status["task_status"][task_id]["finished_at"] = _now_iso()
        self._status["task_status"][task_id]["elapsed_sec"] = float(elapsed)
        if task_id not in self._status["completed_tasks"]:
            self._status["completed_tasks"].append(task_id)
        log = self.logger
        log.info(f"[Task done] {task_id}  elapsed={elapsed:.1f}s")
        for k, v in key_metrics.items():
            log.info(f"  {k} = {v}")
        self._close_task_log(task_id)
        self._emit_progress(
            "task_done", task_id=task_id, stage="done", status="done",
            metrics={"elapsed_sec": float(elapsed), **key_metrics},
        )
        self._flush_status()

    def task_failed(self, task_id: str, stage: str, error: BaseException,
                     extra: Optional[Dict[str, Any]] = None) -> None:
        tb = traceback.format_exc()
        self._status["task_status"].setdefault(
            task_id, {}
        )["status"] = "failed"
        self._status["task_status"][task_id]["stage"] = stage
        self._status["task_status"][task_id]["error"] = repr(error)
        self._status["task_status"][task_id]["finished_at"] = _now_iso()
        if task_id not in self._status["failed_tasks"]:
            self._status["failed_tasks"].append({
                "task_id": task_id, "stage": stage, "error": repr(error),
            })
        log = self.logger
        log.error(f"[Task failed] {task_id}")
        log.error(f"  stage={stage}")
        log.error(f"  error={error!r}")
        log.error(f"  see error.log under {task_id}/")
        # write error.log under task dir
        task_dir = self.output_root / task_id
        if task_dir.is_dir():
            try:
                (task_dir / "error.log").write_text(
                    f"task_id: {task_id}\n"
                    f"stage: {stage}\n"
                    f"error: {error!r}\n"
                    f"---traceback---\n{tb}\n"
                )
            except Exception:
                pass
        self._close_task_log(task_id)
        self._emit_progress(
            "task_failed", task_id=task_id, stage=stage,
            status="failed", message=repr(error),
            metrics=(extra or {}),
        )
        self._flush_status()

    def fatal(self, message: str, error: Optional[BaseException] = None) -> None:
        """Log a run-level fatal error (auth/backend/csv/import)."""
        log = self.logger
        if error:
            log.error(f"[FATAL] {message}: {error!r}")
            log.error(traceback.format_exc())
        else:
            log.error(f"[FATAL] {message}")
        self._status["fatal"] = {
            "message": message,
            "error": repr(error) if error else None,
            "time": _now_iso(),
        }
        self._flush_status()
        self._emit_progress(
            "fatal", message=message,
            metrics={"error": repr(error) if error else None},
        )

    def run_done(self, n_ok: int, n_failed: int, elapsed: float) -> None:
        self._status["current_stage"] = "done"
        self._status["finished_at"] = _now_iso()
        self._status["n_ok"] = n_ok
        self._status["n_failed"] = n_failed
        self._status["elapsed_sec"] = float(elapsed)
        log = self.logger
        log.info(f"[V2] ===== Run finished: ok={n_ok} failed={n_failed} "
                  f"elapsed={elapsed:.1f}s =====")
        log.info(f"[V2] outputs: {self.output_root}")
        log.info(f"[V2] tail -f {self.output_root / 'run.log'}")
        self._emit_progress(
            "run_done", stage="done",
            metrics={
                "n_ok": int(n_ok), "n_failed": int(n_failed),
                "elapsed_sec": float(elapsed),
            },
        )
        self._flush_status()

    # ---------- per-task file handlers ---------- #

    def _open_task_log(self, task_id: str) -> None:
        if not self.write_run_log:
            return
        if task_id in self._task_handlers:
            return
        task_dir = self.output_root / task_id
        try:
            task_dir.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(task_dir / "task_run.log",
                                       mode="w", encoding="utf-8")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)-5s] %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            ))
            self.logger.addHandler(fh)
            self._task_handlers[task_id] = fh
        except Exception:
            pass

    def _close_task_log(self, task_id: str) -> None:
        h = self._task_handlers.pop(task_id, None)
        if h is None:
            return
        try:
            self.logger.removeHandler(h)
            h.close()
        except Exception:
            pass

    # ---------- IBM job status polling helper ---------- #

    @contextmanager
    def quantum_chunk(self, task_id: str, chunk_idx: int, n_chunks: int,
                       n_circuits: int):
        """Context manager wrapping one quantum chunk submission.
        Logs start, exit, and elapsed."""
        t0 = time.time()
        self.stage(task_id, "quantum",
                    f"submitting chunk {chunk_idx}/{n_chunks}, "
                    f"circuits={n_circuits}",
                    chunk_idx=chunk_idx, n_chunks=n_chunks,
                    n_circuits=n_circuits)
        try:
            yield
        finally:
            elapsed = time.time() - t0
            self.stage(task_id, "quantum",
                        f"chunk {chunk_idx}/{n_chunks} returned, "
                        f"elapsed={elapsed:.1f}s",
                        chunk_idx=chunk_idx, elapsed_sec=elapsed)

    def log_ibm_job_ids(self, task_id: str, job_ids: Iterable[str]) -> None:
        ids = list(job_ids)
        if not ids:
            self.stage(task_id, "quantum",
                        "no job_ids found in seed_*/job_metadata.json")
            return
        for jid in ids:
            self.stage(task_id, "quantum", f"job_id={jid}", job_id=jid)


__all__ = ["RunLogger"]
