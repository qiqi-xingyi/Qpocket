#!/usr/bin/env python3
# Author: Yuqi Zhang
"""Extract the quantum arm from its run records into the comparison layout.

Arm Q is not generated here. It was executed on hardware, and this script
reads what that execution produced and normalises it into the same cell
layout arms P and M write, so all three can be compared by one downstream.

Two things are recorded rather than smoothed over:

*The environment captured in the run record is the environment of this
extraction, not of the hardware run.* The fields that do describe the
original execution -- backend, job identifiers, shot and circuit counts,
pre- and post-transpile depth -- are read from its own records and kept
in a separate block, and anything absent there is left null instead of
being reconstructed.

*Arm Q has one repeat.* Arms P and M are run with independent repeats;
the hardware arm was executed once. A statistic that needs within-arm
variability cannot be computed for Q from this data, and the record says
so instead of letting a single run occupy the same column as three.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from revision.runrecord import RunRecord, write_run_record        # noqa: E402

logger = logging.getLogger("revision.extract_q_arm")

DEFAULT_SOURCE = (
    "result/run_v2/final_ibm_v2_36cases_corridor_bezier_20260430_071417"
)


def _read_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _as_circuit_list(obj: Any) -> List[Dict[str, Any]]:
    """Per-circuit entries from either a bare list or a {"circuits": [...]}."""
    if isinstance(obj, list):
        return [c for c in obj if isinstance(c, dict)]
    if isinstance(obj, dict):
        inner = obj.get("circuits")
        if isinstance(inner, list):
            return [c for c in inner if isinstance(c, dict)]
    return []


def collect_hardware(task_dir: Path) -> Dict[str, Any]:
    """Gather hardware provenance across the task's per-seed directories.

    Absent fields stay null. The point of this block is to state what the
    hardware run recorded, not to present a complete-looking picture.
    """
    seed_dirs = sorted((task_dir / "quantum").glob("seed_*"))
    per_seed: List[Dict[str, Any]] = []
    job_ids: List[str] = []
    total_shots = 0

    for sd in seed_dirs:
        cfg = _read_json(sd / "quantum_config.json") or {}
        jobm = _read_json(sd / "job_metadata.json") or {}
        res = _read_json(sd / "backend_result.json") or {}
        tr = _read_json(sd / "transpile_summary.json") or {}
        circ = _read_json(sd / "quantum_circuits_summary.json") or []

        ids = list(jobm.get("job_ids") or res.get("job_ids") or [])
        job_ids.extend(str(j) for j in ids)
        shots = jobm.get("total_shots") or res.get("total_shots")
        if isinstance(shots, int):
            total_shots += shots

        # Both files wrap their per-circuit entries, but not identically:
        # transpile_summary is a dict with a "circuits" key, and
        # quantum_circuits_summary has been seen both as that shape and as
        # a bare list. Accept either rather than silently yielding nothing.
        tr_circuits = _as_circuit_list(tr)
        circ_list = _as_circuit_list(circ)
        pre = [c["depth_pre_transpile"] for c in circ_list
               if c.get("depth_pre_transpile") is not None]
        post = [c["depth"] for c in tr_circuits if c.get("depth") is not None]
        w_log = [c["num_qubits"] for c in circ_list
                 if c.get("num_qubits") is not None]
        w_phy = [c["num_qubits"] for c in tr_circuits
                 if c.get("num_qubits") is not None]

        per_seed.append({
            "seed_dir": sd.name,
            "backend_name": cfg.get("ibm_backend_name")
            or res.get("backend_name"),
            "backend_type": cfg.get("backend_type") or res.get("backend_type"),
            "execution_mode": cfg.get("execution_mode"),
            "status": res.get("status"),
            "shots_per_circuit": cfg.get("shots_per_circuit"),
            "optimization_level": cfg.get("optimization_level"),
            "seed_transpiler": cfg.get("seed_transpiler"),
            "n_circuits": res.get("n_circuits") or len(tr_circuits) or None,
            "total_shots": shots,
            "n_chunks": jobm.get("n_chunks"),
            "n_job_ids": len(ids),
            "max_shots_per_job": jobm.get("max_shots_per_job"),
            "estimated_total_runtime_sec": jobm.get(
                "estimated_total_runtime_sec"),
            "depth_pre_transpile": (
                {"min": min(pre), "max": max(pre)} if pre else None),
            "depth_post_transpile": (
                {"min": min(post), "max": max(post)} if post else None),
            "width_logical_qubits": (
                {"min": min(w_log), "max": max(w_log)} if w_log else None),
            "width_physical_qubits": (
                {"min": min(w_phy), "max": max(w_phy)} if w_phy else None),
        })

    return {
        "executed_on_hardware": bool(seed_dirs),
        "n_seed_directories": len(seed_dirs),
        "job_ids": job_ids,
        "n_job_ids": len(job_ids),
        "total_shots_across_seed_dirs": total_shots or None,
        "per_seed": per_seed,
        "not_recorded": [
            "provider active time (queue and execution are not separated "
            "in these records)",
            "per-job calibration snapshot",
            "physical qubit layout selected by the transpiler",
        ],
    }


def extract_task(task_dir: Path, out_root: Path,
                 repeat: int = 0) -> Optional[Dict[str, Any]]:
    task_id = task_dir.name
    src = task_dir / "sampler" / "accepted_candidates.csv"
    if not src.is_file():
        logger.warning("[%s] no accepted_candidates.csv - skipped", task_id)
        return None

    cell = out_root / "Q" / task_id / f"repeat_{repeat}"
    cell.mkdir(parents=True, exist_ok=True)
    dst = cell / "candidates.csv"

    # Rewrite into the column order arms P and M emit, so one reader
    # serves all three.
    n = 0
    seen = set()
    with src.open(newline="", encoding="utf-8") as fin, \
            dst.open("w", newline="", encoding="utf-8") as fout:
        r = csv.DictReader(fin)
        w = csv.writer(fout, lineterminator="\n")
        w.writerow(["bitstring", "count", "tau", "filter_energy",
                    "full_energy"])
        for row in r:
            bs = row.get("bitstring") or ""
            if bs:
                seen.add(bs)
            w.writerow([bs, row.get("count"), row.get("tau"),
                        row.get("filter_energy"), row.get("full_energy")])
            n += 1

    summary = _read_json(task_dir / "task_summary.json") or {}
    inp = _read_json(task_dir / "input.json") or {}

    record = RunRecord(
        experiment="g1_kras_pilot", arm="Q", task_id=task_id,
        config={
            "repeat": repeat,
            "extraction": True,
            "source_run": (str(task_dir.relative_to(PROJECT_ROOT))
                           if task_dir.is_relative_to(PROJECT_ROOT)
                           else str(task_dir)),
        },
    ).start(PROJECT_ROOT)

    record.inputs = {
        "ref_pdb": (inp.get("metadata") or {}).get("ref_pdb"),
        "n_residues": inp.get("n_residues"),
        "n_bonds": inp.get("n_bonds"),
        "n_qubits": inp.get("n_qubits"),
        "prior_mode": summary.get("prior_mode"),
        "crystal_leakage_mode": "full",
    }
    record.results = {
        "n_accepted_total": n,
        "n_unique_accepted": len(seen),
        "candidates_path": (str(dst.relative_to(PROJECT_ROOT))
                            if dst.is_relative_to(PROJECT_ROOT) else str(dst)),
        "budget_per_tau": (summary.get("shot_allocation") or {}).get(
            "allocated_total_shots"),
        "n_accepted_quantum_reported": summary.get("n_accepted_quantum"),
        "n_prior_rollout_reported": summary.get("n_prior_rollout"),
        "prior_valid_rate_reported": summary.get("prior_valid_rate"),
    }
    record.quantum = collect_hardware(task_dir)
    record.finish()

    record.note(
        "Arm Q was not executed by this script. The environment block "
        "describes this extraction; the quantum block describes the "
        "hardware run that produced the data."
    )
    record.note(
        "Arm Q has a single repeat. Arms P and M are run with independent "
        "repeats, so any statistic requiring within-arm variability is "
        "unavailable for Q from this data and must not be presented as if "
        "one run supplied it."
    )
    reported = summary.get("n_accepted_quantum")
    if isinstance(reported, int) and reported != n:
        record.note(
            f"Row count {n} differs from the reported n_accepted_quantum "
            f"{reported}; the extracted file is the authority for what is "
            f"actually present."
        )
    write_run_record(record, cell / "run_record.json")
    return {"task_id": task_id, "n_rows": n, "n_unique": len(seen)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default=DEFAULT_SOURCE,
                   help="historical run directory holding the hardware arm")
    p.add_argument("--output-root", default="revision/results/g1_pilot")
    p.add_argument("--repeat", type=int, default=0)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s | %(message)s")

    src_root = Path(args.source)
    if not src_root.is_absolute():
        src_root = PROJECT_ROOT / src_root
    if not src_root.is_dir():
        logger.error("source run directory not found: %s", src_root)
        return 2
    out_root = Path(args.output_root)
    if not out_root.is_absolute():
        out_root = PROJECT_ROOT / out_root

    task_dirs = sorted(d for d in src_root.iterdir()
                       if d.is_dir() and (d / "task_summary.json").is_file())
    logger.info("found %d task directories in %s", len(task_dirs), src_root)

    done, skipped = [], []
    for td in task_dirs:
        out = extract_task(td, out_root, repeat=args.repeat)
        (done if out else skipped).append(td.name)
        if out:
            logger.info("  %-26s rows=%-9d unique=%d",
                        out["task_id"], out["n_rows"], out["n_unique"])

    logger.info("extracted %d tasks, skipped %d", len(done), len(skipped))
    if skipped:
        logger.warning("skipped: %s", ", ".join(skipped))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
