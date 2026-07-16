# Author: Yuqi Zhang
"""FinalPipelineValidator — read artifacts written by the real-backend
validation run and produce PASS / WARN / FAIL check results.

The validator does NOT re-run sampling, scoring, or docking. It only
reads finished outputs and reports diagnostics.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from pipeline_validation.types import CheckResult, ValidationResult


_PASS, _WARN, _FAIL, _SKIP = "PASS", "WARN", "FAIL", "SKIP"
_ANCHOR_DIST_THRESHOLD_A = 1.0
_FALLBACK_RATE_HARD_FAIL = 1.0
_FALLBACK_RATE_WARN = 0.95
_DENSE_CAP_EPSILON = 1e-3


# ---------------------------------------------------------------------- #
# small file readers                                                     #
# ---------------------------------------------------------------------- #

def _read_json(p: Path) -> Optional[Dict[str, Any]]:
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_csv(p: Path) -> List[Dict[str, str]]:
    if not p.is_file():
        return []
    with p.open("r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _f(x: Any) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _i(x: Any) -> Optional[int]:
    if x is None or x == "":
        return None
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


def _read_ca_coords(path: Path) -> Optional[List[List[float]]]:
    if not path.is_file():
        return None
    out: List[List[float]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            try:
                out.append([
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                ])
            except ValueError:
                continue
    return out if out else None


def _euclid(a, b) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


# ---------------------------------------------------------------------- #
# validator                                                              #
# ---------------------------------------------------------------------- #

class FinalPipelineValidator:

    def __init__(
        self,
        *,
        anchor_dist_threshold_a: float = _ANCHOR_DIST_THRESHOLD_A,
        max_dense_fraction_threshold: float = 0.6 + _DENSE_CAP_EPSILON,
        fallback_rate_warn: float = _FALLBACK_RATE_WARN,
    ) -> None:
        self.anchor_dist_threshold_a = float(anchor_dist_threshold_a)
        self.max_dense_fraction_threshold = float(max_dense_fraction_threshold)
        self.fallback_rate_warn = float(fallback_rate_warn)

    # ------------------------------------------------------------------ #
    def validate_task(
        self,
        task_dir: Path,
        task_metadata: Dict[str, Any],
    ) -> ValidationResult:
        task_dir = Path(task_dir)
        checks: List[CheckResult] = []
        summary: Dict[str, Any] = {"task_id": task_metadata.get("task_id")}

        checks += self._check_ibm_sampling(task_dir, task_metadata, summary)
        checks += self._check_decode_validity(task_dir, summary)
        checks += self._check_h_filter(task_dir, summary)
        checks += self._check_densify(task_dir, summary)
        checks += self._check_refinement(task_dir, summary)
        checks += self._check_postprocess(task_dir, summary)
        checks += self._check_oracle(task_dir, summary)
        checks += self._check_reconstruction(
            task_dir, task_metadata, "predicted_top1", summary,
        )
        checks += self._check_reconstruction(
            task_dir, task_metadata, "oracle_best", summary,
        )
        checks += self._check_docking(
            task_dir, "predicted_top1", task_metadata, summary,
        )
        checks += self._check_docking(
            task_dir, "oracle_best", task_metadata, summary,
        )
        checks += self._check_kd_relationship(summary)

        return ValidationResult.from_checks(
            task_id=str(task_metadata.get("task_id") or task_dir.name),
            checks=checks,
            summary=summary,
        )

    # ================================================================== #
    # checks                                                             #
    # ================================================================== #

    # 4.1 IBM sampling ------------------------------------------------- #
    def _check_ibm_sampling(
        self, task_dir: Path, meta: Dict[str, Any], summary: Dict[str, Any],
    ) -> List[CheckResult]:
        out: List[CheckResult] = []
        # paired_tau_sampling=True writes one quantum/backend_result.json
        # at the top level. paired_tau_sampling=False writes
        # quantum/seed_<N>/backend_result.json once per (seed,tau) — we
        # must aggregate across all of them so total_shots / job_ids
        # reflect the full sampling run, not just one seed.
        q_dir = task_dir / "quantum"
        backend_files: List[Path] = []
        raw_files: List[Path] = []

        top_backend = q_dir / "backend_result.json"
        top_raw = q_dir / "raw_counts.json"
        if top_backend.is_file():
            backend_files.append(top_backend)
        if top_raw.is_file():
            raw_files.append(top_raw)

        seed_dirs: List[Path] = []
        if q_dir.is_dir():
            seed_dirs = sorted(
                d for d in q_dir.iterdir()
                if d.is_dir() and d.name.startswith("seed_")
            )
        for sd in seed_dirs:
            b = sd / "backend_result.json"
            r = sd / "raw_counts.json"
            if b.is_file():
                backend_files.append(b)
            if r.is_file():
                raw_files.append(r)

        summary["quantum_result_files"] = [str(p) for p in backend_files]
        summary["raw_counts_files"] = [str(p) for p in raw_files]

        if not backend_files or not raw_files:
            summary["ibm_sampling_pass"] = False
            out.append(CheckResult(
                name="ibm_sampling.files_present",
                status=_FAIL,
                message=(
                    f"no usable backend_result.json/raw_counts.json found "
                    f"under {q_dir} (checked top-level and "
                    f"{len(seed_dirs)} seed_* subdirs)"
                ),
            ))
            return out

        # Aggregate backend_result.json across all sources.
        backends: List[Dict[str, Any]] = []
        for p in backend_files:
            d = _read_json(p)
            if d is not None:
                backends.append(d)
        raws: List[Dict[str, Any]] = []
        for p in raw_files:
            d = _read_json(p)
            if d is not None:
                raws.append(d)

        if not backends or not raws:
            summary["ibm_sampling_pass"] = False
            out.append(CheckResult(
                name="ibm_sampling.files_present",
                status=_FAIL,
                message=(
                    f"backend_result.json / raw_counts.json present but "
                    f"unreadable under {q_dir}"
                ),
            ))
            return out

        # status: PASS only if every backend reports "done".
        statuses = [b.get("status") for b in backends]
        status_ok = all(s == "done" for s in statuses) and bool(statuses)
        # job_ids: union, preserving order, deduped.
        job_ids: List[str] = []
        seen_ids: set = set()
        for b in backends:
            for jid in (b.get("job_ids") or []):
                if jid not in seen_ids:
                    seen_ids.add(jid)
                    job_ids.append(jid)
        # First (canonical) backend supplies n_circuits diagnostic — not aggregated.
        backend = backends[0]
        # Aer simulator runs carry no IBM job ids; the job_ids check is
        # IBM-specific and must not FAIL an otherwise-valid local Aer run.
        is_aer = bool(backends) and all(
            str(b.get("backend_type", "")).startswith("aer")
            for b in backends
        )

        # total_shots: prefer summing actual count totals from raw_counts
        # (the source of truth) over backend metadata, which can be a
        # single-tau echo when paired_tau_sampling=False.
        unique_bs: set = set()
        wrong_len = 0
        n_qubits_meta = meta.get("n_qubits")
        if n_qubits_meta is None:
            for r in raws:
                cs = r.get("circuits") or []
                if cs:
                    first_meta = (cs[0].get("metadata") or {})
                    if first_meta.get("n_qubits") is not None:
                        n_qubits_meta = first_meta.get("n_qubits")
                        break

        total_shots_from_counts = 0
        for r in raws:
            for c in r.get("circuits", []) or []:
                counts = c.get("counts") or {}
                for bs, cnt in counts.items():
                    bs2 = "".join(str(bs).split())
                    if (
                        n_qubits_meta is not None
                        and len(bs2) != int(n_qubits_meta)
                    ):
                        wrong_len += 1
                    unique_bs.add(bs2)
                    try:
                        total_shots_from_counts += int(cnt)
                    except (TypeError, ValueError):
                        pass

        # Fall back to summed backend metadata if raw counts had no
        # numerical totals (extremely defensive — the real IBM path
        # always returns counts).
        if total_shots_from_counts > 0:
            total_shots = int(total_shots_from_counts)
        else:
            total_shots = sum(
                (_i(b.get("total_shots")) or 0) for b in backends
            )

        summary["job_ids"] = job_ids
        summary["total_shots"] = total_shots
        summary["unique_bitstrings"] = len(unique_bs)
        summary["n_wrong_length_bitstrings"] = wrong_len
        summary["n_qubits"] = n_qubits_meta

        ok = (
            status_ok
            and (bool(job_ids) or is_aer)
            and total_shots > 0
            and len(unique_bs) > 0
            and wrong_len == 0
        )
        summary["ibm_sampling_pass"] = bool(ok)
        summary["is_aer"] = bool(is_aer)

        # Compose a status message that surfaces every backend's status,
        # not just the first — multiple seeds can independently fail.
        statuses_msg = ", ".join(repr(s) for s in statuses)
        out.append(CheckResult(
            name="ibm_sampling.status",
            status=_PASS if status_ok else _FAIL,
            message=(
                f"backend_result.status(es)={statuses_msg} "
                f"(n_backend_files={len(backends)})"
            ),
            details={
                "statuses": statuses,
                "errors": [b.get("error") for b in backends],
            },
        ))
        out.append(CheckResult(
            name="ibm_sampling.job_ids",
            status=(_SKIP if is_aer else (_PASS if job_ids else _FAIL)),
            message=(
                "aer backend — no IBM job ids (N/A)" if is_aer
                else f"job_ids = {job_ids}"
            ),
        ))
        out.append(CheckResult(
            name="ibm_sampling.total_shots",
            status=_PASS if total_shots > 0 else _FAIL,
            message=f"total_shots = {total_shots}",
        ))
        out.append(CheckResult(
            name="ibm_sampling.unique_bitstrings",
            status=_PASS if unique_bs else _FAIL,
            message=f"unique_bitstrings = {len(unique_bs)}",
        ))
        out.append(CheckResult(
            name="ibm_sampling.bitstring_length_consistency",
            status=_PASS if wrong_len == 0 else _FAIL,
            message=(
                "all bitstrings match n_qubits"
                if wrong_len == 0
                else f"{wrong_len} bitstrings have wrong length "
                f"(expected n_qubits={n_qubits_meta})"
            ),
            details={"n_wrong_length_bitstrings": wrong_len},
        ))
        return out

    # 4.2 decode / validity ------------------------------------------- #
    def _check_decode_validity(
        self, task_dir: Path, summary: Dict[str, Any],
    ) -> List[CheckResult]:
        out: List[CheckResult] = []
        rows = _read_csv(task_dir / "sampler" / "batch_summary_by_tau.csv")
        if not rows:
            summary["decode_validity_pass"] = False
            out.append(CheckResult(
                name="decode_validity.batch_summary_present",
                status=_FAIL,
                message="sampler/batch_summary_by_tau.csv missing or empty",
            ))
            return out

        n_raw = sum(_i(r.get("n_raw")) or 0 for r in rows)
        n_valid = sum(_i(r.get("n_valid")) or 0 for r in rows)
        n_accepted = sum(_i(r.get("n_accepted")) or 0 for r in rows)
        n_fallback = sum(_i(r.get("fallback_triggered")) or 0 for r in rows)
        valid_rate = (n_valid / n_raw) if n_raw > 0 else 0.0
        fallback_rate = (n_fallback / n_raw) if n_raw > 0 else 0.0

        # invalid_reason histogram lives in JSON: per-batch reports were
        # not preserved on the row level by the existing CSV writer, so
        # we approximate with fallback_triggered + (n_raw - n_valid).
        invalid_reason_counts = {"fallback_triggered": n_fallback}
        if n_raw - n_valid - n_fallback > 0:
            invalid_reason_counts["other"] = n_raw - n_valid - n_fallback

        summary["n_raw_total"] = n_raw
        summary["n_valid_total"] = n_valid
        summary["n_accepted_total"] = n_accepted
        summary["valid_rate"] = float(valid_rate)
        summary["fallback_rate"] = float(fallback_rate)
        summary["invalid_reason_counts"] = invalid_reason_counts

        out.append(CheckResult(
            name="decode_validity.n_raw",
            status=_PASS if n_raw > 0 else _FAIL,
            message=f"n_raw_total = {n_raw}",
        ))
        out.append(CheckResult(
            name="decode_validity.valid_rate",
            status=_PASS if n_valid > 0 else _FAIL,
            message=f"valid_rate = {valid_rate:.4f}",
            details={"n_valid_total": n_valid, "valid_rate": valid_rate},
        ))
        out.append(CheckResult(
            name="decode_validity.fallback_rate",
            status=(
                _FAIL if fallback_rate >= _FALLBACK_RATE_HARD_FAIL
                else (_WARN if fallback_rate > self.fallback_rate_warn else _PASS)
            ),
            message=f"fallback_rate = {fallback_rate:.4f}",
            details={"n_fallback": n_fallback},
        ))
        out.append(CheckResult(
            name="decode_validity.n_accepted",
            status=_PASS if n_accepted > 0 else _FAIL,
            message=f"n_accepted_total = {n_accepted}",
        ))
        summary["decode_validity_pass"] = bool(
            n_raw > 0 and n_valid > 0 and n_accepted > 0
            and fallback_rate < _FALLBACK_RATE_HARD_FAIL
        )
        return out

    # 4.3 H_filter ----------------------------------------------------- #
    def _check_h_filter(
        self, task_dir: Path, summary: Dict[str, Any],
    ) -> List[CheckResult]:
        out: List[CheckResult] = []
        rows = _read_csv(task_dir / "sampler" / "batch_summary_by_tau.csv")
        if not rows:
            return [CheckResult(
                name="h_filter.batch_summary_present",
                status=_SKIP,
                message="batch_summary missing",
            )]

        per_tau_filter: Dict[str, Optional[float]] = {}
        per_tau_full: Dict[str, Optional[float]] = {}
        fav_means: List[float] = []
        for r in rows:
            tau = r.get("tau")
            per_tau_filter[str(tau)] = _f(r.get("mean_filter_energy_accepted"))
            per_tau_full[str(tau)] = _f(r.get("mean_full_energy_accepted"))
            v = _f(r.get("favorable_contact_miss_mean"))
            if v is not None:
                fav_means.append(v)

        fav_overall = (
            float(sum(fav_means) / len(fav_means)) if fav_means else None
        )
        summary["favorable_contact_miss_mean"] = fav_overall
        summary["mean_filter_energy_by_tau"] = per_tau_filter
        summary["mean_full_energy_by_tau"] = per_tau_full

        out.append(CheckResult(
            name="h_filter.favorable_contact_miss_present",
            status=_PASS if fav_overall is not None else _WARN,
            message=(
                f"favorable_contact_miss_mean = {fav_overall:.4f}"
                if fav_overall is not None
                else "favorable_contact_miss_mean missing — MJ table or sequence "
                "may be unavailable"
            ),
            details={"favorable_contact_miss_mean": fav_overall},
        ))

        # τ-trend warning: highest-τ accepted mean_filter should be <=
        # lowest-τ + small slack. Pure diagnostic, never FAIL.
        warn_msg = None
        keys = sorted(per_tau_filter.keys(), key=lambda k: float(k) if k else 0)
        if len(keys) >= 2:
            lo = per_tau_filter[keys[0]]
            hi = per_tau_filter[keys[-1]]
            if lo is not None and hi is not None and hi > lo + 0.5:
                warn_msg = (
                    f"mean_filter_energy at τ={keys[-1]} ({hi:.3f}) "
                    f"is higher than at τ={keys[0]} ({lo:.3f}); "
                    "H_filter τ-trend looks inverted"
                )
        summary["h_filter_trend_warning"] = warn_msg
        out.append(CheckResult(
            name="h_filter.tau_trend",
            status=_WARN if warn_msg else _PASS,
            message=warn_msg or "τ-trend OK or insufficient data",
        ))
        return out

    # 4.4 densify ------------------------------------------------------ #
    def _check_densify(
        self, task_dir: Path, summary: Dict[str, Any],
    ) -> List[CheckResult]:
        ds = _read_json(task_dir / "densify" / "densify_summary.json")
        if ds is None:
            summary["densify_pass"] = None
            return [CheckResult(
                name="densify.summary_present",
                status=_SKIP,
                message="densify/densify_summary.json missing — densify disabled",
            )]
        out: List[CheckResult] = []
        n_parents = _i(ds.get("n_parent_selected")) or 0
        n_gen = _i(ds.get("n_children_generated")) or 0
        n_kept = _i(ds.get("n_children_kept")) or 0
        mean_local = _f(ds.get("mean_local_rmsd_to_parent"))
        max_local = _f(ds.get("max_local_rmsd_to_parent"))
        max_local_cap = _f(ds.get("max_local_rmsd"))
        best_dE = _f(ds.get("best_energy_delta"))

        # dense_fraction surfaces from refinement summary
        ref = _read_json(task_dir / "refinement" / "refinement_summary.json") or {}
        dense_frac = _f(ref.get("dense_fraction_in_subspace"))
        dense_cap_applied = bool(ref.get("dense_fraction_cap_applied"))

        summary["n_children_generated"] = n_gen
        summary["n_children_kept"] = n_kept
        summary["n_parent_selected"] = n_parents
        summary["mean_local_rmsd"] = mean_local
        summary["best_energy_delta"] = best_dE
        summary["dense_fraction_in_subspace"] = dense_frac
        summary["dense_cap_applied"] = dense_cap_applied

        out.append(CheckResult(
            name="densify.parents_selected",
            status=_PASS if n_parents > 0 else _FAIL,
            message=f"n_parent_selected = {n_parents}",
        ))
        out.append(CheckResult(
            name="densify.children_generated",
            status=_PASS if n_gen > 0 else _FAIL,
            message=f"n_children_generated = {n_gen}",
        ))
        out.append(CheckResult(
            name="densify.children_kept",
            status=_PASS if n_kept > 0 else _WARN,
            message=f"n_children_kept = {n_kept}",
        ))
        if max_local is not None and max_local_cap is not None:
            out.append(CheckResult(
                name="densify.local_rmsd_cap",
                status=_PASS if max_local <= max_local_cap + 1e-9 else _FAIL,
                message=(
                    f"max_local_rmsd_to_parent={max_local:.3f} "
                    f"<= cap={max_local_cap:.3f}"
                ),
            ))
        if dense_frac is not None:
            out.append(CheckResult(
                name="densify.dense_fraction_within_cap",
                status=(
                    _PASS if dense_frac <= self.max_dense_fraction_threshold
                    else _FAIL
                ),
                message=(
                    f"dense_fraction_in_subspace={dense_frac:.3f} "
                    f"(cap≈{self.max_dense_fraction_threshold:.3f})"
                ),
                details={"dense_cap_applied": dense_cap_applied},
            ))

        summary["densify_pass"] = bool(
            n_parents > 0 and n_gen > 0 and n_kept >= 0
            and (dense_frac is None
                 or dense_frac <= self.max_dense_fraction_threshold)
        )
        return out

    # 4.5 refinement -------------------------------------------------- #
    def _check_refinement(
        self, task_dir: Path, summary: Dict[str, Any],
    ) -> List[CheckResult]:
        ref = _read_json(task_dir / "refinement" / "refinement_summary.json")
        if ref is None:
            summary["refinement_pass"] = False
            return [CheckResult(
                name="refinement.summary_present",
                status=_FAIL,
                message="refinement/refinement_summary.json missing",
            )]
        out: List[CheckResult] = []
        n_selected = _i(ref.get("n_selected")) or 0
        n_eligible = _i(ref.get("n_eligible")) or 0
        rwe = _f(ref.get("refined_weight_entropy"))
        n_couple = _i(ref.get("n_nonzero_couplings")) or 0
        coup_density = _f(ref.get("coupling_density"))
        topk_overlap = _f(ref.get("top_k_overlap_energy_vs_refined"))

        summary["n_selected_subspace"] = n_selected
        summary["n_eligible"] = n_eligible
        summary["refined_weight_entropy"] = rwe
        summary["n_nonzero_couplings"] = n_couple
        summary["coupling_density"] = coup_density
        summary["top_k_overlap_energy_vs_refined"] = topk_overlap

        # refined_candidates count
        cand_rows = _read_csv(
            task_dir / "refinement" / "refined_candidates.csv",
        )
        summary["n_refined_candidates"] = len(cand_rows)
        out.append(CheckResult(
            name="refinement.subspace_nonempty",
            status=_PASS if n_selected > 0 else _FAIL,
            message=f"n_selected_subspace = {n_selected}",
        ))
        out.append(CheckResult(
            name="refinement.refined_candidates_nonempty",
            status=_PASS if cand_rows else _FAIL,
            message=f"refined_candidates rows = {len(cand_rows)}",
        ))
        out.append(CheckResult(
            name="refinement.refined_weight_entropy_present",
            status=_PASS if rwe is not None else _WARN,
            message=f"refined_weight_entropy = {rwe}",
        ))
        out.append(CheckResult(
            name="refinement.couplings",
            status=_PASS if n_couple >= 0 else _WARN,
            message=f"n_nonzero_couplings = {n_couple}",
        ))
        summary["refinement_pass"] = bool(n_selected > 0 and cand_rows)
        return out

    # 4.6 postprocess ------------------------------------------------- #
    def _check_postprocess(
        self, task_dir: Path, summary: Dict[str, Any],
    ) -> List[CheckResult]:
        out: List[CheckResult] = []
        pred = _read_json(task_dir / "postprocess" / "prediction_summary.json")
        top1_pdb = task_dir / "postprocess" / "top1_ca.pdb"
        top_csv = task_dir / "postprocess" / "final_top_candidates.csv"

        if pred is None:
            summary["postprocess_pass"] = False
            out.append(CheckResult(
                name="postprocess.summary_present",
                status=_FAIL,
                message="postprocess/prediction_summary.json missing",
            ))
            return out

        summary["top1_bitstring"] = pred.get("top1_bitstring")
        summary["top1_is_dense"] = pred.get("top1_is_dense")
        summary["top1_full_energy"] = pred.get("top1_full_energy")
        summary["top1_refined_score"] = pred.get("top1_refined_score")
        summary["n_basins"] = pred.get("n_basins")
        summary["top_basin_weight"] = pred.get("top_basin_weight")

        out.append(CheckResult(
            name="postprocess.top1_present",
            status=_PASS if pred.get("top1_bitstring") else _FAIL,
            message=f"top1_bitstring = {pred.get('top1_bitstring')!r}",
        ))
        out.append(CheckResult(
            name="postprocess.top1_pdb",
            status=_PASS if top1_pdb.is_file() else _FAIL,
            message=f"{top1_pdb} {'exists' if top1_pdb.is_file() else 'missing'}",
        ))
        out.append(CheckResult(
            name="postprocess.basins",
            status=_PASS if (pred.get("n_basins") or 0) >= 1 else _FAIL,
            message=f"n_basins = {pred.get('n_basins')}",
        ))
        out.append(CheckResult(
            name="postprocess.final_top_candidates_csv",
            status=_PASS if top_csv.is_file() else _FAIL,
            message=f"final_top_candidates.csv {'exists' if top_csv.is_file() else 'missing'}",
        ))
        summary["postprocess_pass"] = bool(
            pred.get("top1_bitstring")
            and top1_pdb.is_file()
            and (pred.get("n_basins") or 0) >= 1
        )
        return out

    # 4.7 oracle ------------------------------------------------------ #
    def _check_oracle(
        self, task_dir: Path, summary: Dict[str, Any],
    ) -> List[CheckResult]:
        oracle_p = task_dir / "oracle" / "oracle_best_summary.json"
        oracle_pdb = task_dir / "oracle" / "oracle_best_ca.pdb"
        oracle = _read_json(oracle_p)
        if oracle is None:
            summary["oracle_pass"] = False
            return [CheckResult(
                name="oracle.summary_present",
                status=_FAIL,
                message=f"{oracle_p} missing",
            )]

        out: List[CheckResult] = []
        best_rmsd = _f(oracle.get("best_rmsd"))
        source = oracle.get("best_source")
        is_dense = oracle.get("best_is_dense")
        valid_sources = {"accepted", "dense", "refined", "postprocess"}

        # predicted top1 RMSD is recorded by structure_analysis.json
        struct = _read_json(task_dir / "analysis" / "structure_analysis.json")
        pred_top1_rmsd = None
        if struct:
            t1 = struct.get("top1") or {}
            pred_top1_rmsd = _f(t1.get("rmsd_to_reference"))

        summary["predicted_top1_rmsd"] = pred_top1_rmsd
        summary["oracle_best_rmsd"] = best_rmsd
        summary["oracle_best_source"] = source
        summary["oracle_best_is_dense"] = is_dense
        if pred_top1_rmsd is not None and best_rmsd is not None:
            summary["delta_rmsd_pred_minus_oracle"] = (
                pred_top1_rmsd - best_rmsd
            )
        else:
            summary["delta_rmsd_pred_minus_oracle"] = None

        out.append(CheckResult(
            name="oracle.summary_present",
            status=_PASS,
            message="oracle_best_summary.json present",
        ))
        out.append(CheckResult(
            name="oracle.pdb_present",
            status=_PASS if oracle_pdb.is_file() else _WARN,
            message=f"{oracle_pdb} {'exists' if oracle_pdb.is_file() else 'missing'}",
        ))
        out.append(CheckResult(
            name="oracle.best_rmsd_present",
            status=_PASS if best_rmsd is not None else _WARN,
            message=f"oracle_best_rmsd = {best_rmsd}",
        ))
        out.append(CheckResult(
            name="oracle.source_known",
            status=_PASS if (source in valid_sources) else _WARN,
            message=f"oracle_best_source = {source!r}",
        ))
        if pred_top1_rmsd is not None and best_rmsd is not None:
            ok = best_rmsd <= pred_top1_rmsd + 1e-6
            out.append(CheckResult(
                name="oracle.dominates_predicted",
                status=_PASS if ok else _WARN,
                message=(
                    f"oracle_best_rmsd ({best_rmsd:.3f}) <= "
                    f"predicted_top1_rmsd ({pred_top1_rmsd:.3f})"
                    if ok else
                    f"oracle_best_rmsd ({best_rmsd:.3f}) > "
                    f"predicted_top1_rmsd ({pred_top1_rmsd:.3f}) — unexpected"
                ),
            ))
        summary["oracle_pass"] = bool(
            (best_rmsd is not None)
            and (source in valid_sources)
        )
        return out

    # 4.8 reconstruction --------------------------------------------- #
    def _check_reconstruction(
        self,
        task_dir: Path,
        meta: Dict[str, Any],
        receptor_label: str,         # "predicted_top1" | "oracle_best"
        summary: Dict[str, Any],
    ) -> List[CheckResult]:
        rec_dir = task_dir / receptor_label / "reconstruct"
        rec_p = rec_dir / "reconstruction_summary.json"
        rec = _read_json(rec_p)
        prefix = "predicted" if receptor_label == "predicted_top1" else "oracle"

        if rec is None:
            summary[f"{prefix}_reconstruction_pass"] = False
            return [CheckResult(
                name=f"reconstruction[{receptor_label}].summary_present",
                status=_FAIL,
                message=f"{rec_p} missing",
            )]

        out: List[CheckResult] = []
        rebuilt_pdb = rec_dir / "rebuilt_fragment.pdb"
        embedded_pdb = rec_dir / "embedded_receptor.pdb"
        status = rec.get("status")
        recon_meta = rec.get("metadata") or {}

        # Read ca_drift fields from EITHER top-level OR metadata. The
        # pipeline now writes both, but older runs only wrote metadata.
        def _pick(key: str) -> Any:
            v = rec.get(key)
            if v is not None:
                return v
            return recon_meta.get(key)

        ca_drift_rmsd = _f(_pick("ca_drift_rmsd"))
        ca_drift_max = _f(_pick("ca_drift_max"))
        ca_drift_warning = _pick("ca_drift_warning")
        ca_drift_shape_mismatch = _pick("ca_drift_shape_mismatch")

        summary[f"{prefix}_ca_drift_rmsd"] = ca_drift_rmsd
        summary[f"{prefix}_ca_drift_max"] = ca_drift_max
        summary[f"{prefix}_ca_drift_warning"] = ca_drift_warning
        summary[f"{prefix}_ca_drift_shape_mismatch"] = ca_drift_shape_mismatch

        # Status check: surface "stale failed summary" inconsistency
        # (artifacts on disk look done, but summary status says failed).
        # The status field is still authoritative for FAIL/PASS — we
        # just attach a pointer to the inconsistency in the message.
        files_look_done = rebuilt_pdb.is_file() and embedded_pdb.is_file()
        status_ok = (status == "done")
        status_msg = f"reconstruction status = {status!r}"
        if not status_ok and files_look_done:
            status_msg += (
                " (INCONSISTENT: rebuilt_fragment.pdb and embedded_receptor.pdb "
                "exist on disk; summary appears stale — re-run with "
                "--rerun-reconstruction to refresh)"
            )
        out.append(CheckResult(
            name=f"reconstruction[{receptor_label}].status",
            status=_PASS if status_ok else _FAIL,
            message=status_msg,
            details={
                "error": rec.get("error"),
                "rebuilt_fragment_pdb_exists": rebuilt_pdb.is_file(),
                "embedded_receptor_pdb_exists": embedded_pdb.is_file(),
                "stale_summary_suspect": (not status_ok) and files_look_done,
            },
        ))
        out.append(CheckResult(
            name=f"reconstruction[{receptor_label}].rebuilt_pdb",
            status=_PASS if rebuilt_pdb.is_file() else _FAIL,
            message=f"rebuilt_fragment.pdb {'exists' if rebuilt_pdb.is_file() else 'missing'}",
        ))
        out.append(CheckResult(
            name=f"reconstruction[{receptor_label}].embedded_pdb",
            status=_PASS if embedded_pdb.is_file() else _FAIL,
            message=f"embedded_receptor.pdb {'exists' if embedded_pdb.is_file() else 'missing'}",
        ))

        # CA drift fields
        if ca_drift_rmsd is None and ca_drift_warning is None:
            out.append(CheckResult(
                name=f"reconstruction[{receptor_label}].ca_drift_present",
                status=_WARN,
                message="ca_drift fields missing in reconstruction_summary.json",
            ))
        else:
            out.append(CheckResult(
                name=f"reconstruction[{receptor_label}].ca_drift",
                status=_PASS if ca_drift_warning is None else _WARN,
                message=(
                    f"ca_drift_rmsd={ca_drift_rmsd}, "
                    f"ca_drift_max={ca_drift_max}, "
                    f"warning={ca_drift_warning!r}"
                ),
            ))

        # ligand-removed check
        ligand_resname = (meta.get("ligand_resname") or "").strip().upper()
        embedded_ligand_present = False
        if embedded_pdb.is_file() and ligand_resname:
            with embedded_pdb.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if not line.startswith("HETATM"):
                        continue
                    if len(line) < 22:
                        continue
                    if line[17:20].strip().upper() == ligand_resname:
                        embedded_ligand_present = True
                        break
        summary[f"{prefix}_embedded_receptor_ligand_removed"] = (
            not embedded_ligand_present
        )
        out.append(CheckResult(
            name=f"reconstruction[{receptor_label}].ligand_removed",
            status=_PASS if not embedded_ligand_present else _FAIL,
            message=(
                f"native ligand resname {ligand_resname!r} not present in "
                f"embedded receptor"
                if not embedded_ligand_present else
                f"embedded receptor STILL contains HETATM resname "
                f"{ligand_resname!r} — fix embed remove_hetero / ligand removal"
            ),
        ))

        # anchor distance check (first/last residue CA between rebuilt and
        # input predicted CA PDB)
        anchor_first = anchor_last = None
        try:
            chain_id = (meta.get("chain_id") or "A")[:1]
            start = int(meta.get("start_resi"))
            end = int(meta.get("end_resi"))
            ca_pred = _read_ca_coords(
                Path(recon_meta.get("ca_pdb"))
                if recon_meta.get("ca_pdb") else
                task_dir / "postprocess" / "top1_ca.pdb"
            )
            ca_emb = _read_embedded_chain_ca(
                embedded_pdb, chain_id=chain_id,
                start_resi=start, end_resi=end,
            )
            if ca_pred is not None and ca_emb is not None:
                if len(ca_pred) >= 1 and len(ca_emb) >= 1:
                    anchor_first = _euclid(ca_pred[0], ca_emb[0])
                if len(ca_pred) >= 2 and len(ca_emb) >= 2:
                    anchor_last = _euclid(ca_pred[-1], ca_emb[-1])
        except Exception as e:
            summary[f"{prefix}_anchor_check_error"] = repr(e)

        summary[f"{prefix}_embed_anchor_first_dist"] = anchor_first
        summary[f"{prefix}_embed_anchor_last_dist"] = anchor_last

        if anchor_first is not None:
            ok = anchor_first <= self.anchor_dist_threshold_a
            out.append(CheckResult(
                name=f"reconstruction[{receptor_label}].anchor_first",
                status=_PASS if ok else _WARN,
                message=(
                    f"first-CA distance(predicted, embedded) = "
                    f"{anchor_first:.4f} Å"
                ),
            ))
        if anchor_last is not None:
            ok = anchor_last <= self.anchor_dist_threshold_a
            out.append(CheckResult(
                name=f"reconstruction[{receptor_label}].anchor_last",
                status=_PASS if ok else _WARN,
                message=(
                    f"last-CA distance(predicted, embedded) = "
                    f"{anchor_last:.4f} Å"
                ),
            ))

        summary[f"{prefix}_reconstruction_pass"] = bool(
            status == "done"
            and rebuilt_pdb.is_file()
            and embedded_pdb.is_file()
            and not embedded_ligand_present
        )
        return out

    # 4.9 docking ----------------------------------------------------- #
    def _check_docking(
        self,
        task_dir: Path,
        receptor_label: str,
        meta: Dict[str, Any],
        summary: Dict[str, Any],
    ) -> List[CheckResult]:
        dock_dir = task_dir / receptor_label / "docking"
        dock_p = dock_dir / "docking_summary.json"
        dock = _read_json(dock_p)
        prefix = "predicted" if receptor_label == "predicted_top1" else "oracle"

        if dock is None:
            summary[f"{prefix}_docking_pass"] = None
            return [CheckResult(
                name=f"docking[{receptor_label}].summary_present",
                status=_SKIP,
                message=f"{dock_p} missing",
            )]

        out: List[CheckResult] = []
        affinities = dock.get("affinities_kcal_mol") or []
        kds = dock.get("estimated_kd_m") or []
        mean_aff = _f(dock.get("mean_affinity_kcal_mol"))
        mean_kd = _f(dock.get("mean_kd_m"))

        ligand_pdb = dock_dir / "ligand_native.pdb"
        receptor_pdbqt = dock_dir / "receptor.pdbqt"
        ligand_pdbqt = dock_dir / "ligand.pdbqt"
        scores_csv = dock_dir / "docking_scores.csv"

        # native ligand atom count
        n_lig_atoms = 0
        if ligand_pdb.is_file():
            with ligand_pdb.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith(("ATOM", "HETATM")):
                        n_lig_atoms += 1

        box = (dock.get("metadata") or {}).get("box") or {}
        if box:
            summary["box_center"] = [
                box.get("center_x"), box.get("center_y"), box.get("center_z"),
            ]
            summary["box_size"] = [
                box.get("size_x"), box.get("size_y"), box.get("size_z"),
            ]
        summary["ligand_extraction_pass"] = bool(
            ligand_pdb.is_file() and n_lig_atoms > 0
        )
        summary[f"{prefix}_mean_affinity_kcal_mol"] = mean_aff
        summary[f"{prefix}_mean_kd_m"] = mean_kd
        summary[f"{prefix}_n_repeats"] = len(affinities)

        out.append(CheckResult(
            name=f"docking[{receptor_label}].ligand_native",
            status=_PASS if (ligand_pdb.is_file() and n_lig_atoms > 0) else _FAIL,
            message=(
                f"ligand_native.pdb atoms = {n_lig_atoms}"
                if ligand_pdb.is_file()
                else "ligand_native.pdb missing"
            ),
        ))
        out.append(CheckResult(
            name=f"docking[{receptor_label}].pdbqts",
            status=(
                _PASS if (receptor_pdbqt.is_file() and ligand_pdbqt.is_file())
                else _FAIL
            ),
            message=(
                f"receptor.pdbqt={'ok' if receptor_pdbqt.is_file() else 'MISSING'} "
                f"ligand.pdbqt={'ok' if ligand_pdbqt.is_file() else 'MISSING'}"
            ),
        ))
        out.append(CheckResult(
            name=f"docking[{receptor_label}].repeats_completed",
            status=_PASS if affinities else _FAIL,
            message=f"affinities_kcal_mol len = {len(affinities)}",
        ))
        out.append(CheckResult(
            name=f"docking[{receptor_label}].mean_affinity_present",
            status=_PASS if mean_aff is not None else _FAIL,
            message=f"mean_affinity_kcal_mol = {mean_aff}",
        ))
        out.append(CheckResult(
            name=f"docking[{receptor_label}].mean_kd_present",
            status=_PASS if mean_kd is not None else _FAIL,
            message=f"mean_kd_m = {mean_kd}",
        ))
        out.append(CheckResult(
            name=f"docking[{receptor_label}].scores_csv",
            status=_PASS if scores_csv.is_file() else _WARN,
            message=f"docking_scores.csv {'present' if scores_csv.is_file() else 'missing'}",
        ))
        summary[f"{prefix}_docking_pass"] = bool(
            ligand_pdb.is_file()
            and receptor_pdbqt.is_file()
            and ligand_pdbqt.is_file()
            and affinities
            and mean_aff is not None
            and mean_kd is not None
        )
        return out

    # cross-receptor Kd relationship -------------------------------- #
    def _check_kd_relationship(
        self, summary: Dict[str, Any],
    ) -> List[CheckResult]:
        pred_kd = _f(summary.get("predicted_mean_kd_m"))
        oracle_kd = _f(summary.get("oracle_mean_kd_m"))
        pred_aff = _f(summary.get("predicted_mean_affinity_kcal_mol"))
        oracle_aff = _f(summary.get("oracle_mean_affinity_kcal_mol"))

        if pred_kd is None or oracle_kd is None:
            summary["kd_ratio_pred_over_oracle"] = None
            summary["delta_affinity_pred_minus_oracle"] = (
                None if pred_aff is None or oracle_aff is None
                else float(pred_aff) - float(oracle_aff)
            )
            return [CheckResult(
                name="kd.ratio",
                status=_SKIP,
                message="kd_ratio not computable (one or both means missing)",
            )]

        ratio = (pred_kd / oracle_kd) if oracle_kd > 0 else None
        delta_aff = (
            float(pred_aff) - float(oracle_aff)
            if pred_aff is not None and oracle_aff is not None else None
        )
        summary["kd_ratio_pred_over_oracle"] = ratio
        summary["delta_affinity_pred_minus_oracle"] = delta_aff

        # Diagnostic only — never FAIL. WARN if predicted Kd is >> oracle Kd
        # (e.g., > 10× larger).
        out: List[CheckResult] = []
        if ratio is not None and ratio > 10.0:
            out.append(CheckResult(
                name="kd.ratio_pred_over_oracle",
                status=_WARN,
                message=(
                    f"predicted_mean_kd_m / oracle_mean_kd_m = "
                    f"{ratio:.3e} >> 1 — predicted pocket geometry "
                    "may be losing docking quality vs oracle-best"
                ),
            ))
        else:
            out.append(CheckResult(
                name="kd.ratio_pred_over_oracle",
                status=_PASS,
                message=(
                    f"predicted_mean_kd_m / oracle_mean_kd_m = "
                    f"{ratio:.3e}" if ratio is not None
                    else "ratio undefined (oracle_mean_kd_m=0)"
                ),
            ))
        return out


def _read_embedded_chain_ca(
    pdb: Path, chain_id: str, start_resi: int, end_resi: int,
) -> Optional[List[List[float]]]:
    """Return CA coordinates of `chain_id` in [start_resi, end_resi] order."""
    if not pdb.is_file():
        return None
    out: List[List[float]] = []
    seen: set = set()
    with pdb.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            if len(line) < 54:
                continue
            if line[12:16].strip() != "CA":
                continue
            if line[21:22] != chain_id:
                continue
            try:
                rseq = int(line[22:26].strip())
            except ValueError:
                continue
            if not (start_resi <= rseq <= end_resi):
                continue
            if rseq in seen:
                continue
            seen.add(rseq)
            try:
                out.append([
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                ])
            except ValueError:
                continue
    if not out:
        return None
    return out


__all__ = ["FinalPipelineValidator"]
