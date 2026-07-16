# Author: Yuqi Zhang
"""Validation summary + markdown report writers."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from pipeline_validation.types import CheckResult, ValidationResult


# ---------------------------------------------------------------------- #
# tuning advice based on summary fields                                  #
# ---------------------------------------------------------------------- #

def _tuning_advice(summary: Dict[str, Any]) -> List[str]:
    advice: List[str] = []

    valid_rate = summary.get("valid_rate")
    fallback_rate = summary.get("fallback_rate")

    if isinstance(valid_rate, (int, float)) and float(valid_rate) == 0.0:
        advice.append(
            "valid_rate == 0 — check bit-order in counts adapter, decoder "
            "instrumentation, and that the IBM backend actually returned "
            "bitstrings of the expected length."
        )
    if isinstance(fallback_rate, (int, float)) and float(fallback_rate) > 0.95:
        advice.append(
            f"fallback_rate = {fallback_rate:.4f} > 0.95 — increase "
            "moment_match_K (V2 prior samples), raise n_circuits / shots, "
            "or loosen anchor / endpoint constraints."
        )

    n_kept = summary.get("n_children_kept")
    if n_kept is not None and int(n_kept) == 0 and (
        summary.get("densify_pass") is not None
    ):
        advice.append(
            "n_children_kept == 0 — lower angular_sigmas_deg, raise "
            "endpoint_tolerance, or relax max_local_rmsd; densify "
            "currently rejects every child."
        )

    dense_frac = summary.get("dense_fraction_in_subspace")
    if isinstance(dense_frac, (int, float)) and float(dense_frac) > 0.6:
        advice.append(
            f"dense_fraction_in_subspace = {dense_frac:.3f} > 0.6 — "
            "verify the SQD dense-fraction cap actually fires "
            "(check refinement_summary.dense_fraction_cap_applied)."
        )

    pred_rmsd = summary.get("predicted_top1_rmsd")
    oracle_rmsd = summary.get("oracle_best_rmsd")
    if (
        isinstance(pred_rmsd, (int, float))
        and isinstance(oracle_rmsd, (int, float))
        and float(pred_rmsd) - float(oracle_rmsd) > 1.0
    ):
        advice.append(
            f"oracle_best_rmsd ({oracle_rmsd:.3f}) is much smaller than "
            f"predicted_top1_rmsd ({pred_rmsd:.3f}) — review basin "
            "ranking and basin_weight_bonus in PredictionPostProcessor; "
            "the post-process is losing the best basin."
        )

    for prefix in ("predicted", "oracle"):
        warn = summary.get(f"{prefix}_ca_drift_warning")
        if warn:
            advice.append(
                f"{prefix} CA drift warning: {warn}. Check the installed "
                "PULCHRA build; if drift is real, add a Kabsch correction "
                "before embed."
            )

    for prefix in ("predicted", "oracle"):
        a_first = summary.get(f"{prefix}_embed_anchor_first_dist")
        a_last = summary.get(f"{prefix}_embed_anchor_last_dist")
        for label, val in (("first", a_first), ("last", a_last)):
            if isinstance(val, (int, float)) and float(val) > 1.0:
                advice.append(
                    f"{prefix} embed anchor {label}-CA drift = {val:.3f} Å "
                    "(slight endpoint mismatch; not fatal unless much larger "
                    "than the 1.0 Å threshold). Likely cause: PULCHRA local "
                    "geometry reconstruction or oracle dense perturbation; "
                    "treat as a WARN, not a FAIL, unless multiple anchors "
                    "drift simultaneously."
                )

    for prefix in ("predicted", "oracle"):
        if summary.get(f"{prefix}_embedded_receptor_ligand_removed") is False:
            advice.append(
                f"{prefix} embedded receptor still contains the native "
                "ligand resname — fix embed remove_hetero / ligand_resname "
                "handling. THIS IS A FAIL, not a warning."
            )

    for prefix in ("predicted", "oracle"):
        if summary.get(f"{prefix}_docking_pass") is False:
            advice.append(
                f"{prefix} docking failed — check PULCHRA / OpenBabel / "
                "Vina availability, and confirm ligand extraction by "
                "ligand_resname succeeded."
            )

    ratio = summary.get("kd_ratio_pred_over_oracle")
    if isinstance(ratio, (int, float)) and float(ratio) > 10.0:
        advice.append(
            f"predicted_mean_kd_m / oracle_mean_kd_m = {ratio:.3e} >> 1 "
            "— predicted pocket geometry is degrading docking quality "
            "noticeably vs oracle-best. Inspect predicted vs oracle "
            "embedded_receptor.pdb side-by-side."
        )

    return advice


# ---------------------------------------------------------------------- #
# writers                                                                #
# ---------------------------------------------------------------------- #

def write_validation_summary(
    output_dir: Path,
    result: ValidationResult,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    p = output_dir / "validation_summary.json"
    payload = {
        "task_id": result.task_id,
        "passed": bool(result.passed),
        "warnings": int(result.warnings),
        "failures": int(result.failures),
        "checks": [asdict(c) for c in result.checks],
        "summary": dict(result.summary),
    }
    p.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return p


def write_validation_report(
    output_dir: Path,
    result: ValidationResult,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    p = output_dir / "validation_report.md"

    s = result.summary
    lines: List[str] = []
    lines.append(f"# Final pipeline validation — `{result.task_id}`")
    lines.append("")
    lines.append(
        f"- passed: **{result.passed}**  "
        f"(failures: {result.failures}, warnings: {result.warnings})"
    )
    lines.append("")

    # Headline metrics
    lines.append("## Headline metrics")
    rows = [
        ("job_ids", s.get("job_ids")),
        ("total_shots", s.get("total_shots")),
        ("unique_bitstrings", s.get("unique_bitstrings")),
        ("n_wrong_length_bitstrings", s.get("n_wrong_length_bitstrings")),
        ("n_raw_total", s.get("n_raw_total")),
        ("n_valid_total", s.get("n_valid_total")),
        ("n_accepted_total", s.get("n_accepted_total")),
        ("valid_rate", s.get("valid_rate")),
        ("fallback_rate", s.get("fallback_rate")),
        ("favorable_contact_miss_mean", s.get("favorable_contact_miss_mean")),
        ("dense_fraction_in_subspace", s.get("dense_fraction_in_subspace")),
        ("dense_cap_applied", s.get("dense_cap_applied")),
        ("n_basins", s.get("n_basins")),
        ("top_basin_weight", s.get("top_basin_weight")),
        ("top1_bitstring", s.get("top1_bitstring")),
        ("top1_full_energy", s.get("top1_full_energy")),
        ("top1_refined_score", s.get("top1_refined_score")),
        ("top1_is_dense", s.get("top1_is_dense")),
        ("predicted_top1_rmsd", s.get("predicted_top1_rmsd")),
        ("oracle_best_rmsd", s.get("oracle_best_rmsd")),
        ("oracle_best_source", s.get("oracle_best_source")),
        ("oracle_best_is_dense", s.get("oracle_best_is_dense")),
        ("delta_rmsd_pred_minus_oracle", s.get("delta_rmsd_pred_minus_oracle")),
        ("predicted_ca_drift_rmsd", s.get("predicted_ca_drift_rmsd")),
        ("predicted_ca_drift_max", s.get("predicted_ca_drift_max")),
        ("oracle_ca_drift_rmsd", s.get("oracle_ca_drift_rmsd")),
        ("oracle_ca_drift_max", s.get("oracle_ca_drift_max")),
        ("predicted_embedded_receptor_ligand_removed",
            s.get("predicted_embedded_receptor_ligand_removed")),
        ("oracle_embedded_receptor_ligand_removed",
            s.get("oracle_embedded_receptor_ligand_removed")),
        ("predicted_embed_anchor_first_dist",
            s.get("predicted_embed_anchor_first_dist")),
        ("predicted_embed_anchor_last_dist",
            s.get("predicted_embed_anchor_last_dist")),
        ("oracle_embed_anchor_first_dist",
            s.get("oracle_embed_anchor_first_dist")),
        ("oracle_embed_anchor_last_dist",
            s.get("oracle_embed_anchor_last_dist")),
        ("predicted_mean_affinity_kcal_mol",
            s.get("predicted_mean_affinity_kcal_mol")),
        ("oracle_mean_affinity_kcal_mol",
            s.get("oracle_mean_affinity_kcal_mol")),
        ("predicted_mean_kd_m", s.get("predicted_mean_kd_m")),
        ("oracle_mean_kd_m", s.get("oracle_mean_kd_m")),
        ("kd_ratio_pred_over_oracle", s.get("kd_ratio_pred_over_oracle")),
        ("delta_affinity_pred_minus_oracle",
            s.get("delta_affinity_pred_minus_oracle")),
    ]
    for k, v in rows:
        lines.append(f"- {k}: {v}")
    lines.append("")

    # Checks
    lines.append("## Checks")
    lines.append("| status | name | message |")
    lines.append("|---|---|---|")
    for c in result.checks:
        msg = (c.message or "").replace("|", "\\|")
        lines.append(f"| {c.status} | `{c.name}` | {msg} |")
    lines.append("")

    # Tuning advice
    advice = _tuning_advice(s)
    lines.append("## Tuning advice")
    if not advice:
        lines.append("- (no actionable tuning advice — all checks within tolerance)")
    else:
        for a in advice:
            lines.append(f"- {a}")
    lines.append("")

    p.write_text("\n".join(lines), encoding="utf-8")
    return p


__all__ = ["write_validation_summary", "write_validation_report"]
