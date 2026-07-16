# Author: Yuqi Zhang
"""Prior diagnostics writer — task_dir/prior/ outputs.

Files emitted:
  prior/prior_config.json       — config snapshot
  prior/prior_summary.json      — aggregated metrics
  prior/prior_bit_marginals.csv — per-qubit P(bit=1)
  prior/prior_step_stats.csv    — per-bond aggregated stats
  prior/prior_sampled_paths.csv — bitstrings + diagnostic columns
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from ras_folding.prior.environment import EnvironmentPriorContext
from ras_folding.prior.corridor import CorridorPriorContext
from ras_folding.prior.prior_sampler import PriorSamplingResult


def write_prior_diagnostics(
    out_dir: Path,
    task_id: str,
    env_ctx: EnvironmentPriorContext,
    corridor_ctx: CorridorPriorContext,
    sampling_result: PriorSamplingResult,
    *,
    policy_config: Optional[Dict] = None,
) -> Dict[str, Path]:
    """Write all V2 prior diagnostics into out_dir/. Returns dict of paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "task_id": task_id,
        "prior_mode": "corridor_bezier",
        "n_prior_samples": sampling_result.sample_count,
        "seed": sampling_result.seed,
        "policy_config": policy_config or {},
        "corridor": corridor_ctx.to_serializable(),
        "environment": {
            "n_env_atoms": env_ctx.n_env_atoms,
            "n_ligand_atoms": env_ctx.n_ligand_atoms,
            "n_ligand_copies_found": env_ctx.n_ligand_copies_found,
            "ligand_selection_mode": env_ctx.ligand_selection_mode,
            "selected_ligand_resseq": env_ctx.selected_ligand_resseq,
            "selected_ligand_chain": env_ctx.selected_ligand_chain,
            "fragment_chain_id": env_ctx.fragment_chain_id,
            "fragment_start_resi": env_ctx.fragment_start_resi,
            "fragment_end_resi": env_ctx.fragment_end_resi,
            "pdb_path": env_ctx.pdb_path,
            "ligand_resname": env_ctx.metadata.get("ligand_resname"),
        },
    }
    (out_dir / "prior_config.json").write_text(
        json.dumps(config, indent=2)
    )

    # summary aggregates
    bm = sampling_result.bit_marginals
    valid_coords = sampling_result.valid_coords
    if valid_coords.size > 0:
        # path-level mean d_env (per CA position) and corridor distance
        # are derivable from step_stats; we put them into summary.
        step_stats = sampling_result.step_stats
        mean_d_env = float(np.nanmean([s["d_env_mean"]
                                          for s in step_stats]))
        mean_corr_score = float(np.nanmean([s["corridor_score_mean"]
                                               for s in step_stats]))
        mean_entropy = float(np.nanmean([s["entropy_mean"]
                                           for s in step_stats]))
        mean_max_prob = float(np.nanmean([s["max_prob_mean"]
                                            for s in step_stats]))
    else:
        mean_d_env = float("nan")
        mean_corr_score = float("nan")
        mean_entropy = float("nan")
        mean_max_prob = float("nan")

    summary = {
        "task_id": task_id,
        "prior_mode": "corridor_bezier",
        "n_prior_samples": sampling_result.sample_count,
        "valid_count": sampling_result.valid_count,
        "invalid_count": sampling_result.invalid_count,
        "valid_rate": sampling_result.path_stats.get("valid_rate"),
        "fallback_to_uniform": sampling_result.fallback_to_uniform,
        "invalid_reason_counts": sampling_result.invalid_reason_counts,
        "n_qubits": sampling_result.n_qubits,
        "n_bonds": sampling_result.path_stats.get("n_bonds"),
        "bit_probability_stats": {
            "min": float(bm.min()),
            "max": float(bm.max()),
            "mean": float(bm.mean()),
            "median": float(np.median(bm)),
        },
        "mean_d_env_sampled": mean_d_env,
        "mean_corridor_score_sampled": mean_corr_score,
        "mean_entropy": mean_entropy,
        "mean_max_probability": mean_max_prob,
        "corridor_mode": corridor_ctx.corridor_mode,
        "ligand_weight_effective": corridor_ctx.ligand_weight_effective,
        "d_lig_anchor_line": corridor_ctx.d_lig_anchor_line,
    }
    (out_dir / "prior_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    # bit marginals csv
    with (out_dir / "prior_bit_marginals.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["qubit_index", "bond_index", "bit_offset", "p_bit_one"])
        n_q = sampling_result.n_qubits
        bps = 6  # BITS_PER_BOND
        for q in range(n_q):
            bond_idx = q // bps
            bit_off = q % bps
            w.writerow([q, bond_idx, bit_off, float(bm[q])])

    # step stats csv
    with (out_dir / "prior_step_stats.csv").open(
            "w", newline="", encoding="utf-8") as f:
        if sampling_result.step_stats:
            keys = list(sampling_result.step_stats[0].keys())
            w = csv.DictWriter(f, fieldnames=["task_id"] + keys)
            w.writeheader()
            for r in sampling_result.step_stats:
                row = {"task_id": task_id}
                row.update(r)
                w.writerow(row)
        else:
            f.write("task_id,step_index\n")

    # sampled paths csv (bitstrings only — coords go into ensemble dir)
    with (out_dir / "prior_sampled_paths.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample_index", "bitstring", "is_valid"])
        for i, bs in enumerate(sampling_result.valid_bitstrings):
            w.writerow([i, bs, True])

    return {
        "config": out_dir / "prior_config.json",
        "summary": out_dir / "prior_summary.json",
        "bit_marginals": out_dir / "prior_bit_marginals.csv",
        "step_stats": out_dir / "prior_step_stats.csv",
        "sampled_paths": out_dir / "prior_sampled_paths.csv",
    }


__all__ = ["write_prior_diagnostics"]
