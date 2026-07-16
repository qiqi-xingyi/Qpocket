# Author: Yuqi Zhang
"""Landscape reconstruction — PCA projection + (KDE) free-energy summary
of the post-refinement candidate ensemble.

We do NOT introduce any new sampling / scoring algorithm. This module
only interprets the existing candidate set:
  - flatten coords → vector
  - PCA(2)
  - per-candidate row in projection.csv
  - KDE-based free-energy on a 2-D grid (scipy.stats.gaussian_kde)
  - basin-level aggregates (mean PC1/PC2)
  - funnel metrics if reference_coords is supplied; else spread metrics.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ras_folding.refinement.refined_types import RefinedCandidate


_EPS = 1e-12


@dataclass
class LandscapeResult:
    projection_csv: Optional[Path] = None
    basin_summary_csv: Optional[Path] = None
    landscape_summary_json: Optional[Path] = None
    plot_files: Dict[str, str] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------- #
# helpers                                                                #
# ---------------------------------------------------------------------- #

def _is_dense(sample) -> bool:
    return bool(
        (sample.metadata.get("densify") or {}).get("is_perturbed", False)
    )


def _ca_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    diffs = a - b
    return float(math.sqrt(max(float(np.mean(np.sum(diffs * diffs, axis=1))), 0.0)))


def _pca_2d(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (proj_2d, mean_vec, principal_axes_2x_len). Use sklearn if
    available, otherwise plain SVD."""
    if X.shape[0] == 0:
        return np.zeros((0, 2)), np.zeros(X.shape[1]), np.zeros((0,))
    mean = X.mean(axis=0)
    Xc = X - mean
    try:
        from sklearn.decomposition import PCA   # type: ignore
        n_components = min(2, Xc.shape[1], Xc.shape[0])
        pca = PCA(n_components=n_components)
        proj = pca.fit_transform(Xc)
        if proj.shape[1] < 2:
            pad = np.zeros((proj.shape[0], 2 - proj.shape[1]))
            proj = np.concatenate([proj, pad], axis=1)
        return proj.astype(np.float64), mean, pca.explained_variance_ratio_
    except Exception:
        # numpy SVD fallback
        try:
            U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
            proj = (U[:, :2] * S[:2])
            if proj.shape[1] < 2:
                pad = np.zeros((proj.shape[0], 2 - proj.shape[1]))
                proj = np.concatenate([proj, pad], axis=1)
            total_var = float(np.sum(S * S))
            ev_ratio = (S[:2] * S[:2]) / (total_var + _EPS) if total_var > 0 else np.zeros(2)
            return proj.astype(np.float64), mean, ev_ratio
        except Exception:
            return np.zeros((Xc.shape[0], 2)), mean, np.zeros(2)


def _free_energy_grid(
    pc1: np.ndarray, pc2: np.ndarray, weights: np.ndarray,
    *, grid_size: int, bandwidth: Optional[float],
) -> Optional[Dict[str, Any]]:
    """Compute a KDE-based free-energy grid. Returns None if scipy is
    unavailable or the input is degenerate."""
    if pc1.size < 2:
        return None
    try:
        from scipy.stats import gaussian_kde   # type: ignore
    except Exception:
        return None
    pts = np.vstack([pc1, pc2])
    w = np.maximum(weights, 0.0)
    if float(w.sum()) <= 0:
        w = np.ones_like(w)
    try:
        if bandwidth is None:
            kde = gaussian_kde(pts, weights=w)
        else:
            kde = gaussian_kde(pts, bw_method=float(bandwidth), weights=w)
    except Exception:
        return None
    pad = 0.1
    x_min, x_max = float(pc1.min()), float(pc1.max())
    y_min, y_max = float(pc2.min()), float(pc2.max())
    x_pad = (x_max - x_min) * pad if x_max > x_min else 1.0
    y_pad = (y_max - y_min) * pad if y_max > y_min else 1.0
    xs = np.linspace(x_min - x_pad, x_max + x_pad, int(grid_size))
    ys = np.linspace(y_min - y_pad, y_max + y_pad, int(grid_size))
    XX, YY = np.meshgrid(xs, ys)
    coords = np.vstack([XX.ravel(), YY.ravel()])
    try:
        Z = kde(coords).reshape(XX.shape)
    except Exception:
        return None
    free_energy = -np.log(Z + _EPS)
    return {
        "x_grid": xs.tolist(),
        "y_grid": ys.tolist(),
        "density_min": float(Z.min()),
        "density_max": float(Z.max()),
        "free_energy_min": float(free_energy.min()),
        "free_energy_max": float(free_energy.max()),
        "grid_size": int(grid_size),
    }


# ---------------------------------------------------------------------- #
# main class                                                             #
# ---------------------------------------------------------------------- #

class LandscapeReconstructor:
    def __init__(
        self,
        n_components: int = 2,
        use_kabsch: bool = False,
        energy_field: str = "full_energy",
        weight_field: str = "refined_weight",
        grid_size: int = 100,
        kde_bandwidth: Optional[float] = None,
        plot_enabled: bool = True,
    ) -> None:
        if n_components != 2:
            raise ValueError("only n_components=2 is supported in this version")
        self.n_components = int(n_components)
        self.use_kabsch = bool(use_kabsch)
        self.energy_field = str(energy_field)
        self.weight_field = str(weight_field)
        self.grid_size = int(grid_size)
        self.kde_bandwidth = kde_bandwidth
        self.plot_enabled = bool(plot_enabled)

    # ------------------------------------------------------------------ #
    def reconstruct(
        self,
        candidates: List[RefinedCandidate],
        output_dir: Path,
        reference_coords: Optional[np.ndarray] = None,
    ) -> LandscapeResult:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        result = LandscapeResult()

        if not candidates:
            result.summary = {"n_candidates": 0, "note": "empty"}
            return result

        N = len(candidates)
        L = candidates[0].sample.coords.shape[0]
        # Build coordinate matrix (N, 3L). use_kabsch=False per default.
        X = np.zeros((N, 3 * L), dtype=np.float64)
        for i, c in enumerate(candidates):
            X[i, :] = np.asarray(c.sample.coords, dtype=np.float64).ravel()

        # ---- PCA -------------------------------------------------------
        proj, mean_vec, ev_ratio = _pca_2d(X)
        pc1 = proj[:, 0] if proj.shape[1] >= 1 else np.zeros(N)
        pc2 = proj[:, 1] if proj.shape[1] >= 2 else np.zeros(N)

        # ---- per-candidate fields -------------------------------------
        is_dense_arr = np.array([_is_dense(c.sample) for c in candidates])
        full_e = np.array(
            [
                float(c.sample.full_energy)
                if c.sample.full_energy is not None else np.nan
                for c in candidates
            ],
            dtype=np.float64,
        )
        weights = np.array(
            [float(c.refined_weight) for c in candidates],
            dtype=np.float64,
        )
        refined_scores = np.array(
            [float(c.refined_score) for c in candidates],
            dtype=np.float64,
        )
        filter_e = np.array(
            [
                float(c.sample.filter_energy)
                if c.sample.filter_energy is not None else np.nan
                for c in candidates
            ],
            dtype=np.float64,
        )
        counts = np.array(
            [int(c.sample.count) for c in candidates], dtype=np.int64,
        )
        taus = [c.sample.metadata.get("tau") for c in candidates]

        rmsd_to_ref: Optional[np.ndarray] = None
        if reference_coords is not None:
            ref = np.asarray(reference_coords, dtype=np.float64)
            if ref.shape == (L, 3):
                rmsd_to_ref = np.array(
                    [_ca_rmsd(c.sample.coords, ref) for c in candidates],
                    dtype=np.float64,
                )

        # ---- projection.csv -------------------------------------------
        proj_csv = output_dir / "projection.csv"
        with proj_csv.open("w", encoding="utf-8") as fh:
            cols = [
                "candidate_index", "bitstring", "is_dense",
                "parent_bitstring", "pc1", "pc2",
                "full_energy", "refined_score", "refined_weight",
                "basin_id", "filter_energy", "tau", "count",
            ]
            if rmsd_to_ref is not None:
                cols.append("rmsd_to_reference")
            fh.write(",".join(cols) + "\n")
            for i, c in enumerate(candidates):
                d = c.sample.metadata.get("densify") or {}
                row = [
                    str(i),
                    str(c.sample.bitstring or ""),
                    "true" if is_dense_arr[i] else "false",
                    str(d.get("parent_bitstring") or ""),
                    f"{pc1[i]:.6f}",
                    f"{pc2[i]:.6f}",
                    "" if np.isnan(full_e[i]) else f"{full_e[i]:.6f}",
                    f"{refined_scores[i]:.6f}",
                    f"{weights[i]:.6f}",
                    "" if c.basin_id is None else str(int(c.basin_id)),
                    "" if np.isnan(filter_e[i]) else f"{filter_e[i]:.6f}",
                    "" if taus[i] is None else f"{float(taus[i]):.4f}",
                    str(counts[i]),
                ]
                if rmsd_to_ref is not None:
                    row.append(f"{rmsd_to_ref[i]:.6f}")
                fh.write(",".join(row) + "\n")
        result.projection_csv = proj_csv

        # ---- free-energy grid (optional) ------------------------------
        fe_grid = _free_energy_grid(
            pc1, pc2, weights,
            grid_size=self.grid_size, bandwidth=self.kde_bandwidth,
        )

        # ---- basin aggregates -----------------------------------------
        basin_rows: List[Dict[str, Any]] = []
        basin_ids = [
            c.basin_id if c.basin_id is not None else -1 for c in candidates
        ]
        unique_basins = sorted({b for b in basin_ids if b >= 0})
        for bid in unique_basins:
            members = [i for i, b in enumerate(basin_ids) if b == bid]
            if not members:
                continue
            ws = weights[members]
            es = full_e[members]
            rs = refined_scores[members]
            row = {
                "basin_id": int(bid),
                "basin_size": int(len(members)),
                "basin_weight": float(np.sum(ws)),
                "best_full_energy": (
                    None if np.all(np.isnan(es)) else float(np.nanmin(es))
                ),
                "best_refined_score": float(np.min(rs)),
                "mean_full_energy": (
                    None if np.all(np.isnan(es)) else float(np.nanmean(es))
                ),
                "mean_pc1": float(np.mean(pc1[members])),
                "mean_pc2": float(np.mean(pc2[members])),
                "representative_bitstring": (
                    candidates[members[int(np.argmin(rs))]].sample.bitstring
                ),
            }
            basin_rows.append(row)

        basin_csv = output_dir / "landscape_basin_summary.csv"
        with basin_csv.open("w", encoding="utf-8") as fh:
            cols = [
                "basin_id", "basin_size", "basin_weight",
                "best_full_energy", "best_refined_score",
                "mean_full_energy", "mean_pc1", "mean_pc2",
                "representative_bitstring", "is_top_basin",
            ]
            fh.write(",".join(cols) + "\n")
            # top basin: largest basin_weight
            top_idx = (
                int(np.argmax([r["basin_weight"] for r in basin_rows]))
                if basin_rows else -1
            )
            for j, r in enumerate(basin_rows):
                fh.write(",".join([
                    str(r["basin_id"]), str(r["basin_size"]),
                    f"{r['basin_weight']:.6f}",
                    "" if r["best_full_energy"] is None
                    else f"{r['best_full_energy']:.6f}",
                    f"{r['best_refined_score']:.6f}",
                    "" if r["mean_full_energy"] is None
                    else f"{r['mean_full_energy']:.6f}",
                    f"{r['mean_pc1']:.6f}", f"{r['mean_pc2']:.6f}",
                    str(r["representative_bitstring"] or ""),
                    "true" if j == top_idx else "false",
                ]) + "\n")
        result.basin_summary_csv = basin_csv

        # ---- summary --------------------------------------------------
        summary: Dict[str, Any] = {
            "n_candidates": int(N),
            "n_residues": int(L),
            "n_dense": int(np.sum(is_dense_arr)),
            "n_original": int(N - np.sum(is_dense_arr)),
            "pca_explained_variance_ratio": (
                ev_ratio.tolist() if hasattr(ev_ratio, "tolist") else list(ev_ratio)
            ),
            "energy_field": self.energy_field,
            "weight_field": self.weight_field,
            "use_kabsch": self.use_kabsch,
            "free_energy_grid": fe_grid,
            "n_basins": len(basin_rows),
        }
        # Funnel vs spread metrics
        if rmsd_to_ref is not None:
            valid_mask = ~np.isnan(full_e) & ~np.isnan(rmsd_to_ref)
            corr_e = (
                float(np.corrcoef(full_e[valid_mask], rmsd_to_ref[valid_mask])[0, 1])
                if valid_mask.sum() >= 2 and float(np.std(full_e[valid_mask])) > _EPS
                and float(np.std(rmsd_to_ref[valid_mask])) > _EPS
                else None
            )
            corr_s = (
                float(np.corrcoef(refined_scores[valid_mask], rmsd_to_ref[valid_mask])[0, 1])
                if valid_mask.sum() >= 2 and float(np.std(refined_scores[valid_mask])) > _EPS
                and float(np.std(rmsd_to_ref[valid_mask])) > _EPS
                else None
            )
            best_idx = int(np.argmin(refined_scores))
            top5_best_rmsd = (
                float(np.min(rmsd_to_ref[np.argsort(refined_scores)[:min(5, N)]]))
                if N >= 1 else None
            )
            summary["funnel"] = {
                "best_rmsd": float(np.min(rmsd_to_ref)),
                "top1_rmsd": float(rmsd_to_ref[best_idx]),
                "top5_best_rmsd": top5_best_rmsd,
                "corr_full_energy_rmsd": corr_e,
                "corr_refined_score_rmsd": corr_s,
                "near_native_count_1A": int(np.sum(rmsd_to_ref < 1.0)),
                "near_native_count_2A": int(np.sum(rmsd_to_ref < 2.0)),
                "near_native_count_3A": int(np.sum(rmsd_to_ref < 3.0)),
            }
        else:
            w_sum = float(np.sum(weights))
            if w_sum > _EPS:
                w_n = weights / w_sum
                entropy = float(-np.sum(w_n * np.log(w_n + _EPS)))
                eff_n = float(1.0 / np.sum(w_n * w_n))
            else:
                entropy = 0.0
                eff_n = 0.0
            summary["spread"] = {
                "energy_min": float(np.nanmin(full_e)) if np.any(~np.isnan(full_e)) else None,
                "energy_max": float(np.nanmax(full_e)) if np.any(~np.isnan(full_e)) else None,
                "energy_spread": (
                    float(np.nanmax(full_e) - np.nanmin(full_e))
                    if np.any(~np.isnan(full_e)) else None
                ),
                "refined_score_spread": float(np.max(refined_scores) - np.min(refined_scores)),
                "refined_weight_entropy": entropy,
                "effective_sample_size": eff_n,
                "top_basin_weight": (
                    float(max(r["basin_weight"] for r in basin_rows))
                    if basin_rows else 0.0
                ),
            }

        # ---- summary.json --------------------------------------------
        summary_path = output_dir / "landscape_summary.json"
        with summary_path.open("w", encoding="utf-8") as fh:
            json.dump(_coerce_json(summary), fh, indent=2)
        result.landscape_summary_json = summary_path
        result.summary = summary

        # ---- optional plot -------------------------------------------
        if self.plot_enabled:
            plot_path = self._maybe_plot(
                output_dir=output_dir,
                pc1=pc1, pc2=pc2,
                weights=weights,
                full_e=full_e,
                rmsd=rmsd_to_ref,
                is_dense=is_dense_arr,
            )
            if plot_path is not None:
                result.plot_files["scatter_pc"] = str(plot_path)

        return result

    # ------------------------------------------------------------------ #
    def _maybe_plot(
        self, *, output_dir: Path,
        pc1: np.ndarray, pc2: np.ndarray,
        weights: np.ndarray, full_e: np.ndarray,
        rmsd: Optional[np.ndarray],
        is_dense: np.ndarray,
    ) -> Optional[Path]:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt   # type: ignore
        except Exception:
            return None
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        fig, ax = plt.subplots(figsize=(6.0, 5.0), dpi=120)
        # color by full_energy if present, else weight
        if np.any(~np.isnan(full_e)):
            color = full_e
            cbar_label = "full_energy"
        else:
            color = weights
            cbar_label = "refined_weight"
        sc = ax.scatter(
            pc1, pc2, c=color, s=14, alpha=0.85, cmap="viridis",
            edgecolors=np.where(is_dense, "red", "none"),
            linewidths=0.6,
        )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("Landscape projection")
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(cbar_label)
        out = plots_dir / "landscape_pc12.png"
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)
        return out


# ---------------------------------------------------------------------- #
def _coerce_json(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _coerce_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_coerce_json(x) for x in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


__all__ = ["LandscapeReconstructor", "LandscapeResult"]
