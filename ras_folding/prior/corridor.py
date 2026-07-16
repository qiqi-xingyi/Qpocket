# Author: Yuqi Zhang
"""CorridorPriorBuilder — C4 ligand-anchor pocket Bézier corridor.

Constructs a deployable cubic Bézier path from anchor_left → anchor_right
with control points pulled toward (or near) the ligand centroid when
present, with auto-attenuation when the ligand sits far from the
anchor-line corridor (avoids misleading remote-allosteric fragments).

Validated by the V2 segment-corridor simulation (2026-04-30):
    - C4 alpha=0.3, w_lig=0.5, w_corridor=2.0 was best deployable
    - global median oracle Kabsch 2.41 Å vs S2 2.70 Å
    - safe on Frag-D/E (no degradation)
    - first frac_lt_2A > 0 cases on Frag-B/C
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from ras_folding.prior.environment import EnvironmentPriorContext


# Defaults match V2 simulation's best deployable config
DEFAULT_ALPHA: float = 0.3
DEFAULT_LIGAND_WEIGHT: float = 0.5
DEFAULT_LIGAND_FAR_THRESHOLD: float = 12.0
DEFAULT_LIGAND_FAR_ATTENUATION: float = 0.25
CA_LENGTH: float = 3.8

# --------------------------------------------------------------------- #
# Crystal-leakage control (added 2026-05-28)                            #
#                                                                       #
# The legacy V2 corridor injected the EXACT crystallographic ligand    #
# centroid into Bezier control points P1, P2 with weight 0.5 — this is #
# a ground-truth leak for any benchmark whose target is the bound      #
# pocket conformation. We replace this with a calibrated perturbation: #
# the model still "perceives" that a ligand exists and roughly where, #
# but no longer receives the exact crystal position.                   #
# --------------------------------------------------------------------- #
DEFAULT_CRYSTAL_LEAKAGE_MODE: str = "perturbed"   # was "full" historically
DEFAULT_PERTURBATION_SIGMA: float = 5.0   # Å, isotropic Gaussian
DEFAULT_PERTURBATION_CLIP: float = 2.0    # clip displacement to N·σ
_VALID_LEAKAGE_MODES = ("full", "perturbed", "anchor_only")


@dataclass
class CorridorPriorContext:
    P0: np.ndarray
    P1: np.ndarray
    P2: np.ndarray
    P3: np.ndarray
    alpha: float
    ligand_weight: float
    ligand_weight_effective: float
    ligand_far_threshold: float
    ligand_far_attenuation: float
    d_lig_anchor_line: Optional[float]
    corridor_mode: str  # "ligand_anchor_bezier" | "anchor_only_bezier"
                       # | "ligand_anchor_bezier_perturbed"
    n_bonds: int
    # Crystal-leakage diagnostics — record exactly what was used.
    crystal_leakage_mode: str = "full"        # back-compat default
    ligand_centroid_crystal: Optional[np.ndarray] = None  # raw, never used
    ligand_centroid_used: Optional[np.ndarray] = None     # what really drove P1,P2
    perturbation_offset: Optional[np.ndarray] = None      # (3,) Å
    perturbation_sigma: float = 0.0
    perturbation_seed: Optional[int] = None
    metadata: Dict[str, object] = field(default_factory=dict)

    def bezier_point(self, t):
        """Evaluate B(t). Accepts scalar or array; returns (..., 3)."""
        ts = np.asarray(t)
        scalar = ts.ndim == 0
        ts = ts.reshape(-1)
        one_minus = 1.0 - ts
        out = (
            (one_minus**3)[:, None] * self.P0[None, :]
            + (3 * one_minus**2 * ts)[:, None] * self.P1[None, :]
            + (3 * one_minus * ts**2)[:, None] * self.P2[None, :]
            + (ts**3)[:, None] * self.P3[None, :]
        )
        return out[0] if scalar else out

    def bezier_tangent(self, t):
        ts = np.asarray(t)
        scalar = ts.ndim == 0
        ts = ts.reshape(-1)
        one_minus = 1.0 - ts
        out = (
            3 * (one_minus**2)[:, None] * (self.P1 - self.P0)[None, :]
            + 6 * (one_minus * ts)[:, None] * (self.P2 - self.P1)[None, :]
            + 3 * (ts**2)[:, None] * (self.P3 - self.P2)[None, :]
        )
        return out[0] if scalar else out

    def distance_to_anchor_line(self, point: np.ndarray) -> float:
        """Perpendicular distance from `point` to segment P0–P3."""
        return float(_dist_to_segment(point, self.P0, self.P3))

    def to_serializable(self) -> Dict[str, object]:
        return {
            "P0": self.P0.tolist(),
            "P1": self.P1.tolist(),
            "P2": self.P2.tolist(),
            "P3": self.P3.tolist(),
            "alpha": self.alpha,
            "ligand_weight": self.ligand_weight,
            "ligand_weight_effective": self.ligand_weight_effective,
            "ligand_far_threshold": self.ligand_far_threshold,
            "ligand_far_attenuation": self.ligand_far_attenuation,
            "d_lig_anchor_line": self.d_lig_anchor_line,
            "corridor_mode": self.corridor_mode,
            "n_bonds": self.n_bonds,
            "crystal_leakage_mode": self.crystal_leakage_mode,
            "ligand_centroid_crystal":
                (self.ligand_centroid_crystal.tolist()
                 if self.ligand_centroid_crystal is not None else None),
            "ligand_centroid_used":
                (self.ligand_centroid_used.tolist()
                 if self.ligand_centroid_used is not None else None),
            "perturbation_offset":
                (self.perturbation_offset.tolist()
                 if self.perturbation_offset is not None else None),
            "perturbation_sigma": float(self.perturbation_sigma),
            "perturbation_seed":
                (int(self.perturbation_seed)
                 if self.perturbation_seed is not None else None),
            "metadata": dict(self.metadata),
        }


def _dist_to_segment(point: np.ndarray, a: np.ndarray, b: np.ndarray
                      ) -> float:
    ab = b - a
    L2 = float(np.dot(ab, ab))
    if L2 < 1e-12:
        return float(np.linalg.norm(point - a))
    t = float(np.dot(point - a, ab) / L2)
    t = max(0.0, min(1.0, t))
    proj = a + t * ab
    return float(np.linalg.norm(point - proj))


def _derive_perturbation_seed(task_id: Optional[str]) -> int:
    """Deterministic seed from task_id (so per-task perturbation is
    reproducible across runs, but different tasks get different offsets).

    Uses SHA-256 of the task identifier; takes the first 8 hex digits as
    an int. If task_id is None, returns a fixed sentinel seed (0) so
    behaviour is at least reproducible within the same session.
    """
    if task_id is None:
        return 0
    import hashlib
    h = hashlib.sha256(str(task_id).encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _perturb_centroid(
    centroid: np.ndarray,
    sigma: float,
    clip_n_sigma: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply an isotropic Gaussian perturbation to a 3-D point.

    The displacement is drawn from N(0, sigma^2 · I_3) and clipped so
    that ||offset|| ≤ clip_n_sigma · sigma.

    Parameters
    ----------
    centroid     : (3,) original centroid in Å
    sigma        : per-axis standard deviation in Å
    clip_n_sigma : maximum allowed displacement, in units of σ
    seed         : RNG seed (deterministic per task)

    Returns
    -------
    centroid_new : (3,) perturbed centroid
    offset       : (3,) the applied displacement
    """
    rng = np.random.default_rng(int(seed))
    offset = rng.normal(0.0, float(sigma), size=3)
    norm = float(np.linalg.norm(offset))
    max_norm = float(clip_n_sigma) * float(sigma)
    if norm > max_norm and max_norm > 0:
        offset = offset * (max_norm / norm)
    return centroid + offset, offset


def build_corridor_prior(
    encoder_inputs,
    env_ctx: EnvironmentPriorContext,
    *,
    alpha: float = DEFAULT_ALPHA,
    ligand_weight: float = DEFAULT_LIGAND_WEIGHT,
    ligand_far_threshold: float = DEFAULT_LIGAND_FAR_THRESHOLD,
    ligand_far_attenuation: float = DEFAULT_LIGAND_FAR_ATTENUATION,
    ca_length: float = CA_LENGTH,
    # Crystal-leakage control (NEW, default = de-leaked)
    crystal_leakage_mode: str = DEFAULT_CRYSTAL_LEAKAGE_MODE,
    perturbation_sigma: float = DEFAULT_PERTURBATION_SIGMA,
    perturbation_clip_n_sigma: float = DEFAULT_PERTURBATION_CLIP,
    task_id: Optional[str] = None,
) -> CorridorPriorContext:
    """Build the C4 corridor.

    Parameters
    ----------
    encoder_inputs : EncoderInputs (anchor_left, anchor_right,
        v_left_seed, v_right_seed, n_residues)
    env_ctx : EnvironmentPriorContext (provides ligand_centroid)
    alpha : control-point pull factor (push = alpha * n_bonds * 3.8)
    ligand_weight : nominal w_lig pulling P1/P2 toward ligand_centroid
    ligand_far_threshold : Å distance (lig_centroid → anchor_line) at
        which we attenuate w_lig
    ligand_far_attenuation : multiplier when far (e.g. 0.25)

    Crystal-leakage parameters
    --------------------------
    crystal_leakage_mode :
        - "full" : use the exact crystallographic ligand centroid
                   (LEGACY; leaks ground truth pocket position).
        - "perturbed" (default) : add a deterministic isotropic Gaussian
                   offset (per-task seed → reproducible) so the model
                   still perceives that a ligand is roughly in the pocket
                   but does NOT receive the exact crystal coordinates.
        - "anchor_only" : ignore the ligand entirely; reduces to the
                   anchor-only Bezier corridor.
    perturbation_sigma : per-axis Gaussian σ in Å for the "perturbed"
        mode (default 5.0 Å — calibrated to natural inter-PDB
        variability of same-class ligand centroids).
    perturbation_clip_n_sigma : displacement magnitude is clipped to
        N · σ to avoid pathological tails (default 2.0).
    task_id : task identifier used to derive a deterministic
        perturbation seed (so the same task always gets the same
        offset across re-runs, but distinct tasks get distinct
        offsets).
    """
    if crystal_leakage_mode not in _VALID_LEAKAGE_MODES:
        raise ValueError(
            f"crystal_leakage_mode must be one of {_VALID_LEAKAGE_MODES}; "
            f"got {crystal_leakage_mode!r}"
        )
    n_bonds = int(encoder_inputs.n_bonds)
    P0 = np.asarray(encoder_inputs.anchor_left, dtype=np.float64)
    P3 = np.asarray(encoder_inputs.anchor_right, dtype=np.float64)
    push = alpha * n_bonds * ca_length
    v_l = np.asarray(encoder_inputs.v_left_seed, dtype=np.float64)
    v_r = np.asarray(encoder_inputs.v_right_seed, dtype=np.float64)
    P1_seed = P0 + push * v_l
    P2_seed = P3 - push * v_r  # backward into chain from anchor_right
    raw_lig_centroid = env_ctx.ligand_centroid

    # --- decide what centroid (if any) drives the corridor ------------
    lig_centroid_used: Optional[np.ndarray] = None
    offset: Optional[np.ndarray] = None
    perturb_seed_used: Optional[int] = None
    if (raw_lig_centroid is None
            or crystal_leakage_mode == "anchor_only"):
        # Anchor-only path (no ligand info used)
        lig_centroid_used = None
    elif crystal_leakage_mode == "full":
        # Legacy: exact crystal centroid (leaks ground truth)
        lig_centroid_used = np.asarray(raw_lig_centroid, dtype=np.float64)
    elif crystal_leakage_mode == "perturbed":
        # De-leaked: deterministic isotropic Gaussian perturbation
        raw_arr = np.asarray(raw_lig_centroid, dtype=np.float64)
        perturb_seed_used = _derive_perturbation_seed(task_id)
        lig_centroid_used, offset = _perturb_centroid(
            raw_arr,
            sigma=perturbation_sigma,
            clip_n_sigma=perturbation_clip_n_sigma,
            seed=perturb_seed_used,
        )

    # --- if no ligand-driven centroid, return anchor-only corridor ---
    if lig_centroid_used is None:
        return CorridorPriorContext(
            P0=P0, P1=P1_seed, P2=P2_seed, P3=P3,
            alpha=float(alpha),
            ligand_weight=float(ligand_weight),
            ligand_weight_effective=0.0,
            ligand_far_threshold=float(ligand_far_threshold),
            ligand_far_attenuation=float(ligand_far_attenuation),
            d_lig_anchor_line=None,
            corridor_mode="anchor_only_bezier",
            n_bonds=n_bonds,
            crystal_leakage_mode=crystal_leakage_mode,
            ligand_centroid_crystal=(np.asarray(raw_lig_centroid, dtype=np.float64)
                                     if raw_lig_centroid is not None else None),
            ligand_centroid_used=None,
            perturbation_offset=None,
            perturbation_sigma=float(perturbation_sigma),
            perturbation_seed=perturb_seed_used,
        )

    # --- compute Bezier control points with the (possibly perturbed)
    # ligand centroid -----------------------------------------------------
    d_lig_line = _dist_to_segment(lig_centroid_used, P0, P3)
    if d_lig_line > ligand_far_threshold:
        w_eff = float(ligand_far_attenuation * ligand_weight)
    else:
        w_eff = float(ligand_weight)
    P1 = (1.0 - w_eff) * P1_seed + w_eff * lig_centroid_used
    P2 = (1.0 - w_eff) * P2_seed + w_eff * lig_centroid_used
    corridor_mode_str = (
        "ligand_anchor_bezier"
        if crystal_leakage_mode == "full"
        else "ligand_anchor_bezier_perturbed"
    )
    return CorridorPriorContext(
        P0=P0, P1=P1, P2=P2, P3=P3,
        alpha=float(alpha),
        ligand_weight=float(ligand_weight),
        ligand_weight_effective=w_eff,
        ligand_far_threshold=float(ligand_far_threshold),
        ligand_far_attenuation=float(ligand_far_attenuation),
        d_lig_anchor_line=float(d_lig_line),
        corridor_mode=corridor_mode_str,
        n_bonds=n_bonds,
        crystal_leakage_mode=crystal_leakage_mode,
        ligand_centroid_crystal=np.asarray(raw_lig_centroid, dtype=np.float64),
        ligand_centroid_used=lig_centroid_used,
        perturbation_offset=offset,
        perturbation_sigma=float(perturbation_sigma),
        perturbation_seed=perturb_seed_used,
    )


__all__ = [
    "DEFAULT_ALPHA", "DEFAULT_LIGAND_WEIGHT",
    "DEFAULT_LIGAND_FAR_THRESHOLD", "DEFAULT_LIGAND_FAR_ATTENUATION",
    "DEFAULT_CRYSTAL_LEAKAGE_MODE", "DEFAULT_PERTURBATION_SIGMA",
    "DEFAULT_PERTURBATION_CLIP",
    "CA_LENGTH",
    "CorridorPriorContext", "build_corridor_prior",
]
