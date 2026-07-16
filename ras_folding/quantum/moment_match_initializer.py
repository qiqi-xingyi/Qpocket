# Author: Yuqi Zhang
"""Moment-Match Initializer — derive HEA parameters from V2 prior statistics.

This module implements Stage B: a closed-form (no VQE,
no iterative quantum optimisation) computation of θ for the
``hea_with_tf`` ansatz, given:
  - V2 corridor + environment prior (when available), or
  - a random + validity-filter fallback (for synthetic cases without env)

Algorithm
---------
1. Sample K bitstrings from V2 prior (or fallback). All samples are
   env-consistent if V2 is available; only intra-fragment-valid if
   using the fallback.

2. Empirical statistics:
   - 1-pt marginals:  p_q = mean(z_q) over valid samples
   - 2-pt correlations C_{q,q'} = mean(z_q · z_{q'}) - p_q · p_{q'}
     restricted to the CX edges of the HEA topology
   - z* = argmin H_filter over the sampled bitstrings

3. Step 1 — First Ry layer (closed-form):
       θ_{0,q} = 2 · arcsin(√p_q^V2)
   Output state after this layer is a product state matching V2 marginals.

4. Step 2 — CX brick-wall (fixed, no parameters)

5. Step 3 — Second Ry layer (per-edge solve):
   For each CX edge (a, b) with CX_{a→b}, solve for (θ_{1,a}, θ_{1,b}) so
   that the 2-qubit reduced output marginals + correlation match the
   target (p_a^V2, p_b^V2, C_{a,b}^V2). For qubits shared between edges
   (most), a final pass averages the conflicting values.

The result is HEA θ with (R+1) * n_qubits values; for default R=1 this
is 2 * n_qubits.

Cost: ~seconds per task, dominated by V2 sampling. No iterative
quantum measurement / optimisation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ras_folding.encoder.decoder import decode_bitstring
from ras_folding.encoder.reachable import ReachableSet
from ras_folding.quantum.hea_ansatz import (
    cx_edges_bond_aware, n_parameters as hea_n_parameters,
)
from ras_folding.sampler.base_sampler import EncoderBaseSampler
from ras_folding.sampler.context import (
    SamplingContext, get_encoder_inputs, get_sequence,
)
from ras_folding.sampler.filter_hamiltonian import FilterHamiltonian
from ras_folding.sampler.sample_types import (
    CandidateSample, bitstring_str_to_int,
)
from ras_folding.sampler.validity import decode_and_validate


# ---------------------------------------------------------------------- #
# Config / result dataclasses                                            #
# ---------------------------------------------------------------------- #

@dataclass
class MomentMatchConfig:
    """Configuration for the moment-match initializer."""
    K_samples: int = 500                # V2 抽样数
    reps: int = 1                        # HEA 层数(目前只支持 1)
    seed: int = 2024
    use_v2_prior: bool = True            # 若 False 强制 fallback
    fallback_validity_filter: bool = True  # fallback 用 validity-filtered random
    correlation_clip: float = 0.49       # |C_{q,q'}| clip 防止数值边缘
    marginal_clip: float = 0.02          # p clip 到 [clip, 1-clip]


@dataclass
class MomentMatchResult:
    """Output of compute_theta()."""
    theta: np.ndarray                   # 全 HEA 参数 length = (R+1)*n_qubits
    p_q_v2: np.ndarray                  # 1-pt marginals (n_qubits,)
    correlations: Dict[Tuple[int, int], float]  # 2-pt only on CX edges
    z_star: int                          # lowest H_filter sample
    z_star_h_filter: float
    cx_edges: List[Tuple[int, int]]
    n_samples: int
    n_valid_samples: int
    sampling_mode: str                  # "v2_full" | "fallback_random"
    diagnostics: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------- #
# Per-edge 2-qubit solver (Step 3b)                                      #
# ---------------------------------------------------------------------- #

def _two_qubit_state_after_first_ry_cx(
    theta_a: float, theta_b: float, cx_direction: str = "a_to_b",
) -> np.ndarray:
    """Return the 4-component amplitude vector after
       Ry(θ_a) ⊗ Ry(θ_b) → CX_{a→b}  applied to |00⟩.

    Basis order: |b_a b_b⟩ = (00, 01, 10, 11).
    """
    ca, sa = np.cos(theta_a / 2.0), np.sin(theta_a / 2.0)
    cb, sb = np.cos(theta_b / 2.0), np.sin(theta_b / 2.0)
    # Initial product state amplitudes
    psi = np.array([ca * cb, ca * sb, sa * cb, sa * sb], dtype=np.float64)
    if cx_direction == "a_to_b":
        # CX_{a→b}: flip b when a=1, i.e. swap |10⟩ and |11⟩
        psi[2], psi[3] = psi[3], psi[2]
    elif cx_direction == "b_to_a":
        # CX_{b→a}: flip a when b=1, swap |01⟩ and |11⟩
        psi[1], psi[3] = psi[3], psi[1]
    else:
        raise ValueError(f"unknown cx_direction: {cx_direction}")
    return psi


def _apply_ry_to_qubit_in_2qubit_state(
    psi: np.ndarray, delta: float, qubit: int,
) -> np.ndarray:
    """Apply Ry(δ) to qubit ∈ {0=a, 1=b} of a 2-qubit state."""
    c, s = np.cos(delta / 2.0), np.sin(delta / 2.0)
    # |00⟩, |01⟩, |10⟩, |11⟩
    psi_new = np.zeros(4, dtype=np.float64)
    if qubit == 0:  # rotate qubit a
        # |00⟩ → c|00⟩ + s|10⟩
        # |01⟩ → c|01⟩ + s|11⟩
        # |10⟩ → -s|00⟩ + c|10⟩
        # |11⟩ → -s|01⟩ + c|11⟩
        psi_new[0] = c * psi[0] - s * psi[2]
        psi_new[1] = c * psi[1] - s * psi[3]
        psi_new[2] = s * psi[0] + c * psi[2]
        psi_new[3] = s * psi[1] + c * psi[3]
    else:  # rotate qubit b
        psi_new[0] = c * psi[0] - s * psi[1]
        psi_new[1] = s * psi[0] + c * psi[1]
        psi_new[2] = c * psi[2] - s * psi[3]
        psi_new[3] = s * psi[2] + c * psi[3]
    return psi_new


def _two_qubit_marginals_and_correlation(
    psi: np.ndarray,
) -> Tuple[float, float, float]:
    """From a 4-amplitude vector return (p_a, p_b, C_ab) where
       p_a = P(z_a = 1), p_b = P(z_b = 1), C_ab = ⟨z_a z_b⟩ - p_a p_b."""
    p = np.abs(psi) ** 2
    # P(00)=p[0], P(01)=p[1], P(10)=p[2], P(11)=p[3]
    p_a = float(p[2] + p[3])
    p_b = float(p[1] + p[3])
    p_ab = float(p[3])  # P(z_a=1, z_b=1)
    c_ab = p_ab - p_a * p_b
    return p_a, p_b, c_ab


def _solve_edge_thetas(
    theta_0_a: float, theta_0_b: float,
    target_p_a: float, target_p_b: float, target_C: float,
    cx_direction: str = "a_to_b",
) -> Tuple[float, float]:
    """For one CX edge, solve for (θ_{1,a}, θ_{1,b}) minimising
       (p_a - tgt)^2 + (p_b - tgt)^2 + (C - tgt)^2.

    Uses scipy.optimize.minimize (BFGS, < 50 iter, ~ms).
    """
    from scipy.optimize import minimize

    psi_post_cx = _two_qubit_state_after_first_ry_cx(
        theta_0_a, theta_0_b, cx_direction=cx_direction,
    )

    def loss(x):
        d_a, d_b = float(x[0]), float(x[1])
        psi = _apply_ry_to_qubit_in_2qubit_state(psi_post_cx, d_a, qubit=0)
        psi = _apply_ry_to_qubit_in_2qubit_state(psi, d_b, qubit=1)
        p_a, p_b, c_ab = _two_qubit_marginals_and_correlation(psi)
        # Sum of squared residuals
        return (
            (p_a - target_p_a) ** 2
            + (p_b - target_p_b) ** 2
            + 4.0 * (c_ab - target_C) ** 2  # weight correlation 4x
        )

    # Initial guess: zero (no extra rotation)
    res = minimize(
        loss, x0=np.array([0.0, 0.0]),
        method="L-BFGS-B",
        bounds=[(-np.pi, np.pi), (-np.pi, np.pi)],
        options={"maxiter": 50, "ftol": 1e-8},
    )
    return float(res.x[0]), float(res.x[1])


# ---------------------------------------------------------------------- #
# Main Initializer                                                        #
# ---------------------------------------------------------------------- #

class MomentMatchInitializer:
    """Compute HEA θ from V2 prior moment statistics (Stage B)."""

    def __init__(
        self,
        sampling_context: SamplingContext,
        filter_hamiltonian: FilterHamiltonian,
        config: MomentMatchConfig,
        env_ctx: Optional[Any] = None,
        corridor_ctx: Optional[Any] = None,
    ) -> None:
        self.ctx = sampling_context
        self.fh = filter_hamiltonian
        self.config = config
        self.env_ctx = env_ctx
        self.corridor_ctx = corridor_ctx

        self.encoder_inputs = get_encoder_inputs(sampling_context)
        self.n_bonds = int(self.encoder_inputs.n_bonds)
        self.n_qubits = self.n_bonds * 6
        self._reachable = ReachableSet(self.encoder_inputs, epsilon=1.0)

        if config.reps != 1:
            raise NotImplementedError(
                f"Only reps=1 is currently supported; got reps={config.reps}"
            )

    # ------------------------------------------------------------------ #
    def compute_theta(self) -> MomentMatchResult:
        """Main entry point."""
        # Stage A: sample bitstrings
        bitstrings, valid_count, mode = self._sample_bitstrings()
        if len(bitstrings) == 0:
            # Edge case: no valid samples — fall back to uniform
            theta = np.zeros(hea_n_parameters(self.n_qubits, self.config.reps))
            return MomentMatchResult(
                theta=theta, p_q_v2=np.full(self.n_qubits, 0.5),
                correlations={},
                z_star=0, z_star_h_filter=float("inf"),
                cx_edges=cx_edges_bond_aware(self.n_qubits, self.n_bonds),
                n_samples=self.config.K_samples,
                n_valid_samples=0,
                sampling_mode=mode,
                diagnostics={"warning": "no_valid_samples_uniform_fallback"},
            )

        # Stage A.2: empirical statistics
        p_q = self._compute_marginals(bitstrings)
        z_star, h_z_star = self._find_z_star(bitstrings)

        cx_edges = cx_edges_bond_aware(self.n_qubits, self.n_bonds)
        correlations = self._compute_pair_correlations(bitstrings, p_q, cx_edges)

        # Stage B Step 1: first Ry layer = V2 marginals (closed-form)
        p_clipped = np.clip(
            p_q, self.config.marginal_clip, 1.0 - self.config.marginal_clip,
        )
        theta_0 = 2.0 * np.arcsin(np.sqrt(p_clipped))

        # Stage B Step 3b: second Ry layer = per-edge solve
        theta_1 = self._solve_second_layer(theta_0, p_clipped, correlations,
                                            cx_edges)

        theta = np.concatenate([theta_0, theta_1])

        # Diagnostics
        diag: Dict[str, Any] = {
            "K_samples_drawn": int(self.config.K_samples),
            "n_valid_samples": int(valid_count),
            "n_cx_edges": int(len(cx_edges)),
            "p_q_min": float(p_q.min()),
            "p_q_max": float(p_q.max()),
            "p_q_mean": float(p_q.mean()),
            "max_abs_correlation": float(
                max((abs(v) for v in correlations.values()), default=0.0)
            ),
            "theta_0_norm": float(np.linalg.norm(theta_0)),
            "theta_1_norm": float(np.linalg.norm(theta_1)),
        }

        return MomentMatchResult(
            theta=theta,
            p_q_v2=p_q,
            correlations=correlations,
            z_star=z_star,
            z_star_h_filter=h_z_star,
            cx_edges=cx_edges,
            n_samples=self.config.K_samples,
            n_valid_samples=valid_count,
            sampling_mode=mode,
            diagnostics=diag,
        )

    # ------------------------------------------------------------------ #
    # Stage A: sampling                                                   #
    # ------------------------------------------------------------------ #
    def _sample_bitstrings(self) -> Tuple[List[int], int, str]:
        """Sample bitstrings via V2 (if available) or fallback."""
        # Try V2 if requested AND env+corridor available
        if (self.config.use_v2_prior
                and self.env_ctx is not None
                and self.corridor_ctx is not None):
            try:
                return self._sample_v2_full()
            except Exception as e:
                # Graceful degradation
                pass

        # Fallback: random + validity filter
        return self._sample_fallback_random()

    def _sample_v2_full(self) -> Tuple[List[int], int, str]:
        """V2 corridor + environment prior sampling."""
        from ras_folding.prior.prior_sampler import (
            PriorConditionedBaseSampler,
        )
        sampler = PriorConditionedBaseSampler(
            encoder_inputs=self.encoder_inputs,
            env_ctx=self.env_ctx,
            corridor_ctx=self.corridor_ctx,
            n_prior_samples=self.config.K_samples,
            seed=self.config.seed,
        )
        result = sampler.sample()
        bitstrings: List[int] = []
        for bs in result.valid_bitstrings:
            try:
                bitstrings.append(bitstring_str_to_int(bs))
            except Exception:
                continue
        return bitstrings, result.valid_count, "v2_full"

    def _sample_fallback_random(self) -> Tuple[List[int], int, str]:
        """V1-style random + validity filter (no env info)."""
        base = EncoderBaseSampler(
            n_samples=self.config.K_samples,
            seed=self.config.seed,
            mode="random_codes",
        )
        raw = base.sample(self.ctx)
        valid_bitstrings: List[int] = []
        for s in raw:
            if self.config.fallback_validity_filter:
                decode_and_validate(s, self.ctx)
                if not s.valid:
                    continue
            try:
                bs_int = bitstring_str_to_int(s.bitstring)
                valid_bitstrings.append(bs_int)
            except Exception:
                continue
        return valid_bitstrings, len(valid_bitstrings), "fallback_random"

    # ------------------------------------------------------------------ #
    # Stage A.2: empirical statistics                                    #
    # ------------------------------------------------------------------ #
    def _compute_marginals(self, bitstrings: List[int]) -> np.ndarray:
        """p_q = mean(z_q) over the bitstring set.

        Bit-order note: bitstrings are packed via ``bitstring_str_to_int``
        (i.e. ``int(bs_str, 2)``) with the decoder's MSB-first / bond-0-first
        convention, so the LEFTMOST character (qubit 0) is the HIGHEST bit
        of the int and qubit ``q`` sits at bit position ``n_qubits - 1 - q``.
        See ``ras_folding.quantum.counts_adapter`` for the counterpart on
        the measurement side.
        """
        K = len(bitstrings)
        if K == 0:
            return np.full(self.n_qubits, 0.5, dtype=np.float64)
        bs_arr = np.asarray(bitstrings, dtype=np.int64)
        counts = np.zeros(self.n_qubits, dtype=np.int64)
        for q in range(self.n_qubits):
            shift = self.n_qubits - 1 - q
            counts[q] = int(np.sum((bs_arr >> shift) & 1))
        return counts.astype(np.float64) / float(K)

    def _compute_pair_correlations(
        self, bitstrings: List[int], p_q: np.ndarray,
        cx_edges: List[Tuple[int, int]],
    ) -> Dict[Tuple[int, int], float]:
        """C_{q,q'} = mean(z_q · z_{q'}) - p_q · p_{q'} for CX edges only.

        Uses the same MSB-first packing as ``_compute_marginals``: qubit ``q``
        is at bit position ``n_qubits - 1 - q`` inside the packed int.
        """
        K = len(bitstrings)
        if K == 0:
            return {(a, b): 0.0 for (a, b) in cx_edges}
        bs_arr = np.asarray(bitstrings, dtype=np.int64)
        out: Dict[Tuple[int, int], float] = {}
        for (a, b) in cx_edges:
            ones_a = (bs_arr >> (self.n_qubits - 1 - a)) & 1
            ones_b = (bs_arr >> (self.n_qubits - 1 - b)) & 1
            p_ab = float(np.sum(ones_a * ones_b)) / float(K)
            c = p_ab - float(p_q[a]) * float(p_q[b])
            # Clip extreme values for numerical safety
            c = max(-self.config.correlation_clip,
                    min(self.config.correlation_clip, c))
            out[(int(a), int(b))] = c
        return out

    def _find_z_star(self, bitstrings: List[int]) -> Tuple[int, float]:
        """z* = argmin H_filter(decode(z)) over sampled bitstrings."""
        best_z = bitstrings[0]
        best_h = float("inf")
        for z in bitstrings:
            try:
                coords = decode_bitstring(z, self.encoder_inputs,
                                          reachable=self._reachable)
                fmt = f"0{self.n_qubits}b"
                s = CandidateSample(bitstring=format(z, fmt), coords=coords)
                e, _ = self.fh.evaluate(s, self.ctx)
                if e < best_h:
                    best_h = float(e)
                    best_z = int(z)
            except Exception:
                continue
        return best_z, best_h

    # ------------------------------------------------------------------ #
    # Stage B Step 3b: per-edge solver                                   #
    # ------------------------------------------------------------------ #
    def _solve_second_layer(
        self, theta_0: np.ndarray, p_q: np.ndarray,
        correlations: Dict[Tuple[int, int], float],
        cx_edges: List[Tuple[int, int]],
    ) -> np.ndarray:
        """Solve θ_{1,q} per qubit by averaging per-edge solutions.

        For each CX edge (a, b), solve for (δ_a, δ_b) locally
        (treating other qubits as isolated). Then for qubits shared
        across multiple edges, average the proposed δs.
        """
        theta_1_accum = np.zeros(self.n_qubits, dtype=np.float64)
        theta_1_count = np.zeros(self.n_qubits, dtype=np.int32)

        for (a, b) in cx_edges:
            try:
                delta_a, delta_b = _solve_edge_thetas(
                    theta_0_a=float(theta_0[a]),
                    theta_0_b=float(theta_0[b]),
                    target_p_a=float(p_q[a]),
                    target_p_b=float(p_q[b]),
                    target_C=float(correlations.get((a, b), 0.0)),
                    cx_direction="a_to_b",
                )
            except Exception:
                delta_a, delta_b = 0.0, 0.0
            theta_1_accum[a] += delta_a
            theta_1_count[a] += 1
            theta_1_accum[b] += delta_b
            theta_1_count[b] += 1

        # Average; qubits not in any edge stay 0
        with np.errstate(divide="ignore", invalid="ignore"):
            theta_1 = np.where(
                theta_1_count > 0,
                theta_1_accum / np.maximum(theta_1_count, 1),
                0.0,
            )
        return theta_1


# ---------------------------------------------------------------------- #
# Self-test                                                              #
# ---------------------------------------------------------------------- #

def _self_test() -> None:
    """Smoke test using a synthetic n_residues=3 case (fallback mode)."""
    import sys
    from pathlib import Path
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from tests._fixtures import make_test_case

    enc, ctx = make_test_case(3)
    fh = FilterHamiltonian(residue_contact_weights=None)
    cfg = MomentMatchConfig(K_samples=200, seed=42, use_v2_prior=False)
    initializer = MomentMatchInitializer(
        sampling_context=ctx, filter_hamiltonian=fh, config=cfg,
    )
    result = initializer.compute_theta()

    assert result.theta.shape == ((cfg.reps + 1) * enc.n_bonds * 6,), (
        f"theta shape mismatch: got {result.theta.shape}, "
        f"expected ({(cfg.reps + 1) * enc.n_bonds * 6},)"
    )
    assert np.all(np.isfinite(result.theta)), "theta contains NaN/inf"
    assert 0 <= result.p_q_v2.min() <= result.p_q_v2.max() <= 1, (
        "marginals out of [0, 1]"
    )
    assert result.sampling_mode in ("v2_full", "fallback_random")
    print(f"[moment_match] self-test PASSED")
    print(f"  sampling_mode      : {result.sampling_mode}")
    print(f"  n_valid_samples    : {result.n_valid_samples}")
    print(f"  theta.shape        : {result.theta.shape}")
    print(f"  p_q range          : [{result.p_q_v2.min():.3f}, "
          f"{result.p_q_v2.max():.3f}]")
    print(f"  max |correlation|  : "
          f"{result.diagnostics['max_abs_correlation']:.4f}")
    print(f"  z_star H_filter    : {result.z_star_h_filter:.4f}")


if __name__ == "__main__":
    _self_test()


__all__ = [
    "MomentMatchConfig",
    "MomentMatchResult",
    "MomentMatchInitializer",
]
