# Author: Yuqi Zhang
"""SubspaceDiagonalizationRefiner — basin-aware refinement of accepted
valid candidates via diagonalization of the SQD-inspired effective
Hamiltonian.

Pipeline
--------
1. Collect eligible samples (accepted + valid + coords + full_energy)
   from one or more SampleBatches. Deduplicate by bitstring (counts
   accumulate, single representative kept per unique bitstring).
2. Select up to max_subspace_size by:
     - 40% lowest full_energy
     - 30% highest count (~ probability)
     - 30% diversity by farthest-point sampling in RMSD space
   The three buckets are filled in order; if a bucket is exhausted, its
   slack rolls into the next bucket.
3. Build off-diagonal coupling T (KNN RMSD kernel; refinement.coupling).
4. Build H_eff (refinement.effective_hamiltonian.build_effective_hamiltonian).
5. Diagonalize. dense eigh for N <= dense_cutoff (default 300); else
   scipy.sparse.linalg.eigsh for the lowest n_modes. eigsh failure
   falls back to dense eigh.
6. Compute mode Boltzmann weights, candidate refined_weight,
   refined_score, and rank.

The result is wrapped into a RefinementResult.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ras_folding.refinement.coupling import (
    pairwise_rmsd_matrix,
)
from ras_folding.refinement.effective_hamiltonian import (
    build_effective_hamiltonian,
)
from ras_folding.refinement.refined_types import (
    RefinedCandidate,
    RefinementResult,
)
from ras_folding.sampler.sample_types import CandidateSample, SampleBatch


_DENSE_CUTOFF = 300
_EPS_REFINED_WEIGHT_LOG = 1e-12


def _is_perturbed(sample) -> bool:
    """Return True if the sample carries a densify-perturbation flag."""
    d = sample.metadata.get("densify") or {}
    return bool(d.get("is_perturbed", False))


def _bitstring_to_int(bitstring: str) -> int:
    """Parse a candidate bitstring to int, tolerating the densify child tag.

    ``PerturbationDenseFiller._build_child`` appends a ``#dN`` sentinel to
    child bitstrings (see ras_folding/densify/dense_filler.py::_DENSE_CHILD_TAG)
    so dedup keeps children distinct from their parent. The character '#' is
    invalid in a binary string, so ``int(bs, 2)`` would raise on any dense
    child. Dense children share their parent's quantum state (only coords are
    perturbed), so for Pauli-Hamming coupling we strip the sentinel and parse
    the parent's binary bitstring.
    """
    base = bitstring.split("#", 1)[0]
    return int(base, 2)


class SubspaceDiagonalizationRefiner:
    def __init__(
        self,
        max_subspace_size: int = 1000,
        selection: str = "energy_probability_diversity",
        k_neighbors: int = 20,
        kappa: float = 0.2,
        sigma: Optional[float] = None,
        energy_normalization: str = "mad",
        n_modes: int = 10,
        beta_eff: float = 1.0,
        refined_score_alpha: float = 0.5,
        seed: Optional[int] = None,
        max_dense_fraction: Optional[float] = 0.6,
        min_original_fraction: Optional[float] = 0.3,
        # SQD off-diagonal coupling mode (only "hybrid" supported)
        coupling_mode: str = "hybrid",
        g_quantum: float = 0.03,
        alpha_pauli: float = 0.5,
        alpha_rmsd: float = 0.5,
        n_qubits: Optional[int] = None,
    ) -> None:
        if max_subspace_size <= 0:
            raise ValueError(
                f"max_subspace_size must be > 0, got {max_subspace_size}"
            )
        if selection not in ("energy_probability_diversity",):
            raise ValueError(f"unsupported selection {selection!r}")
        if energy_normalization not in ("mad", "zscore"):
            raise ValueError(
                f"unsupported energy_normalization {energy_normalization!r}"
            )
        if n_modes <= 0:
            raise ValueError(f"n_modes must be > 0, got {n_modes}")
        if k_neighbors < 0:
            raise ValueError(f"k_neighbors must be >= 0, got {k_neighbors}")
        if kappa < 0:
            raise ValueError(f"kappa must be >= 0, got {kappa}")
        if beta_eff < 0:
            raise ValueError(f"beta_eff must be >= 0, got {beta_eff}")
        if max_dense_fraction is not None and not (
            0.0 <= max_dense_fraction <= 1.0
        ):
            raise ValueError(
                f"max_dense_fraction must be in [0, 1] or None; "
                f"got {max_dense_fraction}"
            )
        if min_original_fraction is not None and not (
            0.0 <= min_original_fraction <= 1.0
        ):
            raise ValueError(
                f"min_original_fraction must be in [0, 1] or None; "
                f"got {min_original_fraction}"
            )

        self.max_subspace_size = int(max_subspace_size)
        self.selection = selection
        self.k_neighbors = int(k_neighbors)
        self.kappa = float(kappa)
        self.sigma = (None if sigma is None else float(sigma))
        self.energy_normalization = energy_normalization
        self.n_modes = int(n_modes)
        self.beta_eff = float(beta_eff)
        self.refined_score_alpha = float(refined_score_alpha)
        self.seed = seed
        self.max_dense_fraction = (
            None if max_dense_fraction is None else float(max_dense_fraction)
        )
        self.min_original_fraction = (
            None if min_original_fraction is None
            else float(min_original_fraction)
        )
        # coupling-mode config. full_pipline is single-flow: the
        # sole supported SQD off-diagonal coupling is the hybrid
        # Pauli ⊕ RMSD operator (hybrid_coupling_matrix), which requires a
        # transverse-field strength g_quantum > 0 and the qubit count.
        valid_modes = ("hybrid",)
        if coupling_mode not in valid_modes:
            raise ValueError(
                f"coupling_mode must be one of {valid_modes}; got {coupling_mode!r}"
            )
        if g_quantum <= 0:
            raise ValueError(
                f"coupling_mode='hybrid' requires g_quantum > 0; got {g_quantum}"
            )
        if n_qubits is None or n_qubits <= 0:
            raise ValueError(
                "coupling_mode='hybrid' requires n_qubits to be set"
            )
        self.coupling_mode = coupling_mode
        self.g_quantum = float(g_quantum)
        self.alpha_pauli = float(alpha_pauli)
        self.alpha_rmsd = float(alpha_rmsd)
        self.n_qubits = (int(n_qubits) if n_qubits is not None else None)

    # ------------------------------------------------------------------ #
    def refine(
        self,
        batches: Union[SampleBatch, List[SampleBatch]],
    ) -> RefinementResult:
        if isinstance(batches, SampleBatch):
            batches = [batches]
        eligible = self._collect_eligible(batches)

        if not eligible:
            empty_H = sp.csr_matrix((0, 0), dtype=np.float64)
            return RefinementResult(
                candidates=[],
                eigenvalues=np.zeros(0),
                eigenvectors=np.zeros((0, 0)),
                effective_hamiltonian=empty_H,
                selected_indices=[],
                parameters=self._params(),
                summary={
                    "n_eligible": 0,
                    "n_selected": 0,
                    "n_nonzero_couplings": 0,
                },
            )

        selected_idx, selection_meta = self._select_subspace(eligible)
        selected_idx, cap_meta = self._apply_dense_cap(
            selected_idx, eligible,
        )
        selected = [eligible[i] for i in selected_idx]

        coords_stack = np.stack(
            [s.coords for s in selected], axis=0,
        ).astype(np.float64, copy=False)
        energies = np.array(
            [s.full_energy for s in selected], dtype=np.float64,
        )

        # --- coupling matrix (hybrid Pauli ⊕ RMSD only) ----------
        # hybrid_coupling_matrix internally composes the exact Pauli
        # Hamming-1 matrix element (pauli_coupling.pauli_hamming1_matrix)
        # with the RMSD-Gaussian kernel (coupling.rmsd_kernel_coupling).
        from ras_folding.refinement.hybrid_coupling import (
            hybrid_coupling_matrix,
        )
        bitstrings = [
            _bitstring_to_int(s.bitstring) for s in selected if s.bitstring
        ]
        if len(bitstrings) != len(selected):
            raise RuntimeError(
                "hybrid coupling requires all selected samples to have "
                "a bitstring set"
            )
        T, hybrid_info = hybrid_coupling_matrix(
            bitstrings, coords_stack,
            g=self.g_quantum, n_qubits=self.n_qubits,
            alpha_pauli=self.alpha_pauli,
            alpha_rmsd=self.alpha_rmsd,
            k_neighbors_rmsd=self.k_neighbors,
            kappa_rmsd=self.kappa,
            sigma_rmsd=self.sigma,
        )
        coupling_info = {
            "mode": "hybrid",
            "n_nonzero": int(T.nnz),
            "sigma_used": hybrid_info["rmsd_info"].get("sigma_used"),
            **hybrid_info,
        }

        # --- effective hamiltonian ---------------------------------------
        H, E_norm, h_info = build_effective_hamiltonian(
            energies, T,
            energy_normalization=self.energy_normalization,
        )

        # --- diagonalize -------------------------------------------------
        eigvals, eigvecs, diag_info = self._diagonalize(H)

        # --- refined weights ---------------------------------------------
        eps_log = _EPS_REFINED_WEIGHT_LOG
        if eigvals.size == 0:
            mode_b = np.zeros(0)
        else:
            mode_b = np.exp(
                -self.beta_eff * (eigvals - float(eigvals[0]))
            )
        # mode_weights_per_candidate: (N, n_modes) where row i is
        #   mode_b * |v_k[i]|^2
        if eigvecs.size == 0:
            per_mode = np.zeros((len(selected), 0), dtype=np.float64)
        else:
            v2 = eigvecs * eigvecs                   # (N, n_modes)
            per_mode = v2 * mode_b[None, :]          # broadcasted

        w = per_mode.sum(axis=1)                     # (N,)
        w_sum = float(w.sum())
        if w_sum > 0:
            w_norm = w / w_sum
        else:
            w_norm = np.full_like(w, 1.0 / max(len(w), 1))

        refined_score = E_norm - self.refined_score_alpha * np.log(
            w_norm + eps_log
        )

        # --- pack into RefinedCandidate list -----------------------------
        order = np.argsort(refined_score)
        cand: List[RefinedCandidate] = []
        for rank, i in enumerate(order):
            cand.append(RefinedCandidate(
                sample=selected[int(i)],
                refined_weight=float(w_norm[int(i)]),
                refined_score=float(refined_score[int(i)]),
                mode_weights=per_mode[int(i)].copy(),
                basin_id=None,
                metadata={
                    "subspace_index": int(i),
                    "rank": int(rank),
                    "E_norm": float(E_norm[int(i)]),
                },
            ))

        # --- refinement diagnostics --------------------------------------
        diag = _refinement_diagnostics(
            energies=energies,
            refined_scores=refined_score,
            refined_weights=w_norm,
            coords_stack=coords_stack,
            n_nonzero_couplings=int(coupling_info.get("n_nonzero", 0)),
        )

        # --- dense / original split diagnostics --------------------------
        n_dense_eligible = sum(1 for s in eligible if _is_perturbed(s))
        n_original_eligible = len(eligible) - n_dense_eligible
        n_dense_selected = sum(1 for s in selected if _is_perturbed(s))
        dense_fraction_in_subspace = (
            float(n_dense_selected) / float(len(selected))
            if selected else 0.0
        )

        summary: Dict[str, Any] = {
            "n_eligible": len(eligible),
            "n_selected": len(selected),
            "n_nonzero_couplings": int(coupling_info.get("n_nonzero", 0)),
            "sigma_used": coupling_info.get("sigma_used"),
            "sigma": self.sigma,
            "kappa": self.kappa,
            "k_neighbors": self.k_neighbors,
            "energy_normalization_info": h_info["energy_normalization"],
            "diag_info": diag_info,
            "selection_info": selection_meta,
            "n_original_candidates": int(n_original_eligible),
            "n_dense_candidates": int(n_dense_eligible),
            "n_dense_selected_in_subspace": int(n_dense_selected),
            "dense_fraction_in_subspace": float(dense_fraction_in_subspace),
            "max_dense_fraction": self.max_dense_fraction,
            "min_original_fraction": self.min_original_fraction,
            **cap_meta,
            **diag,
        }

        return RefinementResult(
            candidates=cand,
            eigenvalues=eigvals,
            eigenvectors=eigvecs,
            effective_hamiltonian=H,
            selected_indices=list(selected_idx),
            parameters=self._params(),
            summary=summary,
        )

    # ------------------------------------------------------------------ #
    # internals                                                          #
    # ------------------------------------------------------------------ #
    def _params(self) -> Dict[str, Any]:
        return {
            "max_subspace_size": self.max_subspace_size,
            "selection": self.selection,
            "k_neighbors": self.k_neighbors,
            "kappa": self.kappa,
            "sigma": self.sigma,
            "energy_normalization": self.energy_normalization,
            "n_modes": self.n_modes,
            "beta_eff": self.beta_eff,
            "refined_score_alpha": self.refined_score_alpha,
            "seed": self.seed,
            "max_dense_fraction": self.max_dense_fraction,
            "min_original_fraction": self.min_original_fraction,
            # coupling-mode config
            "coupling_mode": self.coupling_mode,
            "g_quantum": self.g_quantum,
            "alpha_pauli": self.alpha_pauli,
            "alpha_rmsd": self.alpha_rmsd,
            "n_qubits": self.n_qubits,
        }

    # ------------------------------------------------------------------ #
    def _apply_dense_cap(
        self,
        selected_idx: List[int],
        eligible: List[CandidateSample],
    ) -> Tuple[List[int], Dict[str, Any]]:
        """Enforce max_dense_fraction / min_original_fraction on the
        already-selected subspace.

        Strategy (does NOT touch H_eff or refined_score):
          1. Detect dense vs original via metadata["densify"]["is_perturbed"].
          2. If either type is absent in the eligible pool → cap is a no-op.
          3. effective_cap = min(max_dense_fraction, 1 - min_original_fraction)
             (None values treated as 1.0 / 0.0 respectively).
          4. If current dense_fraction <= effective_cap → no-op.
          5. Compute n_dense_keep = floor(effective_cap / (1 - effective_cap)
                                          * n_orig_pool), capped at the
             currently-selected dense count.
             n_orig_pool = currently-selected originals + unselected
             eligible originals.
          6. n_orig_used >= ceil(n_dense_keep * (1 - effective_cap)
                                 / effective_cap), and <= n_orig_pool.
          7. Build new selected_idx:
             - keep first n_dense_keep dense from selected (preserves the
               40/30/30 ranking)
             - keep all currently-selected originals + best (by full_energy
               ASC) unselected originals to reach n_orig_used.
        """
        cap_meta: Dict[str, Any] = {
            "dense_fraction_cap_applied": False,
            "n_dense_removed_by_cap": 0,
            "n_original_forced_by_cap": 0,
            "dense_fraction_before_cap": None,
        }
        if not selected_idx:
            return selected_idx, cap_meta
        if (
            self.max_dense_fraction is None
            and self.min_original_fraction is None
        ):
            return selected_idx, cap_meta

        is_dense_eligible = [_is_perturbed(s) for s in eligible]
        n_dense_eligible = sum(is_dense_eligible)
        n_orig_eligible = len(eligible) - n_dense_eligible
        if n_dense_eligible == 0 or n_orig_eligible == 0:
            return selected_idx, cap_meta

        # Compute current selected composition.
        selected_dense = [
            i for i in selected_idx if is_dense_eligible[i]
        ]
        selected_orig = [
            i for i in selected_idx if not is_dense_eligible[i]
        ]
        n_dense_sel = len(selected_dense)
        n_total_sel = len(selected_idx)
        cur_dense_frac = float(n_dense_sel) / float(n_total_sel)
        cap_meta["dense_fraction_before_cap"] = cur_dense_frac

        # Effective cap.
        cap_max = (
            self.max_dense_fraction
            if self.max_dense_fraction is not None else 1.0
        )
        cap_complement = (
            (1.0 - self.min_original_fraction)
            if self.min_original_fraction is not None else 1.0
        )
        effective_cap = float(min(cap_max, cap_complement))
        if effective_cap >= 1.0:
            return selected_idx, cap_meta
        if cur_dense_frac <= effective_cap + 1e-12:
            return selected_idx, cap_meta

        # Pool: currently-selected originals + originals not yet selected.
        selected_set = set(selected_idx)
        unselected_originals = [
            i for i, s in enumerate(eligible)
            if (not is_dense_eligible[i]) and (i not in selected_set)
        ]
        unselected_originals.sort(
            key=lambda i: float(eligible[i].full_energy)
        )
        n_orig_pool = len(selected_orig) + len(unselected_originals)

        # n_dense_keep: largest dense count compatible with available
        # originals at the effective cap.
        if effective_cap == 0.0:
            n_dense_keep = 0
        else:
            ratio = effective_cap / (1.0 - effective_cap)
            n_dense_keep = int(np.floor(ratio * float(n_orig_pool)))
        n_dense_keep = min(n_dense_sel, n_dense_keep)

        # n_orig_used: as many as needed to satisfy the ratio, capped at
        # n_orig_pool and at max_subspace_size - n_dense_keep.
        if effective_cap == 0.0:
            n_orig_min = 0 if n_dense_keep == 0 else 1
        else:
            n_orig_min = int(np.ceil(
                n_dense_keep * (1.0 - effective_cap) / effective_cap
            ))
        n_orig_target = min(
            n_orig_pool,
            max(self.max_subspace_size - n_dense_keep, 0),
        )
        n_orig_used = min(n_orig_pool, max(n_orig_min, n_orig_target))

        # Compose new selection.
        chosen_dense = list(selected_dense[:n_dense_keep])
        chosen_orig = list(selected_orig)
        if n_orig_used > len(chosen_orig):
            need = n_orig_used - len(chosen_orig)
            chosen_orig.extend(unselected_originals[:need])
        # We never trim originals below current count (see proof in docstring).

        new_selected = chosen_dense + chosen_orig

        n_dense_removed = n_dense_sel - n_dense_keep
        n_original_forced = max(0, n_orig_used - len(selected_orig))
        cap_meta["dense_fraction_cap_applied"] = bool(n_dense_removed > 0)
        cap_meta["n_dense_removed_by_cap"] = int(n_dense_removed)
        cap_meta["n_original_forced_by_cap"] = int(n_original_forced)
        cap_meta["effective_dense_cap"] = float(effective_cap)
        return new_selected, cap_meta

    @staticmethod
    def _collect_eligible(
        batches: List[SampleBatch],
    ) -> List[CandidateSample]:
        """Pick samples that are accepted + valid + coords + full_energy.
        Deduplicate by bitstring with count accumulation.
        """
        seen: Dict[str, CandidateSample] = {}
        ordered_keys: List[str] = []
        for b in batches:
            for s in b.samples:
                if not s.is_eligible_for_refinement():
                    continue
                key = s.bitstring if s.bitstring is not None else str(id(s))
                if key in seen:
                    seen[key].count += s.count
                else:
                    seen[key] = s
                    ordered_keys.append(key)
        return [seen[k] for k in ordered_keys]

    def _select_subspace(
        self,
        eligible: List[CandidateSample],
    ) -> Tuple[List[int], Dict[str, Any]]:
        N = len(eligible)
        if N <= self.max_subspace_size:
            return list(range(N)), {
                "all_eligible_used": True,
                "n_total": N,
            }

        rng = np.random.default_rng(self.seed)
        target = self.max_subspace_size
        # split: 40 / 30 / 30
        n_energy = int(round(0.4 * target))
        n_count = int(round(0.3 * target))
        n_div = target - n_energy - n_count

        chosen: List[int] = []
        chosen_set: set = set()

        # 1) lowest full_energy ------------------------------------------
        e_arr = np.array(
            [s.full_energy for s in eligible], dtype=np.float64,
        )
        order_e = np.argsort(e_arr)
        for idx in order_e:
            if len(chosen) >= n_energy:
                break
            chosen.append(int(idx))
            chosen_set.add(int(idx))

        # 2) highest count (~probability) --------------------------------
        c_arr = np.array(
            [s.count for s in eligible], dtype=np.int64,
        )
        # break ties by lower energy
        sort_keys = list(zip(-c_arr, e_arr))
        order_c = sorted(range(N), key=lambda i: sort_keys[i])
        for idx in order_c:
            if len(chosen) >= n_energy + n_count:
                break
            if idx in chosen_set:
                continue
            chosen.append(int(idx))
            chosen_set.add(int(idx))

        # 3) diversity by farthest-point sampling in RMSD space ---------
        # Need RMSD against currently chosen members.
        if n_div > 0 and len(chosen) < target:
            remaining = [i for i in range(N) if i not in chosen_set]
            if remaining:
                coords_all = np.stack(
                    [s.coords for s in eligible], axis=0,
                ).astype(np.float64, copy=False)
                # min RMSD from each remaining to chosen set
                # initial distances:
                if chosen:
                    min_d = np.full(N, np.inf, dtype=np.float64)
                    for ci in chosen:
                        d_row = _rmsd_row(coords_all, ci)
                        min_d = np.minimum(min_d, d_row)
                else:
                    # bootstrap with a random pick if no anchors yet
                    seed_pick = int(rng.integers(0, N))
                    chosen.append(seed_pick)
                    chosen_set.add(seed_pick)
                    min_d = _rmsd_row(coords_all, seed_pick)

                while len(chosen) < target:
                    # mask out already-chosen
                    masked = min_d.copy()
                    masked[list(chosen_set)] = -np.inf
                    pick = int(np.argmax(masked))
                    if masked[pick] == -np.inf or not np.isfinite(
                        masked[pick]
                    ):
                        # No more meaningful candidates; break early.
                        break
                    chosen.append(pick)
                    chosen_set.add(pick)
                    d_new = _rmsd_row(coords_all, pick)
                    min_d = np.minimum(min_d, d_new)

        meta = {
            "n_total": N,
            "target": target,
            "n_energy_filled": min(len(chosen), n_energy),
            "n_count_filled": max(0, min(len(chosen) - n_energy, n_count)),
            "n_diversity_filled": max(
                0, len(chosen) - n_energy - n_count
            ),
        }
        return chosen, meta

    def _diagonalize(
        self, H: sp.csr_matrix,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        N = H.shape[0]
        info: Dict[str, Any] = {"N": N, "n_modes_requested": self.n_modes}

        if N == 0:
            return np.zeros(0), np.zeros((0, 0)), {**info, "method": "empty"}

        if N <= _DENSE_CUTOFF or self.n_modes >= N:
            H_dense = H.toarray() if sp.issparse(H) else np.asarray(H)
            # symmetrize to scrub fp noise
            H_dense = 0.5 * (H_dense + H_dense.T)
            w, V = np.linalg.eigh(H_dense)
            n_keep = min(self.n_modes, N)
            info["method"] = "dense_eigh"
            info["n_modes_kept"] = int(n_keep)
            return w[:n_keep], V[:, :n_keep], info

        # sparse path
        try:
            # eigsh requires k < N
            k = min(self.n_modes, N - 1)
            w, V = spla.eigsh(H, k=k, which="SA")
            order = np.argsort(w)
            w = w[order]
            V = V[:, order]
            info["method"] = "sparse_eigsh"
            info["n_modes_kept"] = int(w.size)
            return w, V, info
        except Exception as e:
            info["method"] = "dense_eigh_fallback"
            info["sparse_eigsh_exception"] = repr(e)
            H_dense = H.toarray()
            H_dense = 0.5 * (H_dense + H_dense.T)
            w, V = np.linalg.eigh(H_dense)
            n_keep = min(self.n_modes, N)
            info["n_modes_kept"] = int(n_keep)
            return w[:n_keep], V[:, :n_keep], info


def _refinement_diagnostics(
    *,
    energies: np.ndarray,
    refined_scores: np.ndarray,
    refined_weights: np.ndarray,
    coords_stack: np.ndarray,
    n_nonzero_couplings: int,
) -> Dict[str, Any]:
    """Compute summary diagnostics for the refined subspace."""
    eps = 1e-12
    N = int(energies.shape[0])
    out: Dict[str, Any] = {
        "spearman_full_energy_rank_vs_refined_rank": None,
        "top_k_overlap_energy_vs_refined": None,
        "top_k_overlap_k": None,
        "mean_pairwise_rmsd_top_energy": None,
        "mean_pairwise_rmsd_top_refined": None,
        "refined_weight_entropy": None,
        "coupling_density": None,
    }
    if N <= 1:
        out["coupling_density"] = 0.0 if N == 1 else None
        # entropy is well-defined for N=1 (=0).
        if N == 1:
            out["refined_weight_entropy"] = 0.0
        return out

    # spearman between full_energy ranks and refined_score ranks
    if (
        float(np.std(energies)) > eps
        and float(np.std(refined_scores)) > eps
    ):
        rx = np.argsort(np.argsort(energies)).astype(np.float64)
        ry = np.argsort(np.argsort(refined_scores)).astype(np.float64)
        if float(np.std(rx)) > eps and float(np.std(ry)) > eps:
            out["spearman_full_energy_rank_vs_refined_rank"] = float(
                np.corrcoef(rx, ry)[0, 1]
            )

    # top-k overlap (k = min(10, N))
    k = int(min(10, N))
    if k > 0:
        top_e = set(np.argsort(energies)[:k].tolist())
        top_r = set(np.argsort(refined_scores)[:k].tolist())
        out["top_k_overlap_k"] = k
        out["top_k_overlap_energy_vs_refined"] = (
            float(len(top_e & top_r)) / float(k)
        )

        # mean pairwise RMSD inside each top-k cluster
        if k >= 2:
            rmsd = pairwise_rmsd_matrix(coords_stack)
            out["mean_pairwise_rmsd_top_energy"] = _mean_pairwise(
                rmsd, sorted(top_e),
            )
            out["mean_pairwise_rmsd_top_refined"] = _mean_pairwise(
                rmsd, sorted(top_r),
            )

    # refined-weight entropy
    w = np.asarray(refined_weights, dtype=np.float64)
    if w.size > 0 and float(w.sum()) > eps:
        w_safe = w / float(w.sum())
        out["refined_weight_entropy"] = float(
            -np.sum(w_safe * np.log(w_safe + eps))
        )

    # coupling density: nnz / (N * (N - 1))
    out["coupling_density"] = float(n_nonzero_couplings) / float(N * (N - 1))
    return out


def _mean_pairwise(rmsd: np.ndarray, idx) -> Optional[float]:
    """Mean pairwise RMSD over the upper triangle of rmsd[idx][:, idx]."""
    sub = rmsd[np.ix_(idx, idx)]
    n = sub.shape[0]
    if n < 2:
        return None
    iu, ju = np.triu_indices(n, k=1)
    return float(np.mean(sub[iu, ju]))


# Required by _refinement_diagnostics — keep import here to avoid an
# unused import at module top level if upstream changes.
from ras_folding.refinement.coupling import pairwise_rmsd_matrix  # noqa: E402


def _rmsd_row(coords_all: np.ndarray, i: int) -> np.ndarray:
    """Row of pairwise RMSD: distance from coords_all[i] to all rows."""
    diffs = coords_all - coords_all[i:i + 1]
    sq = np.sum(diffs * diffs, axis=-1)
    if sq.ndim == 2:
        msd = sq.mean(axis=-1)
    else:
        msd = sq
    np.clip(msd, 0.0, None, out=msd)
    return np.sqrt(msd)


__all__ = ["SubspaceDiagonalizationRefiner"]
