# Author: Yuqi Zhang
"""minimum closed-loop smoke test.

Runs the COMPLETE pipeline on a synthetic n_residues=3 case
(12 qubits) on the Aer simulator. NOT a benchmark — just confirms the
end-to-end pipeline integrates and produces non-NaN output.

Stages exercised:
  A. MomentMatchInitializer → θ (closed-form, no VQE)
  B. HEA circuit construction via QuantumCircuitBuilder (ansatz='hea_with_tf')
  C. Aer sampling (matrix_product_state)
  D. counts_to_candidate_samples → decode_and_validate
  E. QuantumImaginaryTimeSampler classical rejection
  F. SubspaceDiagonalizationRefiner with coupling_mode='hybrid'
  G. metrics summary

Run:
    python -m examples.run_smoke
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ras_folding.quantum.backend_config import QuantumBackendConfig
from ras_folding.quantum.circuit_builder import QuantumCircuitBuilder
from ras_folding.quantum.moment_match_initializer import (
    MomentMatchConfig, MomentMatchInitializer,
)
from ras_folding.sampler.context import SamplingContext
from ras_folding.sampler.filter_hamiltonian import FilterHamiltonian
from ras_folding.sampler.imaginary_time_sampler import (
    QuantumImaginaryTimeSampler,
)
from ras_folding.sampler.quantum_base_sampler import QuantumBackendBaseSampler
from ras_folding.scoring.full_energy import FullEnergyScorer
from ras_folding.refinement.subspace_diagonalization import (
    SubspaceDiagonalizationRefiner,
)
from ras_folding.sampler.sample_types import SampleBatch
from tests._fixtures import make_test_case


def run_smoke() -> Dict[str, Any]:
    """Run end-to-end pipeline on n_residues=3 synthetic case."""
    print("=" * 72)
    print("Pipeline Minimum Closed-Loop Smoke Test")
    print("=" * 72)

    # ---- Stage 0: setup ------------------------------------------------
    print("\n[stage 0] Setup")
    enc, ctx = make_test_case(n_residues=3)
    n_qubits = enc.n_bonds * 6
    n_bonds = enc.n_bonds
    print(f"  case: n_residues={enc.n_residues}, n_bonds={n_bonds}, "
          f"n_qubits={n_qubits}")
    fh = FilterHamiltonian(residue_contact_weights=None)
    scorer = FullEnergyScorer(
        residue_contact_weights=None, rg_target=None,
        term_weights={"overlap_full": 10.0, "contact_full": 1.0,
                       "rg": 0.0, "anchor": 1.0, "turn": 0.0},
    )

    # ---- Stage A: Moment matching -------------------------------------
    print("\n[stage A] MomentMatchInitializer")
    t0 = time.time()
    mm_cfg = MomentMatchConfig(K_samples=200, reps=1, seed=42,
                                use_v2_prior=False)
    mm_init = MomentMatchInitializer(
        sampling_context=ctx, filter_hamiltonian=fh, config=mm_cfg,
    )
    mm_result = mm_init.compute_theta()
    t_mm = time.time() - t0
    print(f"  sampling_mode      : {mm_result.sampling_mode}")
    print(f"  n_valid_samples    : {mm_result.n_valid_samples}")
    print(f"  theta shape        : {mm_result.theta.shape}")
    print(f"  |theta|            : {np.linalg.norm(mm_result.theta):.4f}")
    print(f"  z_star H_filter    : {mm_result.z_star_h_filter:.4f}")
    print(f"  max |C|            : "
          f"{mm_result.diagnostics['max_abs_correlation']:.4f}")
    print(f"  elapsed            : {t_mm*1000:.1f} ms")

    # ---- Stage B: HEA circuit + Aer sampling --------------------------
    print("\n[stage B+C+D] HEA circuit + Aer sampling + counts → samples")
    t0 = time.time()
    backend_config = QuantumBackendConfig(
        backend_type="aer_simulator",
        execution_mode="job",
        shots_per_circuit=512,
        seed_simulator=42, seed_transpiler=42,
    )
    builder = QuantumCircuitBuilder(
        ansatz="hea_with_tf",
        reps=1,
        seed=42,
        ansatz_params=mm_result.theta,
        reps_hea=1,
    )
    out_dir = PROJECT_ROOT / "examples" / "_smoke_run"
    out_dir.mkdir(parents=True, exist_ok=True)
    qbase = QuantumBackendBaseSampler(
        backend_config=backend_config,
        circuit_builder=builder,
        n_circuits=2,
        seeds=[42, 43],
        output_dir=out_dir,
        task_id="smoke_n3",
    )

    # ---- Stage E: imaginary-time sampling -----------------------------
    its = QuantumImaginaryTimeSampler(
        taus=(0.0, 0.1),
        shots_per_tau=2 * 512,
        base_sampler=qbase,
        filter_hamiltonian=fh,
        scorer=scorer,
        seed=42,
    )
    batches = its.sample(ctx)
    t_sampling = time.time() - t0
    accepted_pool = []
    for b in batches:
        for s in b.samples:
            if s.is_eligible_for_refinement():
                accepted_pool.append(s)
    print(f"  n_batches          : {len(batches)}")
    for b in batches:
        print(f"    tau={b.tau}: n_raw={b.n_raw}, n_valid={b.n_valid}, "
              f"n_accepted={b.n_accepted}, acceptance={b.acceptance_rate:.3f}")
    print(f"  n_eligible candidates: {len(accepted_pool)}")
    print(f"  elapsed (B+C+D+E)  : {t_sampling*1000:.1f} ms")

    # ---- Stage F: Hybrid SQD refinement -------------------------------
    print("\n[stage F] SubspaceDiagonalizationRefiner (coupling='hybrid')")
    t0 = time.time()
    if len(accepted_pool) >= 5:
        final_batch = SampleBatch(
            samples=accepted_pool,
            tau=None,
            n_raw=len(accepted_pool),
            n_valid=sum(1 for s in accepted_pool if s.valid),
            n_accepted=sum(1 for s in accepted_pool if s.accepted),
        )
        refiner = SubspaceDiagonalizationRefiner(
            max_subspace_size=min(100, len(accepted_pool)),
            k_neighbors=min(10, max(2, len(accepted_pool) // 2)),
            kappa=0.2,
            n_modes=min(5, max(2, len(accepted_pool) // 3)),
            coupling_mode="hybrid",
            g_quantum=0.03,
            alpha_pauli=0.5,
            alpha_rmsd=0.5,
            n_qubits=n_qubits,
        )
        refine_result = refiner.refine(final_batch)
        t_refine = time.time() - t0
        print(f"  n_eligible         : {refine_result.summary['n_eligible']}")
        print(f"  n_selected         : {refine_result.summary['n_selected']}")
        print(f"  n_nonzero_couplings: "
              f"{refine_result.summary['n_nonzero_couplings']}")
        print(f"  # eigenvalues      : {len(refine_result.eigenvalues)}")
        print(f"  # candidates       : {len(refine_result.candidates)}")
        print(f"  elapsed            : {t_refine*1000:.1f} ms")

        if refine_result.candidates:
            top = refine_result.candidates[0]
            print(f"  top1 refined_score : {top.refined_score:.4f}")
            print(f"  top1 refined_weight: {top.refined_weight:.4f}")
        refinement_summary = refine_result.summary
    else:
        print(f"  SKIP (only {len(accepted_pool)} accepted, need >= 5)")
        refinement_summary = {"skipped_too_few_candidates": True}

    # ---- Final check ---------------------------------------------------
    print("\n[checks]")
    checks = {
        "moment_match_produced_theta": (
            mm_result.theta.shape == (24,)  # 2 * 12 for reps=1
            and np.all(np.isfinite(mm_result.theta))
        ),
        "aer_circuits_ran": all(b.n_raw > 0 for b in batches),
        "candidates_decoded": any(b.n_valid > 0 for b in batches),
        "pipeline_no_crash": True,
    }
    for k, v in checks.items():
        status = "✓" if v else "✗"
        print(f"  {status} {k}: {v}")

    all_pass = all(checks.values())
    print()
    print("=" * 72)
    print(f"SMOKE TEST: {'PASS' if all_pass else 'FAIL'}")
    print("=" * 72)

    return {
        "checks": checks,
        "moment_match": {
            "sampling_mode": mm_result.sampling_mode,
            "n_valid_samples": mm_result.n_valid_samples,
        },
        "batches": [{"tau": b.tau, "n_raw": b.n_raw, "n_valid": b.n_valid,
                      "n_accepted": b.n_accepted} for b in batches],
        "n_accepted_pool": len(accepted_pool),
        "refinement": refinement_summary,
        "pass": all_pass,
    }


if __name__ == "__main__":
    out = run_smoke()
    sys.exit(0 if out["pass"] else 1)
