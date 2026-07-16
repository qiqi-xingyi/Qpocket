# Author: Yuqi Zhang
"""Per-block minimal smoke tests for full_pipline.

Tests EACH functional block of the project in isolation, with the
smallest input that exercises it. The quantum blocks run on the **Aer**
simulator. Run (from the full_pipline/ project root):

    python -m tests.test_blocks_smoke

A shared setup runs ONE tiny Aer pipeline on the synthetic n=3 case and
reuses its outputs (accepted pool, refinement, prediction) for the
downstream blocks, so the suite stays fast. Each block prints PASS/FAIL;
the process exits non-zero if any block fails.
"""
from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests._fixtures import make_test_case

_BLOCKS: List[Tuple[str, Callable[["Ctx"], str]]] = []


def block(name):
    def deco(fn):
        _BLOCKS.append((name, fn))
        return fn
    return deco


class Ctx:
    """Shared state produced by setup(), consumed by the block tests."""
    enc = None
    ctx = None
    fh = None
    scorer = None
    valid_samples: List[Any] = []
    accepted_pool: List[Any] = []
    refinement = None
    prediction = None
    n_qubits = 0
    tmp = None


# --------------------------------------------------------------------- #
# shared setup: one tiny Aer pipeline                                    #
# --------------------------------------------------------------------- #

def setup(C: Ctx) -> None:
    from ras_folding.sampler.filter_hamiltonian import FilterHamiltonian
    from ras_folding.scoring.full_energy import FullEnergyScorer
    from ras_folding.sampler.base_sampler import EncoderBaseSampler
    from ras_folding.sampler.validity import decode_and_validate
    from ras_folding.quantum.backend_config import QuantumBackendConfig
    from ras_folding.quantum.circuit_builder import QuantumCircuitBuilder
    from ras_folding.quantum.moment_match_initializer import (
        MomentMatchConfig, MomentMatchInitializer,
    )
    from ras_folding.sampler.quantum_base_sampler import (
        QuantumBackendBaseSampler,
    )
    from ras_folding.sampler.imaginary_time_sampler import (
        QuantumImaginaryTimeSampler,
    )

    C.enc, C.ctx = make_test_case(3)
    C.n_qubits = C.enc.n_bonds * 6
    C.fh = FilterHamiltonian(residue_contact_weights=None)
    C.scorer = FullEnergyScorer(
        residue_contact_weights=None, rg_target=None,
        term_weights={"overlap_full": 10.0, "contact_full": 1.0,
                       "rg": 0.0, "anchor": 1.0, "turn": 0.0},
    )
    C.tmp = Path(tempfile.mkdtemp(prefix="blocks_smoke_"))

    # classical base samples (used by the sampler block). On the synthetic
    # n=3 case most random codes do NOT reach the anchor, so few/none are
    # valid — that is the validity filter working as intended. The
    # quantum-accepted pool below provides guaranteed-valid decoded samples
    # for the scoring / oracle blocks.
    base = EncoderBaseSampler(n_samples=24, seed=7, mode="random_codes")
    C.classical_samples = base.sample(C.ctx)
    for s in C.classical_samples:
        decode_and_validate(s, C.ctx)
    C.valid_samples = [s for s in C.classical_samples if s.valid]

    # ---- quantum (Aer) pipeline: moment match -> HEA -> Aer -> reject ----
    mm = MomentMatchInitializer(
        sampling_context=C.ctx, filter_hamiltonian=C.fh,
        config=MomentMatchConfig(K_samples=200, reps=1, seed=42,
                                  use_v2_prior=False),
    ).compute_theta()
    builder = QuantumCircuitBuilder(
        ansatz="hea_with_tf", reps=1, seed=42,
        ansatz_params=mm.theta, reps_hea=1,
    )
    qbase = QuantumBackendBaseSampler(
        backend_config=QuantumBackendConfig(
            backend_type="aer_simulator", execution_mode="job",
            shots_per_circuit=256, seed_simulator=42, seed_transpiler=42,
        ),
        circuit_builder=builder, n_circuits=2, seeds=[42, 43],
        output_dir=C.tmp / "qbase", task_id="blocks_n3",
    )
    its = QuantumImaginaryTimeSampler(
        taus=(0.0, 0.1), shots_per_tau=2 * 256, base_sampler=qbase,
        filter_hamiltonian=C.fh, scorer=C.scorer, seed=42,
    )
    batches = its.sample(C.ctx)
    C.batches = batches
    for b in batches:
        for s in b.samples:
            if s.is_eligible_for_refinement():
                C.accepted_pool.append(s)
    C.theta = mm.theta


# --------------------------------------------------------------------- #
# block tests                                                           #
# --------------------------------------------------------------------- #

@block("encoder (lattice/decoder/reachable)")
def b_encoder(C: Ctx) -> str:
    from ras_folding.encoder.decoder import decode_bitstring
    from ras_folding.encoder.reachable import ReachableSet
    from ras_folding.encoder.lattice import lattice_around
    coords = decode_bitstring(0, C.enc)
    assert coords.shape == (C.enc.n_residues, 3), coords.shape
    assert np.all(np.isfinite(coords))
    rs = ReachableSet(C.enc, epsilon=1.0)
    lat = lattice_around(C.enc.v_left_seed)
    assert lat.shape[1] == 3
    return f"decoded coords {coords.shape}, lattice {lat.shape}, reachable OK"


@block("scoring (mj_contact / filter_hamiltonian)")
def b_scoring(C: Ctx) -> str:
    from ras_folding.scoring.mj_contact import load_mj_table_default
    mj = load_mj_table_default()
    assert C.accepted_pool, "no valid (quantum-accepted) samples to score"
    total, terms = C.fh.evaluate(C.accepted_pool[0], C.ctx)
    assert np.isfinite(total) and total >= 0.0, total
    return f"mj table loaded, H_filter={total:.4f}, {len(terms)} terms"


@block("sampler (EncoderBaseSampler + validity)")
def b_sampler(C: Ctx) -> str:
    samples = C.classical_samples
    assert len(samples) == 24, len(samples)
    # decode_and_validate must have set a boolean .valid on every sample
    assert all(isinstance(s.valid, bool) for s in samples), "validity not set"
    # every valid sample must carry decoded coords
    assert all(s.coords is not None for s in samples if s.valid)
    n_valid = len(C.valid_samples)
    return (f"24 sampled + decode_and_validate ran; n_valid={n_valid} "
            f"(random synthetic paths rarely reach the anchor)")


@block("quantum AER (moment-match -> HEA -> Aer sample)")
def b_quantum_aer(C: Ctx) -> str:
    # exercised in setup(); assert outputs are sane
    assert C.theta.shape == (2 * C.n_qubits,), C.theta.shape
    assert np.all(np.isfinite(C.theta))
    assert any(b.n_raw > 0 for b in C.batches), "Aer produced no raw samples"
    n_raw = sum(b.n_raw for b in C.batches)
    return f"theta {C.theta.shape}, Aer raw samples={n_raw}"


@block("imaginary-time rejection AER (QuantumImaginaryTimeSampler)")
def b_imag_time(C: Ctx) -> str:
    assert len(C.batches) == 2, len(C.batches)
    n_valid = sum(b.n_valid for b in C.batches)
    assert n_valid > 0, "no valid decoded samples after rejection"
    return (f"{len(C.batches)} tau-batches, n_valid={n_valid}, "
            f"n_eligible={len(C.accepted_pool)}")


@block("refinement (Hybrid SQD: subspace_diagonalization)")
def b_refinement(C: Ctx) -> str:
    from ras_folding.refinement.subspace_diagonalization import (
        SubspaceDiagonalizationRefiner,
    )
    from ras_folding.sampler.sample_types import SampleBatch
    pool = C.accepted_pool
    assert len(pool) >= 5, f"need >=5 eligible, got {len(pool)}"
    batch = SampleBatch(
        samples=pool, tau=None, n_raw=len(pool),
        n_valid=sum(1 for s in pool if s.valid),
        n_accepted=sum(1 for s in pool if s.accepted),
    )
    refiner = SubspaceDiagonalizationRefiner(
        max_subspace_size=min(80, len(pool)),
        k_neighbors=min(8, max(2, len(pool) // 2)),
        kappa=0.2, n_modes=min(5, max(2, len(pool) // 3)),
        coupling_mode="hybrid", g_quantum=0.03,
        alpha_pauli=0.5, alpha_rmsd=0.5, n_qubits=C.n_qubits,
    )
    C.refinement = refiner.refine(batch)
    assert C.refinement.candidates, "refiner produced no candidates"
    return (f"selected={C.refinement.summary['n_selected']}, "
            f"modes={len(C.refinement.eigenvalues)}, "
            f"candidates={len(C.refinement.candidates)}")


@block("densify (PerturbationDenseFiller)")
def b_densify(C: Ctx) -> str:
    from ras_folding.densify.dense_filler import PerturbationDenseFiller
    assert C.accepted_pool, "no accepted pool to densify"
    filler = PerturbationDenseFiller(
        top_parents=5, children_per_parent=3,
        angular_sigmas_deg=(5.0, 10.0), max_local_rmsd=2.0,
        energy_window=50.0, perturbation_mass=0.5, seed=1,
    )
    res = filler.densify(C.accepted_pool, C.scorer, C.ctx)
    n_children = len(getattr(res, "dense_candidates", []) or [])
    assert res.summary is not None
    return f"dense children={n_children}, all_candidates={len(res.all_candidates)}"


@block("postprocess (PredictionPostProcessor)")
def b_postprocess(C: Ctx) -> str:
    from ras_folding.postprocess.prediction_postprocessor import (
        PredictionPostProcessor,
    )
    assert C.refinement is not None, "refinement block must run first"
    post = PredictionPostProcessor(
        dedup_rmsd_threshold=0.5, basin_rmsd_threshold=1.5,
        top_k_candidates=10, top_k_basins=5, export_pdb=True,
    )
    C.prediction = post.process(
        C.refinement, output_dir=C.tmp / "task", sequence="MAG",
    )
    s = C.prediction.summary
    return (f"top_k={len(C.prediction.top_candidates)}, "
            f"basins={s.get('n_basins')}, files={len(C.prediction.output_files)}")


@block("postprocess RMSD/dedup/clustering helpers")
def b_postprocess_helpers(C: Ctx) -> str:
    from ras_folding.postprocess.rmsd import ca_rmsd, pairwise_ca_rmsd
    # ca_rmsd takes raw (n,3) coord arrays
    a = np.random.default_rng(0).standard_normal((3, 3))
    b = a + 0.1
    r = ca_rmsd(a, b)
    assert np.isfinite(r) and r > 0
    # pairwise_ca_rmsd takes RefinedCandidate-like objects (.sample.coords)
    cands = C.refinement.candidates[:4]
    M = pairwise_ca_rmsd(cands)
    n = len(cands)
    assert M.shape == (n, n) and abs(M[0, 0]) < 1e-9
    return f"ca_rmsd={r:.4f}, pairwise matrix {M.shape}"


@block("landscape (LandscapeReconstructor PCA)")
def b_landscape(C: Ctx) -> str:
    from ras_folding.kras.landscape import LandscapeReconstructor
    assert C.refinement is not None
    recon = LandscapeReconstructor(grid_size=40, plot_enabled=False)
    res = recon.reconstruct(
        C.refinement.candidates, output_dir=C.tmp / "landscape",
        reference_coords=None,
    )
    assert res.summary is not None
    return f"landscape summary keys={sorted(res.summary)[:4]}"


@block("structure_analysis (StructureAnalyzer)")
def b_structure_analysis(C: Ctx) -> str:
    from ras_folding.kras.structure_analysis import StructureAnalyzer
    assert C.prediction is not None, "postprocess block must run first"
    analyzer = StructureAnalyzer()
    summary = analyzer.analyze(
        C.prediction, reference_coords=None, output_dir=C.tmp / "analysis",
    )
    assert isinstance(summary, dict)
    return f"analysis keys={sorted(summary)[:4]}"


@block("kras task_loader (load_kras_tasks)")
def b_task_loader(C: Ctx) -> str:
    from ras_folding.kras.task_loader import load_kras_tasks
    tasks, schema = load_kras_tasks(
        PROJECT_ROOT / "inputs" / "kras_tasks.csv",
        pdb_dir=PROJECT_ROOT / "kras_select_systems",
        flank_size=1, skip_missing_pdb=True,
    )
    assert len(tasks) > 0, "no tasks loaded"
    n_ref = sum(1 for t in tasks if t.reference_coords is not None)
    return f"loaded {len(tasks)} tasks ({n_ref} with reference)"


@block("reconstruct (PULCHRA adapter + CA io)")
def b_reconstruct(C: Ctx) -> str:
    from ras_folding.reconstruct.pulchra_adapter import PulchraAdapter
    from ras_folding.reconstruct.io import read_ca_coords_from_pdb
    ad = PulchraAdapter()  # auto-resolves vendored tools/pulchra/pulchra
    assert ad.check_available(), "PULCHRA binary not available"
    ca = read_ca_coords_from_pdb(PROJECT_ROOT / "kras_select_systems" / "6GJ6.pdb")
    assert ca.ndim == 2 and ca.shape[1] == 3 and ca.shape[0] > 100
    return f"pulchra available, read {ca.shape[0]} CA atoms from 6GJ6.pdb"


@block("oracle_eval (RMSDOracleSelector)")
def b_oracle(C: Ctx) -> str:
    from oracle_eval.types import OracleCandidate
    from oracle_eval.rmsd_oracle import RMSDOracleSelector
    assert len(C.accepted_pool) >= 2, "need >=2 valid samples"
    cands = [
        OracleCandidate(
            task_id="t", source="accepted", bitstring=s.bitstring,
            coords=np.asarray(s.coords, dtype=np.float64),
        )
        for s in C.accepted_pool[:3]
    ]
    ref = cands[0].coords.copy()  # exact match -> best_rmsd ~ 0
    sel = RMSDOracleSelector(use_kabsch=False)
    res = sel.select_best(
        "t", cands, reference_coords=ref, output_dir=C.tmp / "oracle",
        sequence="MAG",
    )
    assert res.reference_available and res.best_rmsd is not None
    assert res.best_rmsd < 1e-6, res.best_rmsd
    return f"considered={res.n_candidates_considered}, best_rmsd={res.best_rmsd:.2e}"


@block("docking_eval (kd math + ligand box; vina absent -> not run)")
def b_docking(C: Ctx) -> str:
    from docking_eval.kd import affinity_to_kd_m
    from docking_eval.ligand import (
        extract_ligand_from_pdb, compute_ligand_box,
    )
    # kd block: stronger (more negative) ΔG -> smaller Kd; non-finite raises
    kd_strong = affinity_to_kd_m(-12.0)
    kd_weak = affinity_to_kd_m(-6.0)
    assert 0 < kd_strong < kd_weak, (kd_strong, kd_weak)
    raised = False
    try:
        affinity_to_kd_m(float("nan"))
    except ValueError:
        raised = True
    assert raised, "affinity_to_kd_m must reject non-finite ΔG"
    # ligand block: extract the native ligand (EZZ) from 6GJ6.pdb -> box
    lig_pdb = C.tmp / "ligand.pdb"
    extract_ligand_from_pdb(
        PROJECT_ROOT / "kras_select_systems" / "6GJ6.pdb",
        ligand_resname="EZZ", output_pdb=lig_pdb,
    )
    box = compute_ligand_box(lig_pdb)
    assert box and box.get("n_ligand_atoms", 0) > 0, box
    return (f"Kd(-12)={kd_strong:.2e} < Kd(-6)={kd_weak:.2e}; "
            f"ligand box n_atoms={box['n_ligand_atoms']}")


@block("pipeline_validation (FinalPipelineValidator)")
def b_validation(C: Ctx) -> str:
    from pipeline_validation.checks import FinalPipelineValidator
    # run the validation engine against the postprocess task_dir built
    # above; it emits a ValidationResult of checks (some PASS, some
    # WARN/FAIL for the stages this synthetic run didn't produce).
    validator = FinalPipelineValidator()
    res = validator.validate_task(
        C.tmp / "task", task_metadata={"task_id": "blocks_n3"},
    )
    assert hasattr(res, "checks") and len(res.checks) > 0, "no checks emitted"
    statuses = {}
    for c in res.checks:
        statuses[c.status] = statuses.get(c.status, 0) + 1
    return f"engine ran, {len(res.checks)} checks: {statuses}"


# --------------------------------------------------------------------- #
# driver                                                                #
# --------------------------------------------------------------------- #

def main() -> int:
    print("=" * 72)
    print("full_pipline — per-block minimal smoke tests (quantum on Aer)")
    print("=" * 72)
    C = Ctx()
    print("\n[setup] running one tiny Aer pipeline on synthetic n=3 ...")
    try:
        setup(C)
        print(f"  setup OK: {len(C.valid_samples)} classical valid, "
              f"{len(C.accepted_pool)} quantum-eligible candidates")
    except Exception:
        print("  setup FAILED:")
        print(traceback.format_exc())
        return 1

    results: List[Tuple[str, bool, str]] = []
    for name, fn in _BLOCKS:
        try:
            detail = fn(C)
            results.append((name, True, detail))
            print(f"  [PASS] {name}\n         {detail}")
        except Exception as exc:
            tb = traceback.format_exc().strip().splitlines()[-1]
            results.append((name, False, tb))
            print(f"  [FAIL] {name}\n         {tb}")

    n_pass = sum(1 for _, ok, _ in results if ok)
    n_fail = len(results) - n_pass
    print("\n" + "=" * 72)
    print("BLOCK SUMMARY")
    print("=" * 72)
    for name, ok, _ in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("-" * 72)
    print(f"  {n_pass} passed, {n_fail} failed, {len(results)} blocks")
    print("=" * 72)

    # cleanup temp dir
    try:
        import shutil
        if C.tmp is not None:
            shutil.rmtree(C.tmp, ignore_errors=True)
    except Exception:
        pass
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
