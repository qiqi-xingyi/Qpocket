# Author: Yuqi Zhang
"""Stage A — V2 environment + corridor prior integration tests.

Runs as a plain script (no pytest dependency). Exits non-zero on any
assertion failure. Each test prints its name and outcome.

These exercise the env-conditioned prior that feeds the moment matcher
(Stage A): the same ``build_environment_prior`` /
``build_corridor_prior`` / ``PriorConditionedBaseSampler`` path that
``KrasFullBatchRunner._build_v2_prior`` wires into
``MomentMatchInitializer``.

Coverage:
  - EnvironmentPriorBuilder (fragment exclusion, ligand selection mode)
  - CorridorPriorBuilder (anchor_only fallback + ligand-anchor Bézier)
  - PriorConditionedDirectionPolicy (probability sums, infeasible mask)
  - PriorConditionedBaseSampler (bitstring length, bit marginals, determinism)
  - shot_budget (length-adaptive + fixed)
  - OracleAnchor injection (source labelling)
  - landscape writer

Requires the copied assets ``inputs/kras_tasks.csv`` and
``kras_select_systems/6GJ6.pdb`` (task ``6GJ6_G12D_Frag-D``).

Run (from the full_pipline/ project root):
    python -m tests.test_prior_integration
"""
from __future__ import annotations

import math
import shutil
import sys
import traceback
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ras_folding.kras.task_loader import load_kras_tasks
from ras_folding.encoder.lattice import lattice_around, N_DIRECTIONS
from ras_folding.encoder.reachable import (
    L_MAX, L_MIN, MIN_SEP, SKIP_RECENT,
    case_reachable, clash_mask, reach_mask,
)
from ras_folding.utils.constants import CA_CA_LENGTH

from ras_folding.prior.environment import build_environment_prior
from ras_folding.prior.corridor import (
    build_corridor_prior, CA_LENGTH,
)
from ras_folding.prior.direction_policy import (
    PriorConditionedDirectionPolicy,
)
from ras_folding.prior.prior_sampler import PriorConditionedBaseSampler
from ras_folding.prior.shot_budget import allocate_shots, SHOT_BUDGET_DEFAULTS
from ras_folding.prior.oracle_anchor import build_native_oracle_anchor
from ras_folding.prior.landscape import write_landscape


_TESTS: List[Tuple[str, Callable]] = []


def test(name):
    def deco(fn):
        _TESTS.append((name, fn))
        return fn
    return deco


def _load_one_task(task_id="6GJ6_G12D_Frag-D"):
    pdb_dir = ROOT / "kras_select_systems"
    tasks, _ = load_kras_tasks(ROOT / "inputs" / "kras_tasks.csv",
                                 pdb_dir=pdb_dir)
    for t in tasks:
        if t.task_id == task_id:
            return t, pdb_dir
    raise RuntimeError(f"task {task_id} not found")


# ------------------------------------------------------------------------- #
# Tests                                                                     #
# ------------------------------------------------------------------------- #

@test("environment: predicted fragment removed from env_tree")
def test_env_excludes_fragment():
    t, pdb = _load_one_task()
    meta = t.metadata
    ctx = build_environment_prior(
        pdb_path=str(pdb / meta["ref_pdb"]),
        chain_id=meta["chain_id"],
        start_resi=int(meta["start_resi"]),
        end_resi=int(meta["end_resi"]),
        ligand_resname=meta.get("ligand_resname"),
        fragment_ca_centroid=t.reference_coords.mean(axis=0),
    )
    # Fragment CA should be at >= some distance from env atoms (since
    # excluded). Closest env atom must be a NEIGHBOUR (i.e. d > 0).
    for ca in t.reference_coords:
        d, _ = ctx.env_tree.query(ca, k=1)
        assert d > 0.5, f"env contains fragment CA: d={d}"
    assert ctx.n_env_atoms > 0
    # ligand atoms are NOT in env_tree
    if ctx.ligand_atom_coords is not None and ctx.ligand_atom_coords.size:
        for la in ctx.ligand_atom_coords:
            d, _ = ctx.env_tree.query(la, k=1)
            assert d > 0.5, f"env_tree contains ligand atom: d={d}"


@test("environment: ligand selection mode reported")
def test_env_ligand_selection():
    t, pdb = _load_one_task()
    meta = t.metadata
    ctx = build_environment_prior(
        pdb_path=str(pdb / meta["ref_pdb"]),
        chain_id=meta["chain_id"],
        start_resi=int(meta["start_resi"]),
        end_resi=int(meta["end_resi"]),
        ligand_resname=meta.get("ligand_resname"),
        fragment_ca_centroid=t.reference_coords.mean(axis=0),
    )
    assert ctx.ligand_selection_mode in (
        "first", "nearest_to_pocket_center", "explicit",
        "first_fallback", "nearest_to_pocket_center_fallback",
        "explicit_not_found", "none",
    )


@test("corridor: t=0 returns P0; t=1 returns P3; tangent finite")
def test_corridor_endpoints():
    t, pdb = _load_one_task()
    meta = t.metadata
    env = build_environment_prior(
        pdb_path=str(pdb / meta["ref_pdb"]),
        chain_id=meta["chain_id"],
        start_resi=int(meta["start_resi"]),
        end_resi=int(meta["end_resi"]),
        ligand_resname=meta.get("ligand_resname"),
        fragment_ca_centroid=t.reference_coords.mean(axis=0),
    )
    cor = build_corridor_prior(t.encoder_inputs, env)
    p0 = cor.bezier_point(0.0)
    p1 = cor.bezier_point(1.0)
    assert np.allclose(p0, cor.P0, atol=1e-9)
    assert np.allclose(p1, cor.P3, atol=1e-9)
    tan = cor.bezier_tangent(0.5)
    assert np.all(np.isfinite(tan))
    assert np.linalg.norm(tan) > 0


@test("corridor: anchor-only fallback when ligand absent")
def test_corridor_anchor_only_fallback():
    t, pdb = _load_one_task()
    meta = t.metadata
    env = build_environment_prior(
        pdb_path=str(pdb / meta["ref_pdb"]),
        chain_id=meta["chain_id"],
        start_resi=int(meta["start_resi"]),
        end_resi=int(meta["end_resi"]),
        ligand_resname=None,  # force no ligand
        fragment_ca_centroid=t.reference_coords.mean(axis=0),
    )
    cor = build_corridor_prior(t.encoder_inputs, env)
    assert cor.corridor_mode == "anchor_only_bezier"
    assert cor.ligand_weight_effective == 0.0


@test("direction policy: probabilities sum to 1; infeasible=0")
def test_direction_policy_sum1():
    t, pdb = _load_one_task()
    meta = t.metadata
    env = build_environment_prior(
        pdb_path=str(pdb / meta["ref_pdb"]),
        chain_id=meta["chain_id"],
        start_resi=int(meta["start_resi"]),
        end_resi=int(meta["end_resi"]),
        ligand_resname=meta.get("ligand_resname"),
        fragment_ca_centroid=t.reference_coords.mean(axis=0),
    )
    cor = build_corridor_prior(t.encoder_inputs, env)
    pol = PriorConditionedDirectionPolicy(env, cor)
    ei = t.encoder_inputs
    cur_pos = ei.anchor_left.copy()
    last_dir = ei.v_left_seed.copy()
    lat = lattice_around(last_dir)
    endpoints = cur_pos[None, :] + CA_CA_LENGTH * lat
    rmask = reach_mask(endpoints, ei.anchor_right, ei.n_bonds - 1, 1.0)
    cmask = clash_mask(endpoints, np.array([cur_pos]))
    feas = rmask & cmask
    res = pol.step_prior(cur_pos, last_dir, 0, ei.n_bonds, lat, feas)
    assert not res.invalid
    assert math.isclose(float(res.probabilities.sum()), 1.0, abs_tol=1e-9)
    assert np.all(res.probabilities[~feas] == 0.0)
    assert np.all(np.isfinite(res.probabilities))


@test("prior sampler: bitstring length = 6 * n_bonds; bit marginals in [0,1]")
def test_prior_sampler_bitstring():
    t, pdb = _load_one_task()
    meta = t.metadata
    env = build_environment_prior(
        pdb_path=str(pdb / meta["ref_pdb"]),
        chain_id=meta["chain_id"],
        start_resi=int(meta["start_resi"]),
        end_resi=int(meta["end_resi"]),
        ligand_resname=meta.get("ligand_resname"),
        fragment_ca_centroid=t.reference_coords.mean(axis=0),
    )
    cor = build_corridor_prior(t.encoder_inputs, env)
    sampler = PriorConditionedBaseSampler(
        t.encoder_inputs, env, cor, n_prior_samples=32, seed=42,
    )
    res = sampler.sample()
    n_qubits = t.encoder_inputs.n_bonds * 6
    assert res.n_qubits == n_qubits
    assert res.bit_marginals.shape == (n_qubits,)
    assert res.bit_marginals.min() >= 0.0
    assert res.bit_marginals.max() <= 1.0
    for bs in res.valid_bitstrings:
        assert len(bs) == n_qubits


@test("prior sampler: deterministic with fixed seed")
def test_prior_sampler_deterministic():
    t, pdb = _load_one_task()
    meta = t.metadata
    env = build_environment_prior(
        pdb_path=str(pdb / meta["ref_pdb"]),
        chain_id=meta["chain_id"],
        start_resi=int(meta["start_resi"]),
        end_resi=int(meta["end_resi"]),
        ligand_resname=meta.get("ligand_resname"),
        fragment_ca_centroid=t.reference_coords.mean(axis=0),
    )
    cor = build_corridor_prior(t.encoder_inputs, env)
    sampler1 = PriorConditionedBaseSampler(
        t.encoder_inputs, env, cor, n_prior_samples=16, seed=2026,
    )
    sampler2 = PriorConditionedBaseSampler(
        t.encoder_inputs, env, cor, n_prior_samples=16, seed=2026,
    )
    a = sampler1.sample()
    b = sampler2.sample()
    assert a.valid_bitstrings == b.valid_bitstrings, \
        "deterministic seed should reproduce bitstrings"
    assert np.allclose(a.bit_marginals, b.bit_marginals)


@test("shot_budget: length-adaptive defaults")
def test_shot_budget_lengths():
    a = allocate_shots(7, 6, budget_mode="length_adaptive")
    assert a.requested_total_shots == SHOT_BUDGET_DEFAULTS["shots_short"]
    a = allocate_shots(10, 9, budget_mode="length_adaptive")
    assert a.requested_total_shots == SHOT_BUDGET_DEFAULTS["shots_medium"]
    a = allocate_shots(12, 11, budget_mode="length_adaptive")
    assert a.requested_total_shots == SHOT_BUDGET_DEFAULTS["shots_long"]
    a = allocate_shots(15, 14, budget_mode="length_adaptive")
    assert a.requested_total_shots == SHOT_BUDGET_DEFAULTS["shots_xlong"]
    # n_circuits * shots_per_circuit >= requested
    assert a.n_circuits * a.shots_per_circuit >= a.requested_total_shots


@test("shot_budget: fixed mode bypasses min/max bounds (V2 revised 2026-04-30)")
def test_shot_budget_fixed():
    # In fixed mode, the user's request is the EXACT target; min/max
    # bounds from config are NOT applied. (Behaviour clarified 2026-04-30.)
    a = allocate_shots(15, 14, budget_mode="fixed",
                        fixed_shots_per_task=4096)
    assert a.requested_total_shots == 4096, (
        "fixed mode must respect user's exact value (no min clamp)"
    )
    # explicit small request
    a = allocate_shots(7, 6, budget_mode="fixed",
                        fixed_shots_per_task=8192,
                        config_override={
                            "shots_per_circuit": 4096,
                        })
    assert a.requested_total_shots == 8192
    assert a.allocated_total_shots == 8192
    assert a.n_circuits == 2
    assert a.shots_per_circuit == 4096


@test("oracle anchor: source label correct")
def test_oracle_anchor_source():
    t, _ = _load_one_task()
    anchor = build_native_oracle_anchor(
        t.reference_coords, t.sequence,
    )
    assert anchor.source == "theoretical_lower_bound"
    assert anchor.anchor_type == "native_reference"
    assert np.array_equal(anchor.coords, t.reference_coords)


@test("landscape writer: produces required files")
def test_landscape_writer():
    out = ROOT / "tests" / "_tmp_landscape" / "landscape"
    if out.parent.exists():
        shutil.rmtree(out.parent)
    coords = np.zeros((5, 7, 3), dtype=np.float64)
    candidate_rows = [
        {"candidate_id": f"c{i}", "source": "prior_rollout",
         "is_oracle_anchor": False, "coords_index": i,
         "kabsch_rmsd": 1.0 + i * 0.1, "basin_id": ""}
        for i in range(5)
    ]
    summary = {
        "task_id": "test", "n_candidates_total": 5,
        "n_generated_candidates": 5, "n_oracle_anchor": 0,
    }
    paths = write_landscape(
        out, task_id="test", case_id="test_case",
        mutation_group=None, ligand_family=None, pocket_module=None,
        candidate_rows=candidate_rows, coords=coords,
        basin_summary_rows=[], landscape_summary=summary,
    )
    for p in paths.values():
        assert p.exists(), f"missing {p}"
    shutil.rmtree(out.parent)


# ------------------------------------------------------------------------- #
# Driver                                                                    #
# ------------------------------------------------------------------------- #

def main():
    n_pass = 0
    n_fail = 0
    failures = []
    for name, fn in _TESTS:
        sys.stdout.write(f"  {name} ... ")
        sys.stdout.flush()
        try:
            fn()
            print("PASS")
            n_pass += 1
        except Exception as exc:
            print("FAIL")
            failures.append((name, traceback.format_exc()))
            n_fail += 1
    print()
    print(f"  ==> {n_pass} passed, {n_fail} failed")
    if failures:
        print()
        for name, tb in failures:
            print(f"  ---- FAIL: {name} ----")
            print(tb)
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
