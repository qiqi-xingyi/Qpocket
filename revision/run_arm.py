#!/usr/bin/env python3
# Author: Yuqi Zhang
"""Driver for one revision comparison arm on one KRAS task.

One invocation == one (task, arm, repeat) cell. The SLURM array script
maps array indices onto cells, so every cell is an independent process
with its own seed and its own run record.

Example
-------
    python revision/run_arm.py \
        --arm P --task-id 8AZZ_G12V_Frag-B --repeat 0 \
        --budget-mode proposal --n-proposals 2007040 \
        --output-root revision/results/g1_pilot

Outputs, under ``<output-root>/<arm>/<task_id>/repeat_<r>/``:
    candidates.csv    bitstring, count  (coords deliberately not stored —
                      the shared downstream decodes them)
    run_record.json   provenance, budget accounting, timing
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ras_folding.kras.task_loader import load_kras_tasks          # noqa: E402
from ras_folding.sampler.context import SamplingContext            # noqa: E402
from ras_folding.sampler.filter_hamiltonian import FilterHamiltonian  # noqa: E402
from ras_folding.sampler.imaginary_time_sampler import (            # noqa: E402
    QuantumImaginaryTimeSampler,
)
from ras_folding.scoring.full_energy import FullEnergyScorer        # noqa: E402
from ras_folding.scoring.mj_contact import load_mj_table_default    # noqa: E402
from ras_folding.prior.environment import build_environment_prior  # noqa: E402
from ras_folding.prior.corridor import build_corridor_prior        # noqa: E402
from revision.arms.prior_arm import (                              # noqa: E402
    BudgetMode, PriorArmBaseSampler, PriorArmBudget,
)
from revision.arms.mcmc_arm import MCMCArmBaseSampler                # noqa: E402
from revision.runrecord import RunRecord, write_run_record         # noqa: E402

logger = logging.getLogger("revision.run_arm")

# Master seed for the revision experiments. A cell's seed is derived
# deterministically from (master, arm, task_id, repeat) so that cells are
# independent, reproducible, and never collide across arms.
MASTER_SEED = 20260825


def derive_seed(arm: str, task_id: str, repeat: int) -> int:
    """Stable per-cell seed. Same inputs always give the same seed."""
    key = f"{MASTER_SEED}|{arm}|{task_id}|{repeat}".encode("utf-8")
    entropy = int.from_bytes(key, "big") % (2 ** 63 - 1)
    return int(np.random.SeedSequence(entropy).generate_state(
        1, dtype=np.uint32)[0])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run one revision comparison-arm cell.",
    )
    p.add_argument("--arm", required=True, choices=["P", "M"],
                   help="comparison arm. Q is not generated here: it is "
                        "hardware-executed and read from its own run records")
    p.add_argument("--task-id", required=True)
    p.add_argument("--repeat", type=int, default=0,
                   help="independent-repeat index; changes only the seed")
    p.add_argument("--tasks", default="inputs/kras_tasks.csv")
    p.add_argument("--structure-root", default="kras_select_systems")
    p.add_argument("--output-root", default="revision/results/g1_pilot")

    b = p.add_argument_group("budget")
    b.add_argument("--budget-mode", choices=["proposal", "valid"],
                   default="proposal")
    b.add_argument("--n-proposals", type=int, default=None,
                   help="PROPOSAL mode: rollout attempts (match Q's shots)")
    b.add_argument("--n-valid-target", type=int, default=None,
                   help="VALID mode: valid paths to collect")
    b.add_argument("--max-attempts", type=int, default=None,
                   help="VALID mode: hard attempt cap")
    b.add_argument("--chunk-size", type=int, default=16384)
    b.add_argument("--taus", type=str, default="0.0,0.1,0.2",
                   help="imaginary-time strengths. The arm is invoked once "
                        "per tau with the full per-tau budget, exactly as the "
                        "quantum arm was: its per-task shot allocation is a "
                        "per-tau figure, so a single-tau run would under-"
                        "budget the arm threefold at the default setting")

    r = p.add_argument_group("prior")
    r.add_argument("--leakage-mode", default="full",
                   choices=["full", "perturbed"],
                   help="corridor crystal-leakage mode. Defaults to 'full' "
                        "because the quantum arm this pilot is compared "
                        "against was run that way; the arms must share one "
                        "prior or the contrast measures the prior, not the "
                        "sampler. 'perturbed' is the de-leaked setting and is "
                        "the right choice for any arm generated fresh")
    r.add_argument("--perturbation-sigma", type=float, default=5.0)
    r.add_argument("--perturbation-clip", type=float, default=2.0)

    m = p.add_argument_group("arm M (Metropolis-Hastings)")
    m.add_argument("--suffix-move-prob", type=float, default=0.5,
                   help="probability of a suffix Gibbs move; the remainder "
                        "are symmetric single-site moves")
    m.add_argument("--burn-in", type=int, default=1000,
                   help="proposals discarded before the chain is recorded")
    m.add_argument("--thin", type=int, default=1)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    root = PROJECT_ROOT
    out_dir = (Path(args.output_root) if Path(args.output_root).is_absolute()
               else root / args.output_root)
    cell_dir = out_dir / args.arm / args.task_id / f"repeat_{args.repeat}"
    cell_dir.mkdir(parents=True, exist_ok=True)

    seed = derive_seed(args.arm, args.task_id, args.repeat)
    record = RunRecord(
        experiment="g1_kras_pilot", arm=args.arm,
        task_id=args.task_id,
        config={
            "repeat": args.repeat,
            "seed": seed,
            "master_seed": MASTER_SEED,
            "budget_mode": args.budget_mode,
            "n_proposals": args.n_proposals,
            "n_valid_target": args.n_valid_target,
            "max_attempts": args.max_attempts,
            "chunk_size": args.chunk_size,
            "leakage_mode": args.leakage_mode,
            "perturbation_sigma": args.perturbation_sigma,
            "perturbation_clip": args.perturbation_clip,
            **({"suffix_move_prob": args.suffix_move_prob,
                "burn_in": args.burn_in,
                "thin": args.thin} if args.arm == "M" else {}),
        },
    ).start(root)

    tasks, _schema = load_kras_tasks(
        root / args.tasks, pdb_dir=root / args.structure_root,
    )
    task = next((t for t in tasks if t.task_id == args.task_id), None)
    if task is None:
        logger.error("task_id %r not found in %s", args.task_id, args.tasks)
        return 2

    ei = task.encoder_inputs
    meta = task.metadata
    centroid = (
        np.asarray(task.reference_coords, dtype=np.float64).mean(axis=0)
        if task.reference_coords is not None else None
    )
    env_ctx = build_environment_prior(
        pdb_path=str(root / args.structure_root / meta["ref_pdb"]),
        chain_id=meta["chain_id"],
        start_resi=int(meta["start_resi"]),
        end_resi=int(meta["end_resi"]),
        ligand_resname=meta.get("ligand_resname"),
        fragment_ca_centroid=centroid,
    )
    corridor_ctx = build_corridor_prior(
        ei, env_ctx,
        crystal_leakage_mode=args.leakage_mode,
        perturbation_sigma=args.perturbation_sigma,
        perturbation_clip_n_sigma=args.perturbation_clip,
        task_id=task.task_id,
    )
    record.inputs = {
        "ref_pdb": meta.get("ref_pdb"),
        "chain_id": meta.get("chain_id"),
        "start_resi": meta.get("start_resi"),
        "end_resi": meta.get("end_resi"),
        "ligand_resname": meta.get("ligand_resname"),
        "n_residues": int(ei.n_residues),
        "n_bonds": int(ei.n_bonds),
        "n_qubits": int(ei.n_bonds) * 6,
        "n_env_atoms": int(getattr(env_ctx, "n_env_atoms", 0)),
        "crystal_leakage_mode": corridor_ctx.crystal_leakage_mode,
        "fragment_ca_centroid_from_native": centroid is not None,
    }

    budget = PriorArmBudget(
        mode=BudgetMode(args.budget_mode),
        n_proposals=args.n_proposals,
        n_valid_target=args.n_valid_target,
        max_attempts=args.max_attempts,
        chunk_size=args.chunk_size,
    )
    if args.arm == "P":
        arm = PriorArmBaseSampler(
            env_ctx=env_ctx, corridor_ctx=corridor_ctx, budget=budget,
            seed=seed, task_id=task.task_id,
        )
    else:
        arm = MCMCArmBaseSampler(
            env_ctx=env_ctx, corridor_ctx=corridor_ctx, budget=budget,
            seed=seed, suffix_move_prob=args.suffix_move_prob,
            burn_in=args.burn_in, thin=args.thin, task_id=task.task_id,
        )

    # The scoring stack is built exactly as the production runner builds it.
    # Acceptance, validity, and energy must come from the same objects for
    # every arm, or the comparison measures the scorer rather than the
    # sampler.
    mj = load_mj_table_default()
    fh = FilterHamiltonian(residue_contact_weights=mj)
    scorer = FullEnergyScorer(
        residue_contact_weights=mj, rg_target=None,
        term_weights={"overlap_full": 10.0, "contact_full": 1.0,
                      "rg": 0.0, "anchor": 1.0, "turn": 0.0},
    )
    ctx = SamplingContext(
        encoder_inputs=ei, sequence=task.sequence,
        metadata={"case_id": task.task_id, **meta},
    )
    taus = [float(x) for x in args.taus.split(",") if x.strip()]

    its = QuantumImaginaryTimeSampler(
        taus=taus,
        shots_per_tau=int(budget.n_proposals or 0),
        base_sampler=arm,
        filter_hamiltonian=fh,
        scorer=scorer,
        seed=seed,
    )
    logger.info(
        "arm=%s task=%s repeat=%d seed=%d taus=%s budget/tau=%s",
        args.arm, args.task_id, args.repeat, seed, taus, budget.n_proposals,
    )
    batches = its.sample(ctx)

    accepted = [s_ for b in batches for s_ in b.samples
                if s_.accepted and s_.valid]
    cand_path = cell_dir / "candidates.csv"
    with cand_path.open("w", newline="", encoding="utf-8") as fh_out:
        w = csv.writer(fh_out)
        w.writerow(["bitstring", "count", "tau", "filter_energy",
                    "full_energy"])
        for b in batches:
            for s_ in b.samples:
                if not (s_.accepted and s_.valid):
                    continue
                w.writerow([s_.bitstring, s_.count, b.tau,
                            s_.filter_energy, s_.full_energy])

    per_tau = [{
        "tau": b.tau, "n_raw": b.n_raw, "n_valid": b.n_valid,
        "n_accepted": b.n_accepted,
    } for b in batches]
    stats = arm.last_stats
    record.results = {
        "n_accepted_total": len(accepted),
        "n_unique_accepted": len({s_.bitstring for s_ in accepted}),
        "per_tau": per_tau,
        "taus": taus,
        "budget_per_tau": int(budget.n_proposals or 0),
        "budget_total_across_taus": int(budget.n_proposals or 0) * len(taus),
        "candidates_path": str(cand_path.relative_to(root))
        if cand_path.is_relative_to(root) else str(cand_path),
        "last_tau_arm_stats": (stats.as_dict() if stats else {}),
    }
    record.finish()
    if stats and not stats.target_met:
        record.note(
            f"VALID budget not met for arm {args.arm}: target {stats.target} "
            f"not reached. Reported, not truncated silently."
        )
    if args.arm == "M" and stats and not stats.chain_initialized:
        record.note(
            "Arm M chain could not be initialised inside the support; "
            "no chain was run and no candidates were produced."
        )
    write_run_record(record, cell_dir / "run_record.json")

    logger.info(
        "done arm=%s: accepted=%d unique=%d taus=%d wall=%.1fs -> %s",
        args.arm, record.results["n_accepted_total"],
        record.results["n_unique_accepted"], len(taus),
        record.timing.get("wall_seconds", -1), cell_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
