#!/usr/bin/env python3
# Author: Yuqi Zhang
"""Score one cell to the frozen G1 endpoints.

Takes the candidates an arm produced and carries them through the rest of
the production pipeline -- densification, subspace refinement, basin
clustering, and the frozen selector -- then measures the result against
the native fragment.

Every stage is the production module at its production default. Nothing
here is a reimplementation, because the arms are only comparable if what
follows them is identical, and a stage rewritten for this comparison
would be a difference between the arms rather than a control on them.

Native coordinates are read once, after selection, to score the structure
the selector already chose. They do not reach any stage that generates,
ranks, or selects. See `configs/G1_FROZEN_ENDPOINTS.md`.
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oracle_eval.rmsd_oracle import _kabsch                       # noqa: E402
from ras_folding.densify.dense_filler import PerturbationDenseFiller  # noqa: E402
from ras_folding.kras.task_loader import load_kras_tasks          # noqa: E402
from ras_folding.postprocess.prediction_postprocessor import (    # noqa: E402
    PredictionPostProcessor,
)
from ras_folding.refinement.subspace_diagonalization import (     # noqa: E402
    SubspaceDiagonalizationRefiner,
)
from ras_folding.sampler.context import SamplingContext           # noqa: E402
from ras_folding.sampler.sample_types import (                    # noqa: E402
    CandidateSample, SampleBatch,
)
from ras_folding.sampler.validity import decode_and_validate      # noqa: E402
from ras_folding.scoring.full_energy import FullEnergyScorer      # noqa: E402
from ras_folding.scoring.mj_contact import load_mj_table_default  # noqa: E402
from revision.runrecord import RunRecord, write_run_record        # noqa: E402

logger = logging.getLogger("revision.score_cell")

# Production defaults, mirrored from ras_folding.kras.full_batch_runner.
DENSIFY = dict(top_parents=50, children_per_parent=8,
               angular_sigmas_deg=(1.5, 3.0, 5.0), max_local_rmsd=1.0,
               energy_window=10.0, perturbation_mass=0.3)
REFINER = dict(max_subspace_size=200, k_neighbors=10, kappa=0.2, n_modes=5,
               max_dense_fraction=0.6, min_original_fraction=0.3,
               coupling_mode="hybrid", g_quantum=0.03,
               alpha_pauli=0.5, alpha_rmsd=0.5)
POSTPROCESS = dict(dedup_rmsd_threshold=0.5, basin_rmsd_threshold=1.5,
                   top_k_candidates=20, top_k_basins=5)


def inframe_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    """Primary endpoint: CA RMSD with no superposition."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(math.sqrt(max(float(np.mean(np.sum(d * d, axis=1))), 0.0)))


def kabsch_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    """Reported beside the primary endpoint; shape only, placement discarded."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ac, bc = a - a.mean(0), b - b.mean(0)
    d = (ac @ _kabsch(ac, bc)) - bc
    return float(math.sqrt(max(float(np.mean(np.sum(d * d, axis=1))), 0.0)))


def pool_best(coords: np.ndarray, native: np.ndarray) -> Dict[str, Any]:
    """Lowest RMSD anywhere in the pool.

    A coverage measure, not accuracy: obtaining it needs the native
    structure the method does not have. Vectorised for the in-frame
    metric; the Kabsch figure is taken at the in-frame winner rather than
    re-minimised, so the two describe the same structure.
    """
    if coords.size == 0:
        return {"n_pool": 0, "inframe": None, "kabsch_at_inframe_best": None,
                "argmin_index": None}
    d = coords - native[None, :, :]
    per = np.sqrt(np.mean(np.sum(d * d, axis=2), axis=1))
    i = int(np.argmin(per))
    return {
        "n_pool": int(coords.shape[0]),
        "inframe": float(per[i]),
        "kabsch_at_inframe_best": kabsch_rmsd(coords[i], native),
        "argmin_index": i,
    }


def load_candidates(path: Path, ctx, limit: Optional[int]) -> List[CandidateSample]:
    """Rebuild scored candidates and decode them through the shared decoder.

    The energies already in the file are kept; the coordinates are not
    stored there and are recovered with the same decoder every arm uses.
    """
    out: List[CandidateSample] = []
    n_decode_fail = 0
    with path.open(newline="", encoding="utf-8") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            if limit is not None and i >= limit:
                break
            bs = (row.get("bitstring") or "").strip()
            if not bs:
                continue

            def _f(key):
                v = row.get(key)
                try:
                    return float(v) if v not in (None, "", "None") else None
                except ValueError:
                    return None

            s = CandidateSample(
                bitstring=bs, count=int(float(row.get("count") or 1)),
                filter_energy=_f("filter_energy"),
                full_energy=_f("full_energy"),
                metadata={"tau": row.get("tau")},
            )
            decode_and_validate(s, ctx)
            if not s.valid or s.coords is None:
                n_decode_fail += 1
                continue
            s.accepted = True
            out.append(s)
    if n_decode_fail:
        logger.warning("%d candidates failed to decode and were dropped",
                       n_decode_fail)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", required=True, choices=["Q", "P", "M"])
    p.add_argument("--task-id", required=True)
    p.add_argument("--repeat", type=int, default=0)
    p.add_argument("--results-root", default="revision/results/g1_pilot")
    p.add_argument("--tasks", default="inputs/kras_tasks.csv")
    p.add_argument("--structure-root", default="kras_select_systems")
    p.add_argument("--limit", type=int, default=None,
                   help="cap candidates read; diagnostic only, and recorded "
                        "in the endpoint record when used")
    p.add_argument("--no-densify", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s | %(message)s")

    root = PROJECT_ROOT
    res_root = Path(args.results_root)
    if not res_root.is_absolute():
        res_root = root / res_root
    cell = res_root / args.arm / args.task_id / f"repeat_{args.repeat}"
    cand_path = cell / "candidates.csv"
    if not cand_path.is_file():
        logger.error("no candidates at %s", cand_path)
        return 2

    record = RunRecord(
        experiment="g1_kras_pilot_scoring", arm=args.arm,
        task_id=args.task_id,
        config={"repeat": args.repeat, "limit": args.limit,
                "densify": not args.no_densify,
                "densify_params": DENSIFY, "refiner_params": REFINER,
                "postprocess_params": POSTPROCESS,
                "selector": "rank_basins(basin_weight_bonus=0.5)",
                "primary_endpoint": "inframe_ca_rmsd_no_superposition"},
    ).start(root)

    tasks, _ = load_kras_tasks(root / args.tasks,
                               pdb_dir=root / args.structure_root)
    task = next((t for t in tasks if t.task_id == args.task_id), None)
    if task is None:
        logger.error("task_id %r not found", args.task_id)
        return 2
    ei = task.encoder_inputs
    ctx = SamplingContext(encoder_inputs=ei, sequence=task.sequence,
                          metadata={"case_id": task.task_id, **task.metadata})

    cands = load_candidates(cand_path, ctx, args.limit)
    logger.info("decoded %d candidates", len(cands))
    if not cands:
        record.note("No candidate decoded; no endpoint could be computed.")
        record.finish()
        write_run_record(record, cell / "endpoint_record.json")
        return 3

    mj = load_mj_table_default()
    scorer = FullEnergyScorer(
        residue_contact_weights=mj, rg_target=None,
        term_weights={"overlap_full": 10.0, "contact_full": 1.0,
                      "rg": 0.0, "anchor": 1.0, "turn": 0.0})

    # Any candidate missing a full energy cannot enter refinement.
    for s in cands:
        if s.full_energy is None:
            e, terms = scorer.evaluate(s, ctx)
            s.full_energy = float(e)
            s.full_energy_terms = terms

    pool = cands
    densify_summary = None
    if not args.no_densify:
        filler = PerturbationDenseFiller(seed=20260825, **DENSIFY)
        dense = filler.densify(cands, scorer, ctx)
        densify_summary = dense.summary
        pool = dense.all_candidates
        logger.info("densified: %d -> %d", len(cands), len(pool))

    batch = SampleBatch(
        samples=pool, tau=None, n_raw=len(pool),
        n_valid=sum(1 for s in pool if s.valid),
        n_accepted=sum(1 for s in pool if s.accepted),
        summary={"source": "revision.score_cell"},
    )
    refiner = SubspaceDiagonalizationRefiner(
        n_qubits=int(ei.n_bonds) * 6, **REFINER)
    refined = refiner.refine(batch)
    logger.info("refined to %d candidates", len(refined.candidates))

    post = PredictionPostProcessor(export_pdb=False, **POSTPROCESS)
    prediction = post.process(refined, output_dir=cell / "postprocess",
                              sequence=task.sequence)

    # --- endpoints -------------------------------------------------- #
    native = (np.asarray(task.reference_coords, dtype=np.float64)
              if task.reference_coords is not None else None)
    endpoints: Dict[str, Any] = {
        "native_available": native is not None,
        "n_basins": len(prediction.basin_summaries),
    }

    if native is None:
        record.note(
            "No native reference for this task; the endpoints are undefined "
            "and the cell contributes only to coverage counts."
        )
    else:
        ranked = prediction.basin_summaries          # already selector-ordered
        top = ranked[0] if ranked else None
        rep = None
        if top is not None and prediction.basin_representatives:
            idx = min(int(top.representative_index),
                      len(prediction.basin_representatives) - 1)
            rep = prediction.basin_representatives[idx]
        rep_coords = getattr(getattr(rep, "sample", None), "coords", None)

        if rep_coords is not None:
            endpoints["primary"] = {
                "inframe_rmsd": inframe_rmsd(rep_coords, native),
                "kabsch_rmsd": kabsch_rmsd(rep_coords, native),
                "basin_id": int(top.basin_id),
                "basin_rank": int(top.basin_rank),
                "basin_score": top.metadata.get("basin_score"),
                "basin_weight": float(top.basin_weight),
                "best_refined_score": float(top.best_refined_score),
            }
        else:
            record.note("Selector produced no representative structure; "
                        "the primary endpoint is undefined for this cell.")

        # Every ranked basin, so the cost of the ranking is visible rather
        # than only its top choice.
        per_basin = []
        for b in ranked:
            r = None
            if prediction.basin_representatives:
                j = min(int(b.representative_index),
                        len(prediction.basin_representatives) - 1)
                c = getattr(getattr(prediction.basin_representatives[j],
                                    "sample", None), "coords", None)
                r = inframe_rmsd(c, native) if c is not None else None
            per_basin.append({
                "basin_rank": int(b.basin_rank), "basin_id": int(b.basin_id),
                "basin_weight": float(b.basin_weight),
                "basin_score": b.metadata.get("basin_score"),
                "inframe_rmsd": r,
            })
        endpoints["per_basin"] = per_basin

        # Over the arm's own candidates, not the densified subset.
        # Densification keeps only a neighbourhood of the best parents, so
        # measuring coverage there would report the reach of the densifier
        # rather than the reach of the sampler under test.
        coords = np.stack([s.coords for s in cands], axis=0)
        endpoints["secondary_oracle_best"] = pool_best(coords, native)
        endpoints["secondary_oracle_best"]["note"] = (
            "coverage and reachability over the arm's own candidates, "
            "not predictive accuracy"
        )

    record.inputs = {
        "candidates_path": str(cand_path.relative_to(root))
        if cand_path.is_relative_to(root) else str(cand_path),
        "n_candidates_decoded": len(cands),
        "n_pool_after_densify": len(pool),
        "n_refined": len(refined.candidates),
        "n_residues": int(ei.n_residues), "n_bonds": int(ei.n_bonds),
    }
    record.results = {"endpoints": endpoints,
                      "densify_summary": densify_summary,
                      "refinement_summary": dict(refined.summary)}
    record.finish()
    if args.limit is not None:
        record.note(f"Candidates were capped at {args.limit}; this cell is "
                    f"not comparable with uncapped cells.")
    write_run_record(record, cell / "endpoint_record.json")

    prim = endpoints.get("primary") or {}
    logger.info("arm=%s task=%s repeat=%d | inframe=%.3f kabsch=%.3f | "
                "oracle-best inframe=%.3f | wall=%.1fs",
                args.arm, args.task_id, args.repeat,
                prim.get("inframe_rmsd", float("nan")),
                prim.get("kabsch_rmsd", float("nan")),
                (endpoints.get("secondary_oracle_best") or {}).get(
                    "inframe") or float("nan"),
                record.timing.get("wall_seconds", -1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
