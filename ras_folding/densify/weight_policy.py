# Author: Yuqi Zhang
"""Per-parent mass-conserving weight policy for dense candidates.

For each parent that produced one or more children:
    parent_weight   = (1 - perturbation_mass) * parent_initial_weight
    child_weight    = (perturbation_mass    * parent_initial_weight) / n_children

Parents without children keep their initial weight unchanged.

The "initial" parent weight is read in this order:
    1. parent.metadata.get("refined_weight")
    2. parent.base_probability
    3. parent.count                 (renormalised across all parents)
    4. uniform 1 / n_parents        (final fallback)

Important: this policy is PER-PARENT mass-conserving — each parent's
total support stays at parent_initial_weight regardless of how many
children it has. The policy does NOT renormalize across parents
(the refiner has its own normalization step). This means dense
candidates cannot inflate basin support by sheer count — that's the
point of the policy.
"""
from __future__ import annotations

from typing import List, Sequence

from ras_folding.sampler.sample_types import CandidateSample


def _parent_initial_weight(parent: CandidateSample) -> float:
    """Resolve a parent's initial weight via the documented fallback chain."""
    rw = parent.metadata.get("refined_weight")
    if isinstance(rw, (int, float)) and rw > 0:
        return float(rw)
    if parent.base_probability is not None and parent.base_probability > 0:
        return float(parent.base_probability)
    if parent.count and parent.count > 0:
        return float(parent.count)
    return 1.0  # uniform fallback (renormalisation across parents not done here)


def assign_dense_weights(
    parent_candidates: Sequence[CandidateSample],
    dense_candidates: Sequence[CandidateSample],
    perturbation_mass: float = 0.3,
) -> None:
    """Mutate ``parent_candidates`` and ``dense_candidates`` metadata in
    place to record per-parent mass-conserving weights.

    Parameters
    ----------
    parent_candidates : iterable of CandidateSample
        The parents that were considered. Each gets
        ``metadata["dense_parent_weight_after_split"]`` set.
    dense_candidates : iterable of CandidateSample
        The children produced by the filler. Each must already have
        ``metadata["densify"]["parent_bitstring"]`` set so we can group
        children by parent. Each gets
        ``metadata["densify"]["dense_weight"]`` set.
    perturbation_mass : float in [0, 1]
        Fraction of each parent's mass redistributed to its children.
    """
    if not (0.0 <= perturbation_mass <= 1.0):
        raise ValueError(
            f"perturbation_mass must be in [0, 1]; got {perturbation_mass}"
        )

    children_by_parent: dict = {}
    for c in dense_candidates:
        d = c.metadata.get("densify") or {}
        parent_bs = d.get("parent_bitstring")
        if parent_bs is None:
            # malformed dense child — surface but do not fail
            c.metadata.setdefault("densify", {})["dense_weight"] = 0.0
            continue
        children_by_parent.setdefault(parent_bs, []).append(c)

    for parent in parent_candidates:
        bs = parent.bitstring
        children = children_by_parent.get(bs, [])
        w0 = _parent_initial_weight(parent)
        if not children:
            parent.metadata["dense_parent_weight_after_split"] = float(w0)
            parent.metadata["dense_initial_weight"] = float(w0)
            continue
        kept_for_parent = (1.0 - perturbation_mass) * w0
        share_per_child = (perturbation_mass * w0) / float(len(children))
        parent.metadata["dense_parent_weight_after_split"] = float(kept_for_parent)
        parent.metadata["dense_initial_weight"] = float(w0)
        for c in children:
            c.metadata.setdefault("densify", {})["dense_weight"] = float(
                share_per_child
            )


__all__ = ["assign_dense_weights"]
