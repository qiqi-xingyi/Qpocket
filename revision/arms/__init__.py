# Author: Yuqi Zhang
"""Comparison arms for the Qpocket revision.

Every arm implements the base-sampler contract:

    sample(context_or_inputs, n_samples=None, seed=None) -> List[CandidateSample]

returning candidates with ``coords=None``, ``valid=False``,
``accepted=False`` — identical to ``EncoderBaseSampler`` and
``QuantumBackendBaseSampler``. This lets an arm be injected directly as the
``base_sampler`` of ``QuantumImaginaryTimeSampler``, so acceptance,
densification, subspace refinement, ranking, and evaluation are literally
the same code path for all arms.
"""
from revision.arms.prior_arm import (
    BudgetMode, PriorArmBaseSampler, PriorArmBudget,
)
from revision.arms.prior_density import PathDensity, PriorPathEvaluator
from revision.arms.mcmc_arm import MCMCArmBaseSampler, MoveType

__all__ = ["BudgetMode", "PriorArmBaseSampler", "PriorArmBudget",
           "PathDensity", "PriorPathEvaluator",
           "MCMCArmBaseSampler", "MoveType"]
