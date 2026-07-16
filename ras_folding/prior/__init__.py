# Author: Yuqi Zhang
"""V2 corridor-conditioned prior package.

Modules
-------
environment       : EnvironmentPriorContext + builder (PDB → KDTree + ligand)
corridor          : CorridorPriorContext (C4 ligand-anchor pocket Bézier)
direction_policy  : per-step PriorConditionedDirectionPolicy (env + corridor)
prior_sampler     : PriorConditionedBaseSampler — path rollout → bitstrings
                    + bit_marginals for HEA moment matching
diagnostics       : write prior/ outputs (config, summary, bit marginals,
                    step stats, sampled paths)
shot_budget       : length-adaptive shot allocation for V2
oracle_anchor     : oracle anchor injection helpers (theoretical lower bound)
landscape         : V2 landscape candidate file writer

V1 backward compatibility:
    Importing this package does NOT change V1 behaviour. None of these
    modules touch ras_folding.encoder, ras_folding.scoring, or any
    pipeline orchestrator. They are opt-in V2 components.
"""
from __future__ import annotations

from ras_folding.prior.environment import (
    EnvironmentPriorContext,
    build_environment_prior,
)
from ras_folding.prior.corridor import (
    CorridorPriorContext,
    build_corridor_prior,
)
from ras_folding.prior.direction_policy import (
    PriorDirectionResult,
    PriorConditionedDirectionPolicy,
)
from ras_folding.prior.prior_sampler import (
    PriorSamplingResult,
    PriorConditionedBaseSampler,
)
from ras_folding.prior.shot_budget import (
    allocate_shots,
    SHOT_BUDGET_DEFAULTS,
)
from ras_folding.prior.oracle_anchor import (
    OracleAnchor,
    build_native_oracle_anchor,
)

__all__ = [
    "EnvironmentPriorContext", "build_environment_prior",
    "CorridorPriorContext", "build_corridor_prior",
    "PriorDirectionResult", "PriorConditionedDirectionPolicy",
    "PriorSamplingResult", "PriorConditionedBaseSampler",
    "allocate_shots", "SHOT_BUDGET_DEFAULTS",
    "OracleAnchor", "build_native_oracle_anchor",
]
