# Author: Yuqi Zhang
"""Revision-stage engineering code for the Qpocket PNAS revision.

This package contains ONLY the comparison arms, run-record machinery, and
OSC submission scaffolding that the revision adds. It does not modify or
re-implement the method itself: every arm reuses the production modules in
``ras_folding`` unchanged, so the downstream is shared by construction.

Layout
------
arms/       alternative base samplers (P, M) implementing the same contract
            as ``ras_folding.sampler.quantum_base_sampler``
runrecord/  input manifest + environment/provenance capture
configs/    frozen per-experiment configuration
osc/        SLURM submission scripts for the OSC Cardinal cluster
results/    run outputs (git-ignored)
"""
