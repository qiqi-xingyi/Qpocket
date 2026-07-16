# Author: Yuqi Zhang
"""Portable synthetic test fixtures for the pipeline.

This module is the single source of truth for ``make_test_case`` — a
small, dependency-light EncoderInputs + SamplingContext builder used by
the module self-tests (e.g. ``ras_folding.quantum.moment_match_initializer``)
and by ``examples.run_smoke``.

It depends ONLY on ``ras_folding.encoder.inputs`` and
``ras_folding.sampler.context`` so that ``full_pipline`` is fully
self-contained (no dependency on the legacy ``pauli_analysis`` package).
"""
from __future__ import annotations

import numpy as np

from ras_folding.encoder.inputs import EncoderInputs
from ras_folding.sampler.context import SamplingContext


def make_test_case(n_residues: int):
    """Construct a small but non-trivial EncoderInputs + sequence + context.

    Geometry is hand-picked so that the case is reachable (anchor in
    range of the autoregressive walk) but non-trivial (most bitstrings
    will miss the endpoint, giving a rich H_filter landscape).

    Returns
    -------
    (EncoderInputs, SamplingContext)
        For ``n_residues=3`` the encoder has 2 bonds → 12 qubits, so the
        HEA θ vector for ``reps=1`` has shape ``(24,)``.
    """
    if n_residues == 3:
        # 2 bonds, 12 qubits
        anchor_left = np.array([0.0, 0.0, 0.0])
        anchor_right = np.array([4.0, 2.0, 0.0])   # ~4.47 Å
        v_left_seed = np.array([1.0, 0.0, 0.0])
        v_right_seed = np.array([1.0, 0.0, 0.0])
        seq = "MAG"
    elif n_residues == 4:
        # 3 bonds, 18 qubits
        anchor_left = np.array([0.0, 0.0, 0.0])
        anchor_right = np.array([6.0, 3.0, 0.0])   # ~6.71 Å
        v_left_seed = np.array([1.0, 0.0, 0.0])
        v_right_seed = np.array([1.0, 0.0, 0.0])
        seq = "MAGV"
    elif n_residues == 5:
        # 4 bonds, 24 qubits
        anchor_left = np.array([0.0, 0.0, 0.0])
        anchor_right = np.array([8.0, 3.0, 0.0])   # ~8.54 Å
        v_left_seed = np.array([1.0, 0.0, 0.0])
        v_right_seed = np.array([1.0, 0.0, 0.0])
        seq = "MAGVA"
    else:
        raise ValueError(
            f"n_residues must be 3, 4, or 5 for this prototype; got {n_residues}"
        )

    enc = EncoderInputs(
        n_residues=n_residues,
        anchor_left=anchor_left,
        anchor_right=anchor_right,
        v_left_seed=v_left_seed / np.linalg.norm(v_left_seed),
        v_right_seed=v_right_seed / np.linalg.norm(v_right_seed),
    )
    ctx = SamplingContext(
        encoder_inputs=enc, sequence=seq,
        metadata={"case_id": f"synthetic_n{n_residues}"},
    )
    return enc, ctx


__all__ = ["make_test_case"]
