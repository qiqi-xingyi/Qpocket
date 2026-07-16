# Author: Yuqi Zhang
"""EncoderBaseSampler — produce raw CandidateSamples from random codes.

This sampler does NOT decode and does NOT score. It only produces the
bitstring/codes representation. Decoding, validity checking, and scoring
happen downstream (validity.decode_and_validate, FilterHamiltonian,
FullEnergyScorer).

Mode "random_codes" (the only mode in the current cut):
  - Each bond k draws an independent code uniformly from [0, N_DIRECTIONS).
  - N_DIRECTIONS = 64 from ras_folding/encoder/lattice.py.
  - bitstring is packed MSB-first to match decoder.py:_bitstring_to_codes.

The number of bonds is taken from encoder_inputs.n_bonds.

Reproducibility
---------------
Passing ``seed`` to __init__ or to sample() yields a deterministic
sequence of raw samples. Passing seed at sample() takes precedence over
the constructor seed.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from ras_folding.encoder.lattice import N_DIRECTIONS
from ras_folding.sampler.context import get_encoder_inputs
from ras_folding.sampler.sample_types import (
    CandidateSample,
    codes_to_bitstring_str,
)


_SUPPORTED_MODES = ("random_codes",)


class EncoderBaseSampler:
    """Generate raw bitstring / codes candidates compatible with the encoder."""

    def __init__(
        self,
        n_samples: int = 4096,
        seed: Optional[int] = None,
        mode: str = "random_codes",
    ) -> None:
        if n_samples <= 0:
            raise ValueError(f"n_samples must be > 0, got {n_samples}")
        if mode not in _SUPPORTED_MODES:
            raise ValueError(
                f"unsupported mode {mode!r}; supported = {_SUPPORTED_MODES}"
            )
        self.n_samples = int(n_samples)
        self.seed = seed
        self.mode = mode

    def sample(
        self,
        context_or_inputs,
        n_samples: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[CandidateSample]:
        """Produce raw CandidateSamples.

        Parameters
        ----------
        context_or_inputs : SamplingContext or EncoderInputs (anything
            exposing ``n_bonds`` after unwrap).
        n_samples : optional override of the constructor n_samples
        seed : optional override of the constructor seed

        Returns
        -------
        list of CandidateSample with ``codes`` AND ``bitstring`` populated,
        ``coords=None``, ``valid=False``, ``accepted=False``. metadata
        contains the mode and the seed effectively used.
        """
        n = int(n_samples) if n_samples is not None else self.n_samples
        if n <= 0:
            raise ValueError(f"n_samples must be > 0, got {n}")

        eff_seed = seed if seed is not None else self.seed
        rng = np.random.default_rng(eff_seed)

        encoder_inputs = get_encoder_inputs(context_or_inputs)
        n_bonds = int(encoder_inputs.n_bonds)
        if n_bonds < 0:
            raise ValueError(f"n_bonds must be >= 0, got {n_bonds}")

        out: List[CandidateSample] = []
        if self.mode == "random_codes":
            # Vectorized draw: shape (n, n_bonds), values in [0, N_DIRECTIONS)
            if n_bonds > 0:
                codes_arr = rng.integers(
                    low=0, high=N_DIRECTIONS, size=(n, n_bonds), dtype=np.int64,
                )
            else:
                codes_arr = np.zeros((n, 0), dtype=np.int64)
            for i in range(n):
                codes_i = codes_arr[i].tolist()
                bs = codes_to_bitstring_str(codes_i, n_bonds) if n_bonds else ""
                out.append(CandidateSample(
                    bitstring=bs,
                    codes=codes_i,
                    coords=None,
                    count=1,
                    accepted=False,
                    valid=False,
                    metadata={
                        "mode": self.mode,
                        "seed": eff_seed,
                        "draw_index": i,
                    },
                ))
        else:
            # Defensive — should be unreachable due to __init__ check.
            raise ValueError(f"unsupported mode {self.mode!r}")

        return out


__all__ = ["EncoderBaseSampler"]
