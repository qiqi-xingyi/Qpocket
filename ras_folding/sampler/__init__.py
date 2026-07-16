# Author: Yuqi Zhang
"""ras_folding.sampler — base sampling, validity, filter Hamiltonian,
imaginary-time-inspired sampler.

Public API:
  CandidateSample, SampleBatch       -- sample dataclasses
  EncoderBaseSampler                 -- raw candidate generator
  decode_and_validate, validate_decoded_coords  -- validity wrapper
  FilterHamiltonian                  -- diagonal H_filter for rejection
  QuantumImaginaryTimeSampler        -- imaginary-time-inspired sampler
  codes_to_bitstring_int, bitstring_int_to_codes,
  codes_to_bitstring_str, bitstring_str_to_int
                                     -- bitstring/code conversion utilities
"""
from ras_folding.sampler.context import (
    SamplingContext,
    get_encoder_inputs,
    get_sequence,
)
from ras_folding.sampler.sample_types import (
    CandidateSample,
    SampleBatch,
    bitstring_int_to_codes,
    bitstring_str_to_int,
    codes_to_bitstring_int,
    codes_to_bitstring_str,
)
from ras_folding.sampler.base_sampler import EncoderBaseSampler
from ras_folding.sampler.validity import (
    decode_and_validate,
    validate_decoded_coords,
)
from ras_folding.sampler.filter_hamiltonian import FilterHamiltonian
from ras_folding.sampler.imaginary_time_sampler import (
    QuantumImaginaryTimeSampler,
)

# NOTE: QuantumBackendBaseSampler is intentionally NOT imported eagerly
# here. It depends on ``ras_folding.quantum``, which in turn imports
# ``ras_folding.sampler.context`` — eagerly importing both at package
# load time creates a circular import. Users import it explicitly via:
#   from ras_folding.sampler.quantum_base_sampler import QuantumBackendBaseSampler

__all__ = [
    "SamplingContext",
    "get_encoder_inputs",
    "get_sequence",
    "CandidateSample",
    "SampleBatch",
    "EncoderBaseSampler",
    "decode_and_validate",
    "validate_decoded_coords",
    "FilterHamiltonian",
    "QuantumImaginaryTimeSampler",
    "codes_to_bitstring_int",
    "bitstring_int_to_codes",
    "codes_to_bitstring_str",
    "bitstring_str_to_int",
]
