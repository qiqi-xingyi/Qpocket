# Author: Yuqi Zhang
"""counts_to_candidate_samples — merge backend counts into CandidateSamples.

Bit-order convention
--------------------
Qiskit's ``get_counts()`` returns bitstrings whose LEFTMOST character
corresponds to the HIGHEST qubit index. Our circuit_builder lays
qubits as ``qubit_index = bond_index * bits_per_step + bit_offset``,
where ``bit_offset == 0`` is the MSB of that bond's code.

To match the decoder's MSB-first / bond-0-first bitstring convention
used by ``ras_folding.encoder.decoder.decode_bitstring``, the Qiskit
output must be REVERSED. This is the default ``bit_order`` and is the
only place that translation happens — circuit_builder and the rest of
the pipeline never reverse anything.
"""
from __future__ import annotations

import warnings
from typing import Any, List, Optional

from ras_folding.encoder.decoder import BITS_PER_BOND
from ras_folding.quantum.result_types import QuantumBackendResult
from ras_folding.sampler.context import get_encoder_inputs
from ras_folding.sampler.sample_types import (
    CandidateSample,
    bitstring_int_to_codes,
    bitstring_str_to_int,
)


_VALID_BIT_ORDERS = ("qiskit_little_endian", "msb_first")


def counts_to_candidate_samples(
    backend_result: QuantumBackendResult,
    context_or_inputs,
    bit_order: str = "qiskit_little_endian",
    *,
    bits_per_step: int = BITS_PER_BOND,
) -> List[CandidateSample]:
    """Merge counts from a QuantumBackendResult into CandidateSamples.

    Parameters
    ----------
    backend_result : QuantumBackendResult — output of an aer or ibm backend run.
    context_or_inputs : SamplingContext or EncoderInputs — used to validate
        bitstring length (must equal n_bonds * bits_per_step).
    bit_order :
        - "qiskit_little_endian" (default): backend bitstring rightmost
          char = qubit 0 = bond 0 MSB → reverse to get MSB-first.
        - "msb_first": already in the decoder's MSB-first convention.
    bits_per_step : usually BITS_PER_BOND=6.

    Returns
    -------
    list[CandidateSample] — one per UNIQUE decoder-convention bitstring,
        with ``count`` accumulated across all source circuits and
        ``base_probability = count / total_shots``. ``coords`` is None,
        ``valid`` is False, ``accepted`` is False.
    """
    if bit_order not in _VALID_BIT_ORDERS:
        raise ValueError(
            f"bit_order must be one of {_VALID_BIT_ORDERS}; got {bit_order!r}"
        )

    encoder_inputs = get_encoder_inputs(context_or_inputs)
    n_bonds = int(encoder_inputs.n_bonds)
    expected_len = n_bonds * int(bits_per_step)

    accumulator: dict = {}     # bs_str (decoder-convention) → int count
    sources: dict = {}         # bs_str → set of circuit names
    total_shots = 0
    skipped_len = 0
    skipped_examples: List[str] = []

    for cc in backend_result.circuit_counts:
        # honor the per-result shots count — sum of counts may differ if
        # the backend reports "skipped" or non-meas bits, but for our
        # measurement circuits sum(counts.values()) == shots in normal cases.
        for raw_bs, n in cc.counts.items():
            bs = _normalize_bitstring(raw_bs)
            if bit_order == "qiskit_little_endian":
                bs = bs[::-1]
            if len(bs) != expected_len:
                skipped_len += int(n)
                if len(skipped_examples) < 4:
                    skipped_examples.append(bs)
                continue
            accumulator[bs] = accumulator.get(bs, 0) + int(n)
            sources.setdefault(bs, set()).add(cc.circuit_name)
        total_shots += int(cc.shots)

    if skipped_len > 0:
        warnings.warn(
            f"counts_to_candidate_samples skipped {skipped_len} shots "
            f"with bitstring length != {expected_len} "
            f"(examples: {skipped_examples})",
            stacklevel=2,
        )

    if total_shots == 0:
        return []

    samples: List[CandidateSample] = []
    for bs_str, count in accumulator.items():
        try:
            bs_int = bitstring_str_to_int(bs_str)
            codes = bitstring_int_to_codes(bs_int, n_bonds).tolist()
        except Exception as e:
            warnings.warn(
                f"failed to parse bitstring {bs_str!r}: {e}",
                stacklevel=2,
            )
            continue
        samples.append(CandidateSample(
            bitstring=bs_str,
            codes=codes,
            coords=None,
            count=int(count),
            base_probability=float(count) / float(total_shots),
            accepted=False,
            valid=False,
            metadata={
                "backend_type": backend_result.backend_type,
                "backend_name": backend_result.backend_name,
                "execution_mode": backend_result.execution_mode,
                "job_ids": list(backend_result.job_ids),
                "source_circuit_names": sorted(sources.get(bs_str, [])),
                "total_shots": total_shots,
            },
        ))
    return samples


def _normalize_bitstring(s: str) -> str:
    """Strip whitespace / spaces (Qiskit sometimes inserts spaces between
    classical registers). Validate it's a binary string."""
    s = "".join(s.split())
    if not all(c in "01" for c in s):
        raise ValueError(f"non-binary char in bitstring {s!r}")
    return s


__all__ = ["counts_to_candidate_samples"]
