# Author: Yuqi Zhang
"""SamplingContext — bundles EncoderInputs + sequence + free-form metadata
without modifying the encoder's frozen EncoderInputs dataclass.

Two helpers (``get_encoder_inputs`` / ``get_sequence``) let consumers
accept EITHER a SamplingContext OR a raw EncoderInputs without branching.

This module intentionally has no dependency on scoring or refinement —
it sits at the same layer as encoder.inputs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Union

from ras_folding.encoder.inputs import EncoderInputs


@dataclass(frozen=True)
class SamplingContext:
    """Bundle the geometric encoder context with sequence + provenance.

    Fields
    ------
    encoder_inputs : EncoderInputs
        The raw geometric inputs the encoder consumes.
    sequence : str | list[str] | None
        One-letter (str) or 3-letter (list[str]) per-residue sequence.
        None ⇒ MJ-aware terms gracefully degrade to 0 with metadata flags.
    metadata : dict
        Free-form provenance (case_id, ref_pdb, etc.). NEVER consumed by
        the sampler/scorer/refiner — purely for traceability.
    """
    encoder_inputs: EncoderInputs
    sequence: Optional[Union[str, Sequence[str]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Forward the most-used encoder fields so simple callers can still
    # write ``ctx.n_bonds`` without unwrapping. NOTE: do NOT add fields
    # that conflict with EncoderInputs (e.g. anchor_left) — consumers
    # that need geometric fields must call get_encoder_inputs() first.
    @property
    def n_residues(self) -> int:
        return self.encoder_inputs.n_residues

    @property
    def n_bonds(self) -> int:
        return self.encoder_inputs.n_bonds


# ---------------------------------------------------------------------- #
# helpers                                                                #
# ---------------------------------------------------------------------- #

def get_encoder_inputs(obj: Any) -> EncoderInputs:
    """Return the raw EncoderInputs from either a SamplingContext or a
    direct EncoderInputs (or anything else exposing ``.encoder_inputs``)."""
    if obj is None:
        raise ValueError("get_encoder_inputs(None)")
    if isinstance(obj, EncoderInputs):
        return obj
    inner = getattr(obj, "encoder_inputs", None)
    if inner is not None:
        return inner
    # If `obj` quacks like an EncoderInputs (has n_residues, anchor_left,
    # ...), accept it. This keeps tests/synthetic objects working.
    if hasattr(obj, "n_residues") and hasattr(obj, "anchor_left"):
        return obj  # type: ignore[return-value]
    raise TypeError(
        f"Cannot extract EncoderInputs from {type(obj).__name__}"
    )


def get_sequence(obj: Any) -> Optional[Union[str, Sequence[str]]]:
    """Return the sequence attribute if present, else None.

    Looks at obj.sequence first, then tries getattr — never raises."""
    if obj is None:
        return None
    return getattr(obj, "sequence", None)


__all__ = ["SamplingContext", "get_encoder_inputs", "get_sequence"]
