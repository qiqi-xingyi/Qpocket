# Author: Yuqi Zhang
"""Reconstruction dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


_VALID_STATUS = ("done", "failed", "skipped")


@dataclass
class ReconstructionInput:
    task_id: str
    predicted_ca_pdb: Path
    reference_pdb: Path
    chain_id: str
    start_resi: int
    end_resi: int
    sequence: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReconstructionResult:
    task_id: str
    predicted_ca_pdb: Path
    rebuilt_fragment_pdb: Optional[Path]
    embedded_receptor_pdb: Optional[Path]
    status: str
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in _VALID_STATUS:
            raise ValueError(
                f"status must be one of {_VALID_STATUS}; got {self.status!r}"
            )


__all__ = ["ReconstructionInput", "ReconstructionResult"]
