# Author: Yuqi Zhang
"""Oracle / theoretical-anchor injection helpers.

A scientifically faithful way to inject a control "lower-bound" CA path
into the V2 candidate ensemble. The anchor MUST be labeled
``source ∈ {'oracle_anchor', 'theoretical_lower_bound'}`` and MUST NOT
be counted toward generated metrics (oracle_best_kabsch on generated
pool, frac_lt_2A_generated, etc.).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np


@dataclass
class OracleAnchor:
    coords: np.ndarray              # (n_res, 3) CA only
    sequence: str
    source: str                     # "theoretical_lower_bound" | "oracle_anchor"
    anchor_type: str                # "native_reference" | "s7_upper_bound" | ...
    candidate_id: str
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_summary(self) -> Dict[str, object]:
        return {
            "source": self.source,
            "anchor_type": self.anchor_type,
            "candidate_id": self.candidate_id,
            "n_residues": int(self.coords.shape[0]),
            "metadata": dict(self.metadata),
        }

    def write_pdb(self, pdb_path: Path) -> None:
        """Write CA-only PDB. Standard residue ALA placeholder for atoms."""
        pdb_path = Path(pdb_path)
        pdb_path.parent.mkdir(parents=True, exist_ok=True)
        n = int(self.coords.shape[0])
        # Use sequence to set residue names if available; else ALA
        from ras_folding.utils.data_loader import _THREE_TO_ONE
        one_to_three = {v: k for k, v in _THREE_TO_ONE.items()}
        with pdb_path.open("w", encoding="utf-8") as f:
            f.write(
                f"REMARK  oracle_anchor: source={self.source} "
                f"anchor_type={self.anchor_type} "
                f"candidate_id={self.candidate_id}\n"
            )
            for i in range(n):
                aa1 = self.sequence[i] if i < len(self.sequence) else "A"
                aa3 = one_to_three.get(aa1, "ALA")
                x, y, z = self.coords[i]
                f.write(
                    f"ATOM  {i+1:5d}  CA  {aa3} A{i+1:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
                )
            f.write("END\n")


def build_native_oracle_anchor(
    native_ca: np.ndarray,
    sequence: str,
    *,
    candidate_id: str = "oracle_anchor_native",
    metadata: Optional[Dict[str, object]] = None,
) -> OracleAnchor:
    """Construct an OracleAnchor from native fragment CA coords.

    source = 'theoretical_lower_bound' (the truest anchor: native).
    """
    if native_ca.ndim != 2 or native_ca.shape[1] != 3:
        raise ValueError(
            f"native_ca must have shape (N, 3); got {native_ca.shape}"
        )
    if native_ca.shape[0] != len(sequence):
        raise ValueError(
            f"native_ca rows ({native_ca.shape[0]}) != sequence length "
            f"({len(sequence)})"
        )
    md = {"origin": "native_reference"}
    if metadata:
        md.update(metadata)
    return OracleAnchor(
        coords=np.asarray(native_ca, dtype=np.float64).copy(),
        sequence=str(sequence),
        source="theoretical_lower_bound",
        anchor_type="native_reference",
        candidate_id=candidate_id,
        metadata=md,
    )


def gap_to_anchor_summary(
    anchor: OracleAnchor,
    generated_oracle_best_kabsch: Optional[float],
) -> Dict[str, object]:
    return {
        "source": anchor.source,
        "anchor_type": anchor.anchor_type,
        "anchor_kabsch_rmsd_to_native": 0.0,  # by construction if native
        "generated_oracle_best_kabsch": generated_oracle_best_kabsch,
        "gap_generated_to_anchor": (
            generated_oracle_best_kabsch
            if generated_oracle_best_kabsch is not None else None
        ),
    }


__all__ = [
    "OracleAnchor",
    "build_native_oracle_anchor",
    "gap_to_anchor_summary",
]
