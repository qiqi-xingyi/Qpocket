# Author: Yuqi Zhang
"""docking_eval dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np


@dataclass
class DockingInput:
    task_id: str
    receptor_pdb: Path
    ligand_source_pdb: Path
    ligand_resname: str
    output_dir: Path
    box_padding: float = 8.0
    min_box_size: float = 20.0
    repeats: int = 5
    exhaustiveness: int = 8
    num_modes: int = 9
    seed: int = 2024
    # Pocket center used to disambiguate multiple residue copies of the
    # same ligand_resname in the reference PDB (see
    # docking_eval.ligand.extract_ligand_from_pdb_with_metadata).
    pocket_center: Optional[Union[np.ndarray, Sequence[float]]] = None
    ligand_selection_mode: str = "nearest_to_pocket_center"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DockingRunResult:
    task_id: str
    receptor_label: str   # "predicted_top1" | "oracle_best"
    affinities_kcal_mol: List[float]
    estimated_kd_m: List[float]
    mean_affinity_kcal_mol: Optional[float]
    std_affinity_kcal_mol: Optional[float]
    mean_kd_m: Optional[float]
    std_kd_m: Optional[float]
    best_affinity_kcal_mol: Optional[float]
    output_files: Dict[str, str] = field(default_factory=dict)
    status: str = "done"
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


__all__ = ["DockingInput", "DockingRunResult"]
