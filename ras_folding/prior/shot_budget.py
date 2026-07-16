# Author: Yuqi Zhang
"""Length-adaptive shot allocation for V2 sampling.

V1 used ~49152 shots/task. V2 scales with fragment length so longer
fragments (where Frag-B/C bottlenecks live) get more samples. The
allocation decomposes total shots into (n_circuits, shots_per_circuit)
honoring backend per-circuit limits.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional


# V2 defaults (revised 2026-04-30): smaller budget, IBM-friendly.
# Length-adaptive: 300k → 500k → 1M → 2M (cap).
SHOT_BUDGET_DEFAULTS: Dict[str, int] = {
    "shots_short": 300_000,      # n_res <= 7
    "shots_medium": 500_000,     # 8 <= n_res <= 10
    "shots_long": 1_000_000,     # 11 <= n_res <= 12
    "shots_xlong": 2_000_000,    # 13 <= n_res <= 15
    "shots_max": 2_000_000,      # n_res > 15 (capped)
    "shots_per_circuit": 4096,
    "min_shots_per_task": 300_000,
    "max_shots_per_task": 2_000_000,
}


@dataclass
class ShotAllocation:
    n_res: int
    n_bonds: int
    n_qubits: int
    budget_mode: str            # "fixed" | "length_adaptive"
    requested_total_shots: int
    allocated_total_shots: int
    shots_per_circuit: int
    n_circuits: int
    n_seeds: int                # reproducibility seeds for circuits
    notes: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "n_res": int(self.n_res),
            "n_bonds": int(self.n_bonds),
            "n_qubits": int(self.n_qubits),
            "budget_mode": self.budget_mode,
            "requested_total_shots": int(self.requested_total_shots),
            "allocated_total_shots": int(self.allocated_total_shots),
            "shots_per_circuit": int(self.shots_per_circuit),
            "n_circuits": int(self.n_circuits),
            "n_seeds": int(self.n_seeds),
            "notes": self.notes,
        }


def _length_adaptive_total(
    n_res: int,
    config: Dict[str, int],
) -> int:
    if n_res <= 7:
        return config["shots_short"]
    if n_res <= 10:
        return config["shots_medium"]
    if n_res <= 12:
        return config["shots_long"]
    if n_res <= 15:
        return config["shots_xlong"]
    return config["shots_max"]


def allocate_shots(
    n_res: int,
    n_bonds: int,
    *,
    budget_mode: str = "length_adaptive",
    fixed_shots_per_task: Optional[int] = None,
    config_override: Optional[Dict[str, int]] = None,
    bits_per_step: int = 6,
) -> ShotAllocation:
    """Compute shot allocation for one task.

    Parameters
    ----------
    n_res, n_bonds : task length
    budget_mode    : "fixed" | "length_adaptive"
    fixed_shots_per_task : required when budget_mode="fixed"
    config_override      : dict overriding SHOT_BUDGET_DEFAULTS keys
    """
    cfg = dict(SHOT_BUDGET_DEFAULTS)
    if config_override:
        cfg.update(config_override)
    if budget_mode == "fixed":
        if fixed_shots_per_task is None:
            raise ValueError(
                "budget_mode='fixed' requires fixed_shots_per_task"
            )
        target = int(fixed_shots_per_task)
        notes = "fixed by user (min/max bounds bypassed)"
        # In fixed mode, the user's request is the exact target.
        # Min/max from cfg are only applied in length_adaptive mode.
    elif budget_mode == "length_adaptive":
        target = _length_adaptive_total(int(n_res), cfg)
        # bound by min/max only in length-adaptive mode
        target = max(int(cfg["min_shots_per_task"]), target)
        target = min(int(cfg["max_shots_per_task"]), target)
        notes = "length_adaptive"
    else:
        raise ValueError(
            f"unknown budget_mode={budget_mode!r}"
        )
    spc = int(cfg["shots_per_circuit"])
    if spc <= 0:
        raise ValueError(f"shots_per_circuit must be > 0; got {spc}")
    n_circ = int(math.ceil(target / spc))
    allocated = n_circ * spc
    n_qubits = int(n_bonds) * int(bits_per_step)
    return ShotAllocation(
        n_res=int(n_res),
        n_bonds=int(n_bonds),
        n_qubits=int(n_qubits),
        budget_mode=budget_mode,
        requested_total_shots=int(target),
        allocated_total_shots=int(allocated),
        shots_per_circuit=spc,
        n_circuits=int(n_circ),
        n_seeds=int(n_circ),
        notes=notes,
    )


__all__ = ["SHOT_BUDGET_DEFAULTS", "ShotAllocation", "allocate_shots"]
