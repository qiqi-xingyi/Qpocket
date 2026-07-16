# Author: Yuqi Zhang
"""Validation dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


_VALID_STATUS = ("PASS", "WARN", "FAIL", "SKIP")


@dataclass
class CheckResult:
    name: str
    status: str          # PASS | WARN | FAIL | SKIP
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in _VALID_STATUS:
            raise ValueError(
                f"status must be one of {_VALID_STATUS}; got {self.status!r}"
            )


@dataclass
class ValidationResult:
    task_id: str
    checks: List[CheckResult]
    summary: Dict[str, Any]
    passed: bool
    warnings: int
    failures: int

    @classmethod
    def from_checks(
        cls,
        task_id: str,
        checks: List[CheckResult],
        summary: Dict[str, Any],
    ) -> "ValidationResult":
        warns = sum(1 for c in checks if c.status == "WARN")
        fails = sum(1 for c in checks if c.status == "FAIL")
        return cls(
            task_id=task_id,
            checks=list(checks),
            summary=dict(summary),
            passed=(fails == 0),
            warnings=warns,
            failures=fails,
        )


__all__ = ["CheckResult", "ValidationResult"]
