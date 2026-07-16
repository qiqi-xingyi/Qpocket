# Author: Yuqi Zhang
"""pipeline_validation — read finished real-backend validation artifacts
and emit PASS / WARN / FAIL diagnostics + tuning advice.

Public API:
  CheckResult, ValidationResult           -- dataclasses
  FinalPipelineValidator                  -- reads artifacts, runs checks
  write_validation_summary, write_validation_report
                                          -- persist results to disk
"""
from pipeline_validation.types import CheckResult, ValidationResult
from pipeline_validation.checks import FinalPipelineValidator
from pipeline_validation.report import (
    write_validation_summary,
    write_validation_report,
)

__all__ = [
    "CheckResult",
    "ValidationResult",
    "FinalPipelineValidator",
    "write_validation_summary",
    "write_validation_report",
]
