# Author: Yuqi Zhang
"""ras_folding.kras — full-batch task runner + landscape.

Public API:
  KrasTask, load_kras_tasks                 -- task ingestion from CSV
  KrasFullBatchRunner                       -- end-to-end per-task runner
  LandscapeReconstructor, LandscapeResult   -- PCA / free-energy landscape
  StructureAnalyzer                         -- per-task structural diagnostics
  write_task_report, write_global_report    -- markdown reports

The runner implements the single pipeline flow: env-conditioned moment
matching → hea_with_tf → imaginary-time reject → Hybrid SQD.
"""
from ras_folding.kras.task_loader import KrasTask, load_kras_tasks
from ras_folding.kras.landscape import (
    LandscapeReconstructor,
    LandscapeResult,
)
from ras_folding.kras.structure_analysis import StructureAnalyzer
from ras_folding.kras.report import write_task_report, write_global_report
from ras_folding.kras.full_batch_runner import KrasFullBatchRunner

__all__ = [
    "KrasTask",
    "load_kras_tasks",
    "KrasFullBatchRunner",
    "LandscapeReconstructor",
    "LandscapeResult",
    "StructureAnalyzer",
    "write_task_report",
    "write_global_report",
]
