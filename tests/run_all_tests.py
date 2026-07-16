# Author: Yuqi Zhang
"""full_pipline complete test runner.

Runs the entire test flow and reports a single PASS/FAIL table.
Each target runs as an isolated subprocess (so a ``sys.exit`` / assertion
in one cannot abort the suite, and each runs under its own ``__main__``).
The aggregate exit code is 0 only if every target passes.

Targets, in order:
  1. ras_folding.quantum.hea_ansatz              (module self-test)
  2. ras_folding.quantum.moment_match_initializer(module self-test)
  3. ras_folding.refinement.pauli_coupling       (module self-test)
  4. ras_folding.refinement.hybrid_coupling      (module self-test)
  5. examples.run_smoke                    (end-to-end Aer)
  6. tests.test_blocks_smoke                       (16 per-block minimal tests)
  7. tests.test_prior_integration                (Stage A prior integration)

Run (from the full_pipline/ project root, or anywhere):
    python -m tests.run_all_tests
    python tests/run_all_tests.py
Pass ``--verbose`` to stream each target's stdout/stderr.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# (label, python -m target)
_TARGETS: List[Tuple[str, str]] = [
    ("hea_ansatz self-test", "ras_folding.quantum.hea_ansatz"),
    ("moment_match self-test", "ras_folding.quantum.moment_match_initializer"),
    ("pauli_coupling self-test", "ras_folding.refinement.pauli_coupling"),
    ("hybrid_coupling self-test", "ras_folding.refinement.hybrid_coupling"),
    ("pipeline end-to-end smoke", "examples.run_smoke"),
    ("per-block minimal smoke", "tests.test_blocks_smoke"),
    ("prior Stage-A integration", "tests.test_prior_integration"),
]


def _run_one(module: str, verbose: bool) -> Tuple[int, str]:
    """Run ``python -m <module>`` in a subprocess rooted at PROJECT_ROOT.

    Returns (returncode, combined_output).
    """
    proc = subprocess.run(
        [sys.executable, "-m", module],
        cwd=str(PROJECT_ROOT),
        capture_output=not verbose,
        text=True,
    )
    out = ""
    if not verbose:
        out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, out


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    verbose = "--verbose" in argv or "-v" in argv

    print("=" * 72)
    print("full_pipline — complete test suite")
    print(f"project root: {PROJECT_ROOT}")
    print("=" * 72)

    results: List[Tuple[str, bool, str]] = []
    for label, module in _TARGETS:
        print(f"\n>>> {label}  (python -m {module})")
        sys.stdout.flush()
        rc, out = _run_one(module, verbose)
        ok = (rc == 0)
        if not verbose:
            # Echo a short tail so failures are diagnosable inline.
            tail = "\n".join(out.strip().splitlines()[-8:])
            if tail:
                print(tail)
            if not ok and ("qiskit_aer" in out or "qiskit.aer" in out):
                print(
                    "    HINT: this target needs qiskit-aer. Install it with "
                    "`pip install qiskit-aer` (see environment.yml)."
                )
        print(f"    -> {'PASS' if ok else f'FAIL (exit {rc})'}")
        results.append((label, ok, module))

    n_pass = sum(1 for _, ok, _ in results if ok)
    n_fail = len(results) - n_pass

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for label, ok, module in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label:<32} ({module})")
    print("-" * 72)
    print(f"  {n_pass} passed, {n_fail} failed, {len(results)} total")
    print("=" * 72)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
