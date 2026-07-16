# KRAS Quantum Pocket Conformational Landscape & Docking Pipeline

A **self-contained, single-pipeline** build of the KRAS pocket quantum
conformational-landscape benchmark. There is exactly one pipeline: an
environment-conditioned, genuinely-entangled HEA sampler followed by a hybrid
Pauli ⊕ RMSD SQD refinement, then classical post-processing, landscape
reconstruction, and oracle + docking validation.

> Full design + scientific positioning: [`doc/SYSTEM_REPORT.md`](doc/SYSTEM_REPORT.md).
> **Honest statement:** the effective Hamiltonian is stoquastic (classically
> simulable via sign-problem-free QMC). This is an NISQ-era *methodology and
> infrastructure*, not a claim of quantum advantage.

This directory has no dependency on its parent repository — there is **one**
entry (`run_pipeline.py`), one ansatz (`hea_with_tf`), and one SQD
coupling (`hybrid`).

## The seven-stage pipeline

| Stage | What | Implementation |
|------|------|----------------|
| A | env + corridor V2 prior → marginals/correlations | `ras_folding/prior/*`, wired by `KrasFullBatchRunner._build_v2_prior` |
| B | closed-form HEA θ (no VQE) | `ras_folding/quantum/moment_match_initializer.py` |
| C | HEA circuit assembly (Ry·CX·Ry) | `ras_folding/quantum/hea_ansatz.py`, `circuit_builder.py` |
| D | quantum sampling (Aer / IBM Runtime) | `ras_folding/quantum/{aer_backend,ibm_runtime_backend}.py` |
| E | classical imaginary-time rejection | `ras_folding/sampler/imaginary_time_sampler.py` |
| F | hybrid SQD refinement (Pauli ⊕ RMSD) | `ras_folding/refinement/{subspace_diagonalization,hybrid_coupling,pauli_coupling}.py` |
| G | postprocess + landscape + docking | `ras_folding/postprocess/*`, `ras_folding/kras/landscape.py`, `run_oracle_docking_eval.py` |

## Layout

```
run_pipeline.py             # THE entry — ANY system, COMPLETE flow
run_sampling.py             # stage 1 only — sampling (A–G)
run_oracle_docking_eval.py  # downstream: RMSD oracle → PULCHRA → Vina docking
check_external_tools.py     # preflight for pulchra/obabel/vina
external_tools.json         # tool paths (pulchra → vendored; obabel/vina → PATH)
environment.yml             # conda environment
inputs/                     # kras_tasks.csv + analysis groups + sequences
kras_select_systems/        # 9 reference PDB structures
tools/pulchra/              # vendored PULCHRA (source + arm64 binary)
doc/SYSTEM_REPORT.md        # full system report
ras_folding/                # core library (encoder, prior, quantum, sampler,
                            #   scoring, refinement, densify, reconstruct,
                            #   postprocess, kras, utils)
docking_eval/               # Vina docking wrapper + Kd
oracle_eval/                # RMSD oracle candidate selection
pipeline_validation/        # PASS/WARN/FAIL diagnostic report
examples/run_smoke.py       # end-to-end closed-loop smoke (synthetic n=3)
tests/                      # complete test flow (see Testing)
```

## Quickstart — `run_pipeline.py` (any system, complete flow)

`run_pipeline.py` is the recommended entry: ONE command runs the COMPLETE flow
(sampling → reconstruction → oracle → docking → validation) on **any** protein
system. It is not tied to KRAS. Run everything from this directory.

**Single system — just a PDB + a residue range (no CSV needed):**

```bash
python run_pipeline.py \
    --pdb /path/to/structure.pdb --chain A \
    --start-resi 10 --end-resi 20 --ligand LIG \
    --run-name my_run
```

**Batch — a tasks CSV over any structures:**

```bash
python run_pipeline.py \
    --tasks my_tasks.csv --structure-root /path/to/pdbs \
    --run-name my_batch
```

The CSV needs at least `ref_pdb, chain_id, start_resi, end_resi`
(`ligand_resname` optional; the fragment sequence is read from the PDB).
Everything for a run lands self-contained under
`logs/pipeline/<run-name>/<task_id>/` (sampling + oracle + reconstruction +
docking + validation in one place).

Defaults to the **local Aer** simulator. Docking **auto-skips** when Vina /
OpenBabel are absent (the rest of the flow still completes). Useful flags:
`--skip-eval` (sampling only), `--skip-docking`, `--no-validation`,
`--skip-sampling` (reuse an existing run_dir for eval only).

### IBM hardware (explicit opt-in)

Real submission requires **both** `--backend ibm` and `--submit-ibm`
(a saved `QiskitRuntimeService` account is needed). `--backend ibm` alone
stays in dry-run mode.

```bash
python run_pipeline.py --pdb structure.pdb --chain A --start-resi 10 --end-resi 20 \
    --backend ibm --submit-ibm \
    --backend-name ibm_cleveland --execution-mode batch \
    --n-circuits 8 --shots 2048 --run-name my_ibm_run
```

### Running the stages separately (advanced)

`run_pipeline.py` orchestrates two lower-level scripts you can also run directly:

```bash
# stage 1 — sampling only (A–G); defaults to --backend aer
python run_sampling.py --run-name pipeline_aer

# downstream — oracle + PULCHRA reconstruction + Vina docking
python check_external_tools.py --config external_tools.json
python run_oracle_docking_eval.py \
    --run-dir logs/kras_full_batch/pipeline_aer \
    --tasks inputs/kras_tasks.csv --structure-root kras_select_systems \
    --external-tools-config external_tools.json
```

PULCHRA is vendored under `tools/pulchra/`; OpenBabel (`obabel`) and AutoDock
Vina (`vina`) are expected on `PATH` (install separately, e.g. via conda).

## Testing

```bash
python -m tests.run_all_tests          # full suite: PASS/FAIL table
```

This runs, as isolated subprocesses:

1. four module self-tests — `ras_folding.quantum.hea_ansatz`,
   `ras_folding.quantum.moment_match_initializer`,
   `ras_folding.refinement.pauli_coupling`,
   `ras_folding.refinement.hybrid_coupling`;
2. `examples.run_smoke` — end-to-end pipeline on a synthetic n=3 case;
3. `tests.test_blocks_smoke` — 16 per-block minimal tests (encoder, scoring,
   sampler, quantum-on-Aer, imaginary-time, refinement, densify, postprocess,
   landscape, structure analysis, task loader, reconstruct/PULCHRA,
   oracle, docking kd-math, validation);
4. `tests.test_prior_integration` — Stage A env/corridor prior integration
   (uses `inputs/kras_tasks.csv` + `kras_select_systems/6GJ6.pdb`).

Individual entrypoints:

```bash
python -m examples.run_smoke        # expects "SMOKE TEST: PASS"
python -m tests.test_blocks_smoke          # 16 per-block tests (quantum on Aer)
python -m tests.test_prior_integration     # 11 prior tests
```

The docking block needs `vina` on `PATH`; without it, `run_oracle_docking_eval.py`
runs the oracle + PULCHRA reconstruction with `--skip-docking`.

## Requirements

- Python ≥ 3.10, `numpy`, `scipy`, `qiskit`, `qiskit-aer` (see `environment.yml`).
- For docking only: `obabel`, `vina` on `PATH`. PULCHRA is bundled.

```bash
conda env create -f environment.yml
```
