# Revision experiments

Engineering code for the comparison arms the revision adds. The method
itself is not reimplemented here: every arm reuses `ras_folding`
unchanged, so the downstream is shared by construction rather than by
convention.

## Why the arms are comparable

All arms implement one contract:

```python
sample(context_or_inputs, n_samples=None, seed=None) -> list[CandidateSample]
```

returning candidates with `coords=None`, `valid=False`, `accepted=False`.
This is the same contract as `QuantumBackendBaseSampler`, so an arm is
injected directly as the `base_sampler` of `QuantumImaginaryTimeSampler`.
Acceptance, densification, subspace refinement, ranking, and evaluation
are then literally the same code path for every arm.

Coordinates are never forwarded from an arm even when it computes them
internally. Each arm returns bitstrings only, and the shared downstream
runs its own `decode_and_validate`. Passing coordinates through would
give one arm a different decoder path.

## Arms

| arm | sampler | what it isolates |
|---|---|---|
| Q | `ras_folding.sampler.quantum_base_sampler` | hardware-executed circuit sampling |
| P | `arms/prior_arm.py` | the prior alone, drawn exactly |
| M | `arms/mcmc_arm.py` | local Markov-chain exploration of the same space |

Arm P is the control for the central methodological claim. The quantum
arm does not sample the prior directly: `moment_match_initializer`
compresses the prior into HEA parameters, keeping only the 1-point bit
marginals and the 2-point correlations on the ansatz CX edges, and
discarding every higher-order and non-edge dependency of the
autoregressive joint. The circuit is therefore a lossy second-order
re-encoding of the same prior, and arm P draws from that prior exactly.

Arm M targets that same prior, so any difference from arm P is a property
of the sampling mechanism rather than of a different objective. Its kernel
mixes a symmetric single-site move with a suffix Gibbs move; independence
MH is deliberately not used, because a prior proposal against a prior
target accepts with probability one and reduces exactly to arm P.

Both arms refuse to run without `env_ctx` and `corridor_ctx`. The
env-blind fallback prior is a different distribution, and silently
falling back to it would compare the quantum arm against something other
than its own prior.

## Prior density

`arms/prior_density.py` evaluates the log-density of a path by replaying
the rollout under a fixed code sequence. Every step mirrors the prior
sampler exactly — the same lattice, masks, policy call, normalisation, and
endpoint tolerance — because a divergence would make the Metropolis
acceptance ratio target a distribution other than the one arm P samples.

## Budget

The arms are not comparable by raw count alone: every quantum shot yields
a bitstring, while a prior rollout that hits an infeasible step or misses
the endpoint tolerance yields none. Both matched definitions are
supported and both counts are always recorded.

- `proposal` — a fixed number of rollout attempts, matching the quantum
  arm on proposals offered to the downstream.
- `valid` — draw until a target number of valid paths is reached,
  matching the quantum arm on candidates delivered to the downstream.

A `valid` run that exhausts its attempt cap records the shortfall in the
run record and emits a warning. It is never silently truncated.

## Run records

`runrecord/` captures provenance at execution time: repository commit and
working-tree cleanliness, interpreter and package versions, host, SLURM
context, and wall/CPU time. A value that cannot be observed is recorded
as null with a reason rather than reconstructed, so the SI can state the
exact boundary of what is known.

## Layout

```
arms/       comparison-arm samplers
runrecord/  provenance capture
configs/    frozen cell manifests and configuration
osc/        Cardinal cluster setup, sync, and submission
results/    run outputs (git-ignored)
run_arm.py  driver — one invocation is one (arm, task, repeat) cell
            (Q is not generated here: it is hardware-executed and read
            from its own run records)
```

## Running on OSC

```bash
bash revision/osc/sync.sh                      # local -> cardinal
ssh cardinal
cd ~/Project/Qpocket
bash revision/osc/setup_env.sh                 # one time
python revision/osc/make_cells.py --arms P M --repeats 3
sbatch --array=1-216 revision/osc/submit_g1_pilot.sh
```

Then pull results back with `bash revision/osc/pull_results.sh`.

Code and results both live under the home project folder
(`~/Project/Qpocket`, results in `revision/results/`). The conda environment is
the one exception and is created on scratch: solving it writes tens of
thousands of small files, which is a file-count concern rather than a
space one.

Cells are frozen in a manifest before submission, so the mapping from
array index to experimental cell stays auditable after the run. Seeds are
derived deterministically from `(master seed, arm, task, repeat)`: cells
are independent, reproducible, and never collide across arms.
