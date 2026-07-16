# KRAS Pocket Conformational Landscape Pipeline — System Reference

This document is the end-to-end system-flow and implementation reference for
the KRAS pocket quantum conformational-landscape and docking pipeline. The
pipeline is a single flow: an environment-conditioned, genuinely-entangled HEA
sampler, a hybrid Pauli ⊕ RMSD SQD refinement, then classical post-processing,
landscape reconstruction, and oracle + docking validation.

---

## 1. Scientific Positioning

### 1.1 Objective

The KRAS pocket pipeline performs **drug-, mutation-, and binding-mode-conditioned
pocket conformational-landscape analysis** over a curated set of KRAS fragments.

Input: `inputs/kras_tasks.csv`, defining fragments that span nine reference PDB
structures and three comparison groups (`drug_effect_G12D`,
`binding_mode_contrast`, `mutation_panel_BI2865`).

Output: a **local conformational-landscape distribution** per fragment —
explicitly, an environment-aware multi-basin weighted distribution rather than a
single optimal conformation.

### 1.2 Quantum Design Positioning

| Property | In this design |
|---|---|
| Genuine entangled quantum state preparation | Yes (Hardware-Efficient Ansatz with bond-aware CX) |
| Mathematically rigorous Pauli matrix-element SQD post-processing | Yes (exact Hamming-1 matrix element) |
| Environment-aware sampling (receptor + ligand geometric constraints) | Yes (via V2 corridor prior moment matching) |
| Provable quantum advantage | No (the underlying $\hat H$ is stoquastic; classically QMC simulable) |
| Asymptotic speedup over classical methods | No |
| Quantum acceleration of the conformational sampling | No (no rigorous advantage exists in the NISQ regime) |

This is an **honest NISQ-era quantum-enabled methodology** providing:
1. A genuinely entangled sampler (beyond product-state biased-Bernoulli);
2. A Pauli-rigorous + RMSD-biological hybrid SQD post-processing layer;
3. Environment information (receptor + ligand coordinates) injected into the
   quantum state via V2 moment matching;
4. A complete `ibm_cleveland` hardware execution path.

The work makes **no claim** of computational-complexity advantage, provable
quantum speedup, or rigorous SQD convergence.

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  ┌──────────────────┐                            │
│   KrasTask  ───→ │  Stage A: V2     │                            │
│  (CSV + PDB)     │  prior sampling  │                            │
│                  │  → marginals     │                            │
│                  │  → correlations  │                            │
│                  └────────┬─────────┘                            │
│                           ↓                                       │
│                  ┌──────────────────┐                            │
│                  │  Stage B: closed-│                            │
│                  │  form θ derivation│                           │
│                  │  → θ_0, θ_1      │                            │
│                  │  (no VQE)        │                            │
│                  └────────┬─────────┘                            │
│                           ↓                                       │
│                  ┌──────────────────┐                            │
│                  │  Stage C: HEA    │                            │
│                  │  circuit assembly│  ← Genuine entanglement     │
│                  │  Ry · CX · Ry    │                            │
│                  └────────┬─────────┘                            │
│                           ↓                                       │
│                  ┌──────────────────┐                            │
│                  │  Stage D: Aer /  │                            │
│                  │  ibm_cleveland   │                            │
│                  │  Z-basis measure │                            │
│                  └────────┬─────────┘                            │
│                           ↓                                       │
│                  ┌──────────────────┐                            │
│                  │  Stage E: class. │                            │
│                  │  imaginary-time  │                            │
│                  │  rejection       │                            │
│                  │  exp(-τ·H_filter)│                            │
│                  └────────┬─────────┘                            │
│                           ↓                                       │
│                  ┌──────────────────┐                            │
│                  │  Stage F: Hybrid │                            │
│                  │  SQD refinement  │  ← Pauli + RMSD             │
│                  │  α·T_Pauli +     │                            │
│                  │  β·T_RMSD        │                            │
│                  └────────┬─────────┘                            │
│                           ↓                                       │
│                  ┌──────────────────┐                            │
│                  │  Stage G: post-  │                            │
│                  │  process + PCA   │  ← classical downstream     │
│                  │  + PULCHRA       │                            │
│                  │  + Vina docking  │                            │
│                  └──────────────────┘                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Stage-by-Stage Description

### 3.1 Stage A — V2 Prior Sampling and Statistical Extraction

**Implementation**: `ras_folding/quantum/moment_match_initializer.py:MomentMatchInitializer._sample_bitstrings`

**Inputs**:
- `encoder_inputs`: the task-specific `anchor_left`, `anchor_right`, `v_left_seed`, `v_right_seed`;
- `env_ctx`: receptor and ligand heavy-atom coordinates (KDTree), produced by `ras_folding.prior.environment`;
- `corridor_ctx`: the C4 ligand-anchor pocket Bézier curve, produced by `ras_folding.prior.corridor`;
- `K_samples`: number of prior samples (default 500).

**Algorithm**:
1. When `env_ctx` and `corridor_ctx` are available, invoke `PriorConditionedBaseSampler.sample()` (V2 full mode).
2. Otherwise, fall back to `EncoderBaseSampler(random_codes)` followed by `decode_and_validate` filtering.

**Outputs**:
- `bitstrings: List[int]` — K environment-consistent valid bitstrings;
- $p_q^{\mathrm{V2}} = \mathrm{mean}(z_q)$ — single-qubit marginals, shape `(n_qubits,)`;
- $C_{q,q'}^{\mathrm{V2}} = \mathrm{mean}(z_q \cdot z_{q'}) - p_q \cdot p_{q'}$ — bit-pair correlations, computed only on CX edges of the HEA topology;
- $z^{*} = \arg\min H_{\mathrm{filter}}(\mathrm{decode}(z))$ over the sampled bitstrings;
- `sampling_mode ∈ {"v2_full", "fallback_random"}`.

**Retention of environmental information**:
- The V2 corridor prior encodes receptor and ligand coordinates into the valid-path sampling probabilities;
- The marginals $p_q^{\mathrm{V2}}$ carry the one-point environmental statistics;
- The bit-pair correlations $C_{q,q'}^{\mathrm{V2}}$ carry the two-point environmental statistics;
- Higher-order joint statistics are not retained (lossy compression).

### 3.2 Stage B — Closed-Form HEA Parameter Derivation

**Implementation**: `ras_folding/quantum/moment_match_initializer.py:MomentMatchInitializer.compute_theta`

**Objective**: Without VQE, derive HEA parameters $\theta$ in closed form such that the prepared state $|\psi(\theta)\rangle$ matches the V2 one-point and two-point statistics.

**Step 1 — First Ry layer (strict closed form)**:
$$
\theta_{0,q} = 2 \arcsin\bigl(\sqrt{p_q^{\mathrm{V2}}}\bigr).
$$
This produces a product state $\bigotimes_q \bigl(\sqrt{1 - p_q^{\mathrm{V2}}}\,|0\rangle + \sqrt{p_q^{\mathrm{V2}}}\,|1\rangle\bigr)$ that exactly matches the V2 marginals.

**Step 2 — CX brick-wall layer (fixed topology, no parameters)**:
- Bond-internal: within each six-qubit bond block, a brick-wall pattern (even pairs followed by odd pairs);
- Bond-bridge: a CX between the last qubit of bond $k$ and the first qubit of bond $k+1$, for all $k = 0, \dots, n_{\mathrm{bonds}} - 2$;
- See `hea_ansatz.cx_edges_bond_aware` for the exact ordering.

After this layer, the product state becomes entangled: the marginals of control qubits remain fixed, whereas the marginals of target qubits shift.

**Step 3 — Second Ry layer (per-edge numerical solve)**:

For each CX edge $(a, b)$, solve in the two-qubit subsystem for $(\theta_{1,a}, \theta_{1,b})$ minimising
$$
\mathcal{L}(\theta_{1,a}, \theta_{1,b}) = (p_a^{\mathrm{out}} - p_a^{\mathrm{V2}})^2 + (p_b^{\mathrm{out}} - p_b^{\mathrm{V2}})^2 + 4(C_{a,b}^{\mathrm{out}} - C_{a,b}^{\mathrm{V2}})^2.
$$
The solver uses `scipy.optimize.minimize` (L-BFGS-B, at most fifty iterations, approximately one millisecond per edge).

For qubits shared across multiple CX edges, the per-edge solutions are **averaged in a single coordinate-descent pass**.

**Stage B output**:
- `theta = concat([theta_0, theta_1])`, shape `(2 · n_qubits,)` for $R = 1$;
- All derivations are closed-form or single-pass numerical with **no iterative optimisation on the quantum device**;
- Wall time per task is approximately 10–100 ms (dominated by V2 sampling).

### 3.3 Stage C — HEA Circuit Assembly

**Implementation**:
- Operator: `ras_folding/quantum/hea_ansatz.py:build_hea_circuit`;
- Integration: the `"hea_with_tf"` branch in `ras_folding/quantum/circuit_builder.py`.

**Circuit structure** (for $R = 1$):
$$
|0\rangle^{\otimes n} \;\rightarrow\; R_y(\theta_0) \;\rightarrow\; \mathrm{CX}_{\mathrm{brick}} \;\rightarrow\; R_y(\theta_1) \;\rightarrow\; \mathrm{measure}\,(Z).
$$

**Gate count** (for $n_{\mathrm{bonds}} = 9$, $n_{\mathrm{qubits}} = 54$):
- Single-qubit Ry gates: $2 \times 54 = 108$;
- Two-qubit CX gates (brick-wall + bond-bridge):
  - Five CX per bond × nine bonds = 45;
  - Bond-bridge: 8;
  - **Total: 53 CX per layer**;
- Total CX for $R = 1$: **53** (comfortably within the two-qubit-noise budget of `ibm_cleveland`).

**Rationale for this topology**:
- Bond-internal CX entangles the six qubits encoding a single bond's lattice direction;
- Bond-bridge CX entangles consecutive bonds — mirroring the autoregressive geometric structure of the encoder in the quantum register;
- The topology is compatible with the heavy-hex coupling of `ibm_cleveland`, with post-transpile depth at most twenty layers.

### 3.4 Stage D — Quantum Sampling

**Implementation**: `ras_folding/quantum/aer_backend.py` and `ras_folding/quantum/ibm_runtime_backend.py`.

**Operation**: Z-basis measurement of $|\psi(\theta^{*})\rangle$.

**Configuration**:
- `n_circuits = 4` (default; configurable);
- `shots_per_circuit = 2048` (default; configurable);
- Total shots per task: $4 \times 2048 \times 3 \,\text{taus} = 24{,}576$.

**Output**: `List[CandidateSample]` via `counts_to_candidate_samples`.

### 3.5 Stage E — Classical Imaginary-Time Rejection

**Implementation**: `ras_folding/sampler/imaginary_time_sampler.py:QuantumImaginaryTimeSampler`.

**Operation**: For each sampled bitstring $z$,
$$
A(z) = \exp\bigl(-\tau_c \, H_{\mathrm{filter}}(\mathrm{decode}(z))\bigr).
$$
Draw $u \sim U(0, 1)$ and accept iff $u \le A(z)$.

The default $\tau_c$ values are `(0.0, 0.1, 0.2)` (three temperatures stratified).

**Why classical rejection is part of the flow**:
- V2 moment matching matches only the one-point and two-point statistics; it does **not** strictly minimise $\langle\hat H\rangle$;
- The quantum state does not enforce anchor-endpoint matching strictly;
- Classical rejection supplies the anchor and clash penalties of $H_{\mathrm{filter}}$;
- This is the canonical structure of the "quantum-prior + classical imaginary-time-filter" architecture.

### 3.6 Stage F — Hybrid SQD Refinement

**Implementation**: `ras_folding/refinement/subspace_diagonalization.py:SubspaceDiagonalizationRefiner` with `coupling_mode="hybrid"`.

**Subspace selection** (40/30/30 heuristic, default):
1. 40 % lowest `full_energy`;
2. 30 % highest `count` (quantum sampling frequency);
3. 30 % RMSD diversity (farthest-point sampling);
4. Dense-fraction cap: perturbed descendants are limited to at most 60 % of the subspace.

**Effective Hamiltonian**:
$$
H_{\mathrm{eff}}[i, j] = \tilde E_i \, \delta_{ij} + \alpha_{\mathrm{pauli}} \, T_{\mathrm{Pauli}}[i, j] + \alpha_{\mathrm{rmsd}} \, T_{\mathrm{RMSD}}[i, j],
$$
where:
- $\tilde E_i$ is the MAD-normalised $H_{\mathrm{filter}}(z_i)$;
- $T_{\mathrm{Pauli}}[i, j] = -g \cdot \mathbb{1}[\mathrm{popcount}(z_i \oplus z_j) = 1]$, the **exact Pauli matrix element** of $\hat H = H_{\mathrm{filter}} - g \sum_q X_q$ restricted to the sampled subspace;
- $T_{\mathrm{RMSD}}[i, j] = -\kappa \exp(-\mathrm{RMSD}_{ij}^2 / 2\sigma^2)$ on the KNN graph (the RMSD-Gaussian kernel `rmsd_kernel_coupling`);
- Default values: $\alpha_{\mathrm{pauli}} = \alpha_{\mathrm{rmsd}} = 0.5$, $g = 0.03$.

**Operator properties**:
- The Pauli component is **mathematically rigorous**: it is the exact projection of $\hat H$ onto the sampled subspace;
- The RMSD component encodes **biological precision**: structural similarity reflecting basin geometry;
- The hybrid form provides simultaneous mathematical rigour and biological grounding.

**Diagonalisation**: `scipy.linalg.eigh` (dense, for $N \le 300$) or `scipy.sparse.linalg.eigsh` (sparse, the lowest `n_modes` eigenpairs).

**Boltzmann mode reweighting**:
$$
b_k = \exp\bigl(-\beta(\lambda_k - \lambda_0)\bigr), \quad w_i = \sum_k b_k \, |v_k[i]|^2, \quad \hat w_i = \frac{w_i}{\sum_j w_j},
$$
$$
\mathrm{refined\_score}_i = \tilde E_i - \alpha_{\mathrm{score}} \log(\hat w_i + \epsilon).
$$

**Output**: `RefinementResult` containing `candidates: List[RefinedCandidate]`, sorted by `refined_score` ascending.

### 3.7 Stage G — Post-Processing, Landscape, and Downstream

The classical downstream comprises:
1. `PredictionPostProcessor`: bitstring deduplication, structural deduplication (RMSD threshold 0.5 Å), basin clustering (RMSD threshold 1.5 Å), and top-K PDB export;
2. `LandscapeReconstructor`: **PCA on the weighted refined coordinates**, producing the free-energy surface;
3. `StructureAnalyzer`: RMSD metrics relative to `reference_coords`;
4. `run_oracle_docking_eval.py`: `RMSDOracleSelector` → PULCHRA all-atom reconstruction → AutoDock Vina docking;
5. `pipeline_validation`: PASS/WARN/FAIL diagnostic report.

**Role of PCA**:
- PCA operates on decoded CA coordinates (geometric space), independent of the quantum-side design;
- broad basin coverage produces a richer variance signal for PCA and yields more interpretable landscape plots;
- The Hybrid SQD operates on bitstrings (`z`-space), whereas PCA operates on 3D coordinates — the two are complementary rather than conflicting.

---

## 4. Self-Consistency Analysis

### 4.1 Shared $\hat H$ Across Stages

| Stage | Use of $\hat H$ |
|---|---|
| Stage B (parameter derivation) | Does **not** strictly minimise $\langle\hat H\rangle$; matches V2 one-point and two-point statistics |
| Stage E (classical rejection) | Uses only the diagonal $H_{\mathrm{filter}}$ |
| Stage F (SQD) | Uses the **exact** Pauli matrix elements of $\hat H$ on the sampled subspace |

**Design property**: Stage B is not strict ground-state preparation of $\hat H$. Consequently, the Pauli SQD in Stage F does **not** strictly converge to the projection of the true ground state of $\hat H$ onto the sampled subspace. It instead converges to the low-energy modes of $\hat H$ restricted to the V2-statistics-matched subspace. This is a deliberate trade-off: biologically meaningful, mathematically not strictly rigorous in the SQD-convergence sense.

### 4.2 Environmental Information Flow

```
Receptor + ligand heavy-atom coordinates
       ↓
   env_ctx + corridor_ctx (V2 prior)
       ↓
   V2 valid-path sampling (K = 500)
       ↓
   bit marginals  p_q^V2   +   correlations  C_{q,q'}^V2
       ↓
   HEA θ via moment matching
       ↓
   |ψ(θ*)⟩ — entangled quantum state retaining environment info
       ↓
   measurement → bitstrings (environment-aware distribution)
       ↓
   classical rejection (H_filter; intra-fragment + anchor)
       ↓
   Hybrid SQD (Pauli + RMSD)
       ↓
   Classical downstream
```

**Retention level**: one-point and two-point statistics are fully retained; higher-order joint statistics are not (acceptable lossy compression for the KRAS pocket use case).

### 4.3 Stoquasticity Statement

The Hamiltonian $\hat H = H_{\mathrm{filter}}(z) - g \sum_q X_q$ with $g > 0$, expressed in the computational basis, has:
- Diagonal entries $H_{\mathrm{filter}}(z)$, arbitrary real values;
- Off-diagonal entries $-g$ for Hamming-1 neighbours and $0$ otherwise.

Since all off-diagonal entries are non-positive, the Hamiltonian is **strictly stoquastic**.

By the Bravyi–Terhal (2008) theorem, the Stoquastic Local Hamiltonian Problem is in $\mathrm{MA}$ (equivalently, $\mathrm{StoqMA}$). The corresponding distributions can therefore be simulated classically via sign-problem-free Quantum Monte Carlo.

**Disclosure statement for publication**:
> "The effective Hamiltonian $\hat H$ used in our SQD refinement is stoquastic; classical Quantum Monte Carlo can in principle simulate the same target distribution. This work demonstrates a methodology and an infrastructure rather than asymptotic quantum speedup."

---

## 5. Configuration

The pipeline is a single flow with exactly one ansatz (`hea_with_tf`) and one SQD
coupling (`hybrid`). There is no `sampler_mode` switch and no front-end mode
selection — the quantum state is always prepared by environment-conditioned
moment matching, and the SQD always uses the hybrid Pauli ⊕ RMSD operator.

### 5.1 Default Configuration

```python
from ras_folding.kras.full_batch_runner import (
    KrasFullBatchRunner, _RunnerDefaults,
)

runner = KrasFullBatchRunner(backend_config=backend_config)
# Equivalent to the explicit defaults:
runner = KrasFullBatchRunner(
    backend_config=backend_config,
    defaults=_RunnerDefaults(
        moment_match_K=500,            # V2 prior samples for statistics
        moment_match_reps=1,           # HEA layers (R = 1)
        refiner_coupling_mode="hybrid",
        refiner_g_quantum=0.03,        # transverse-field strength g
        refiner_alpha_pauli=0.5,       # Pauli coupling weight
        refiner_alpha_rmsd=0.5,        # RMSD coupling weight
        # ... see _RunnerDefaults for the full set of fields
    ),
)
```

### 5.2 Quantum-Side Summary

| Component | Value | Property |
|---|---|---|
| Ansatz | `hea_with_tf` | HEA with V2-derived $\theta$; genuine entanglement |
| SQD coupling | `hybrid` | $\alpha\,T_{\mathrm{Pauli}} + \beta\,T_{\mathrm{RMSD}}$; partially rigorous + biological |

---

## 6. Execution

### 6.1 Foundation Self-Tests and Smoke Test

```bash
# Foundation module self-tests
python -m ras_folding.quantum.hea_ansatz
python -m ras_folding.refinement.pauli_coupling
python -m ras_folding.refinement.hybrid_coupling
python -m ras_folding.quantum.moment_match_initializer

# End-to-end closed-loop smoke
python -m examples.run_smoke
```

Expected terminal output: `SMOKE TEST: PASS`.

### 6.2 Running the Pipeline

`run_pipeline.py` is the single entry for the complete flow on any system:

```bash
# Local Aer (default)
python run_pipeline.py \
    --pdb structure.pdb --chain A --start-resi 10 --end-resi 20 \
    --run-name my_run

# IBM Runtime (opt-in: requires a saved QiskitRuntimeService account)
python run_pipeline.py \
    --pdb structure.pdb --chain A --start-resi 10 --end-resi 20 \
    --backend ibm --submit-ibm \
    --backend-name ibm_cleveland --execution-mode batch \
    --run-name my_ibm_run
```

For the KRAS benchmark, pass a tasks CSV over the reference structures:

```bash
python run_pipeline.py \
    --tasks inputs/kras_tasks.csv --structure-root kras_select_systems \
    --run-name kras_batch
```

Real IBM submission requires **both** `--backend ibm` and `--submit-ibm`;
`--backend ibm` alone stays in dry-run mode.

---

## 7. Summary

The pipeline is a single flow defined by the Hamiltonian
$\hat H = H_{\mathrm{filter}} - g \sum_q X_q$, which threads through the
front-end state preparation, the classical imaginary-time rejection, and the
back-end SQD refinement. Its defining properties are:

- Genuine entangled quantum state preparation without VQE training, obtained via
  closed-form moment matching of HEA parameters against V2 prior statistics;
- Environmental information conveyed through the V2 corridor and environment
  contexts, fully retained at the one-point and two-point statistical levels;
- A hybrid SQD post-processing layer combining mathematically rigorous Pauli
  matrix elements with biologically meaningful RMSD-Gaussian coupling;
- A single configuration — `hea_with_tf` ansatz and `hybrid` SQD coupling — with
  no mode switches;
- No claim of quantum advantage: the underlying Hamiltonian is stoquastic and
  therefore classically simulable via sign-problem-free Quantum Monte Carlo. The
  contribution is methodological and infrastructural, not
  algorithmic-complexity-theoretic.
