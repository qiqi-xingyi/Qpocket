# KRAS Pocket Perturbation Benchmark: System Selection Logic and Comparison Objectives

## 1. Research Objective

The goal of this benchmark is to evaluate local pocket-region structure prediction and conformational perturbation analysis in KRAS. The benchmark is designed not only to measure fragment-level structural accuracy, but also to examine how ligand identity, binding mode, and mutation background affect KRAS pocket conformational landscapes.

Specifically, this benchmark is designed to answer three questions:

1. Under the same KRAS G12D background, do different Asp12-recognizing SII-P binders induce different pocket conformational landscapes?
2. When the ligand binding mode changes from an Asp12-recognizing SII-P pocket binder to a Switch-I/II groove or dimer-interface binder, does the local KRAS pocket response change?
3. Under the same BI-2865-bound condition, do WT, G12C, G12D, and G12V KRAS states produce different pocket ensembles?

Therefore, this benchmark is defined as:

```text
A KRAS pocket-region fragment benchmark for drug-conditioned, mutation-conditioned, and binding-mode-conditioned landscape analysis.
```

The focus is local pocket-region conformational landscape analysis, not full-length KRAS structure modeling.

---

## 2. Overall Benchmark Design

The benchmark contains three complementary groups of KRAS systems. Each group corresponds to a specific biological or biophysical comparison.

| Group | PDB Systems | Main Variable | Main Objective |
|---|---|---|---|
| Asp12-recognizing SII-P binder series | 7RT4, 7RT1, 7RPZ, 8AVZ | Different ligands under KRAS G12D | Evaluate drug-conditioned changes in the SII-P pocket landscape |
| Switch-I/II groove or dimer-interface binder series | 6GJ6, 6GJ8 | Different binding mode under KRAS G12D | Evaluate binding-mode-conditioned pocket response |
| BI-2865 mutation panel | 8AZV, 8AZX, 8AZY, 8AZZ | Different KRAS mutations under the same ligand | Evaluate mutation-conditioned pocket ensemble redistribution |

The three analysis roles are:

```text
drug_potency_series
binding_mode_contrast
mutation_panel
```

This design allows the benchmark to evaluate three different sources of pocket perturbation:

```text
ligand identity / potency
binding mode
mutation background
```

---

## 3. Group 1: Asp12-Recognizing Noncovalent SII-P Binder Series

### 3.1 System Composition

This group contains:

| PDB ID | Small Molecule | Mutation Background | Binding Mode |
|---|---|---|---|
| 7RT4 | Compound 5B | KRAS G12D | Asp12-recognizing noncovalent SII-P binder |
| 7RT1 | Compound 15 | KRAS G12D | Asp12-recognizing noncovalent SII-P binder |
| 7RPZ | MRTX1133 | KRAS G12D | Asp12-recognizing noncovalent SII-P binder |
| 8AVZ | BI-2865 | KRAS G12D | Asp12-recognizing noncovalent SII-P binder |

All systems in this group share the KRAS G12D mutation background and a broadly similar Asp12-recognizing SII-P binding mode. This makes the group suitable for comparing how different ligands reshape the local pocket landscape under a controlled mutation context.

### 3.2 Comparison Objective

This group is used for:

```text
Drug-conditioned pocket landscape analysis under the KRAS G12D background.
```

The specific objectives are:

1. Compare local prediction accuracy across P-loop, Switch-II, and alpha3/allosteric regions.
2. Evaluate whether different ligands produce different low-energy basin distributions.
3. Examine whether ligand potency is associated with native-like basin occupancy, energy funnel quality, or conformational compactness.
4. Test whether the sampling framework captures ligand-dependent structural constraints within the SII-P pocket.

### 3.3 Recommended Metrics

| Metric | Purpose |
|---|---|
| Fragment-level RMSD | Evaluate local structural accuracy |
| Switch-II module RMSD | Evaluate the core SII pocket region |
| Basin count | Measure landscape diversity |
| Native-like occupancy | Measure sampling enrichment near the experimental structure |
| Funnel score | Measure whether the landscape forms a native-directed energy trend |
| Ligand-potency association | Explore whether Kd is associated with landscape features |

Because this group has a limited number of systems, Kd-related analysis should be presented as an association or trend rather than a strict causal conclusion.

---

## 4. Group 2: Switch-I/II Groove or Dimer-Interface Binder Series

### 4.1 System Composition

This group contains:

| PDB ID | Small Molecule | Mutation Background | Binding Mode |
|---|---|---|---|
| 6GJ6 | Compound 18 | KRAS G12D | Switch-I/II groove or dimer-interface binder |
| 6GJ8 | BI-2852 | KRAS G12D | Switch-I/II groove or dimer-interface binder |

These systems also use the KRAS G12D background, but their binding mode differs from the Asp12-recognizing SII-P binders. They are included to provide a binding-mode contrast.

### 4.2 Comparison Objective

This group is used for:

```text
Binding-mode-conditioned pocket response analysis.
```

The purpose of this group is not to provide a fully paired statistical comparison with Group 1. Instead, it serves as a mechanistic contrast to test whether a different binding mode changes the local conformational response of KRAS pocket regions.

The specific objectives are:

1. Compare the pocket landscapes of 6GJ6 and 6GJ8 under the same G12D background.
2. Analyze the response of Switch-I edge, P-loop edge, and Switch-II regions under groove or dimer-interface binding.
3. Compare the landscape topology and conformational spread with those observed in the Asp12-recognizing SII-P binder series.
4. Examine whether stronger and weaker binders show different degrees of local pocket stabilization.

### 4.3 Special Consideration

The fragment selection in this group is intentionally different from the other groups because the binding mode is different.

This group includes:

```text
Frag-D: residues 3-9
Frag-E: residues 36-42
```

These fragments cover the N-terminal/P-loop edge and Switch-I edge regions, which are relevant to the Switch-I/II groove or dimer-interface binding mode.

---

## 5. Group 3: BI-2865 Mutation Panel

### 5.1 System Composition

This group contains:

| PDB ID | Small Molecule | Mutation Background | Binding Mode |
|---|---|---|---|
| 8AZV | BI-2865 | KRAS WT | BI-2865-bound KRAS WT |
| 8AZX | BI-2865 | KRAS G12C | BI-2865-bound KRAS G12C |
| 8AZY | BI-2865 | KRAS G12D | BI-2865-bound KRAS G12D |
| 8AZZ | BI-2865 | KRAS G12V | BI-2865-bound KRAS G12V |

This is the cleanest mutation-controlled comparison in the benchmark. The ligand is fixed as BI-2865, while the mutation background varies across WT, G12C, G12D, and G12V.

### 5.2 Comparison Objective

This group is used for:

```text
Mutation-conditioned pocket landscape analysis.
```

The specific objectives are:

1. Compare how WT, G12C, G12D, and G12V affect the P-loop and mutation-site region.
2. Analyze whether different mutations redistribute the Switch-II pocket basin population.
3. Examine whether the alpha3/allosteric region shows distal conformational responses.
4. Compare native-like basin occupancy, funnel score, and conformational spread across mutation states.
5. Test whether mutation-specific pocket ensembles are associated with ligand affinity or ligand-compatible conformer enrichment.

### 5.3 Scientific Significance

This group supports the central mutation-related claim of the benchmark:

```text
Under the same BI-2865-bound condition, different KRAS mutations reshape the local pocket conformational ensemble and redistribute accessible low-energy basins.
```

This allows the benchmark to evaluate whether the model captures mutation-conditioned conformational redistribution, rather than only predicting a single local structure.

---

## 6. Fragment Selection Logic

This benchmark uses pocket-region fragments rather than full-length KRAS prediction. The reasons are:

1. KRAS ligand binding and mutation effects are concentrated in local pocket regions.
2. Short fragments are more suitable for the current quantum sampling framework because they reduce encoding size and sampling complexity.
3. Fragment-level predictions can be aggregated into biologically meaningful pocket modules.
4. Local landscapes are easier to interpret in terms of ligand, mutation, and binding-mode effects.

---

## 7. Fragment Definitions and Biological Meaning

| Fragment | Residue Range | Pocket Module | Biological Interpretation |
|---|---:|---|---|
| Frag-A | 9-18 | P-loop / mutation-site module | Covers the G12 mutation site and nearby P-loop region |
| Frag-B | 54-68 | Switch-II N-terminal module | Covers the N-terminal half of Switch II and the core SII pocket region |
| Frag-F | 69-78 | Switch-II C-terminal module | Covers the C-terminal half of Switch II and complements Frag-B |
| Frag-C | 92-103 | Alpha3 / allosteric module | Captures possible distal or allosteric pocket responses |
| Frag-D | 3-9 | N-terminal / P-loop edge module | Used in the groove/dimer-interface binder group to cover the P-loop edge |
| Frag-E | 36-42 | Switch-I edge module | Used in the groove/dimer-interface binder group to cover Switch-I/II groove effects |

---

## 8. Module-Level Aggregation

The analysis should be performed at two levels:

```text
fragment-level analysis
module-level analysis
```

### 8.1 Fragment-Level Analysis

Fragment-level analysis directly evaluates each short segment:

```text
Frag-A
Frag-B
Frag-C
Frag-D
Frag-E
Frag-F
```

This level is suitable for reporting:

1. Fragment RMSD.
2. Fragment energy distribution.
3. Fragment basin count.
4. Fragment native-like occupancy.
5. Fragment landscape shape.

### 8.2 Module-Level Analysis

Module-level analysis combines adjacent or functionally related fragments into biologically meaningful regions.

| Module | Fragments | Residue Range | Meaning |
|---|---|---:|---|
| P-loop / mutation-site module | Frag-D + Frag-A | 3-18 | Covers the P-loop edge and G12 mutation site |
| Switch-I module | Frag-E | 36-42 | Covers the Switch-I edge |
| Switch-II module | Frag-B + Frag-F | 54-78 | Covers the core SII pocket |
| Alpha3 / allosteric module | Frag-C | 92-103 | Covers the alpha3/allosteric response region |

The most important module is:

```text
Switch-II module = Frag-B + Frag-F = residues 54-78
```

Drug-induced pocket responses may involve both halves of Switch II. Therefore, Frag-B and Frag-F should be reported individually for fragment-level accuracy, but interpreted together for SII pocket-level biological analysis.

---

## 9. Rationale for the Current Case Setup

The current case setup is appropriate for the intended KRAS pocket benchmark for the following reasons.

### 9.1 Coverage of Key KRAS Drug-Binding Regions

The selected fragments cover:

```text
P-loop / mutation site
Switch-I edge
Switch-II pocket
Alpha3 / allosteric region
```

These regions are central to KRAS small-molecule binding, mutation effects, and local pocket remodeling.

### 9.2 Coverage of Multiple Perturbation Sources

The benchmark includes:

```text
drug identity / potency perturbation
binding mode perturbation
mutation perturbation
```

Therefore, the benchmark can evaluate not only local structure prediction, but also how different biochemical perturbations reshape the pocket conformational landscape.

### 9.3 Clean Mutation-Controlled Panel

The 8AZV, 8AZX, 8AZY, and 8AZZ systems fix the ligand as BI-2865 and vary the mutation background. This makes the group suitable for mutation-conditioned comparison.

### 9.4 Drug Series Under the Same Mutation Background

The 7RT4, 7RT1, 7RPZ, and 8AVZ systems share the G12D background and belong to the Asp12-recognizing SII-P binder class. This makes the group suitable for drug-conditioned landscape comparison.

### 9.5 Binding-Mode Contrast

The 6GJ6 and 6GJ8 systems represent Switch-I/II groove or dimer-interface binders. They provide a complementary contrast to the SII-P binder systems.

---

## 10. Comparisons That Should Be Avoided

Although the benchmark design is reasonable, several over-interpretations should be avoided.

### 10.1 Do Not Mix All PDBs Into One Global Ranking

Different systems vary in mutation background, ligand identity, and binding mode. A single global ranking may obscure the source of the observed differences.

The analysis should be performed group by group:

```text
Group 1: Asp12-recognizing SII-P binder series
Group 2: Switch-I/II groove or dimer-interface binder series
Group 3: BI-2865 mutation panel
```

### 10.2 Do Not Treat the Groove/Dimer-Interface Group as a Fully Paired Comparison

The 6GJ6 and 6GJ8 systems have different binding modes and a partially different fragment set. This group is best interpreted as a binding-mode contrast, not as a fully matched statistical panel.

### 10.3 Do Not Present Kd Association as Strong Causality

Because each group contains a limited number of systems, the relationship between Kd and landscape features should be described as an association or trend.

Recommended wording:

```text
We examined whether ligand potency is associated with changes in native-like basin occupancy and landscape compactness.
```

Avoid wording such as:

```text
Ligand potency determines the pocket landscape.
```

---

## 11. Recommended Analysis Workflow

### Step 1: Fragment-Level Structural Accuracy

For each case, compute:

```text
backbone RMSD
C-alpha RMSD
native contact recovery
local geometry deviation
```

This evaluates local prediction accuracy.

### Step 2: Module-Level Structural Accuracy

Aggregate biologically related fragments, especially:

```text
Switch-II module = Frag-B + Frag-F
P-loop / mutation-site module = Frag-D + Frag-A
```

This improves biological interpretability.

### Step 3: Landscape Reconstruction

For each fragment or module, reconstruct the landscape and analyze:

```text
basin count
basin occupancy
native-like basin occupancy
energy funnel score
conformational spread
low-energy conformer diversity
```

### Step 4: Group-Wise Comparison

Analyze each group separately:

```text
Asp12-recognizing SII-P binder series:
  drug-conditioned landscape difference

Switch-I/II groove or dimer-interface binder series:
  binding-mode-conditioned pocket response

BI-2865 mutation panel:
  mutation-conditioned pocket ensemble redistribution
```

### Step 5: Association Analysis

Within each group, examine associations such as:

```text
Kd value vs native-like occupancy
Kd value vs funnel score
Kd value vs basin compactness
Mutation type vs basin redistribution
Binding mode vs conformational spread
```

These analyses should be treated as supporting evidence rather than the only basis for the main conclusion.

---

## 12. Recommended Additional CSV Fields

To make post-processing clearer, the original case table should include the following fields:

```csv
evaluation_group,analysis_role,mutation_group,ligand_family,pocket_module
```

Field definitions:

| Field | Meaning |
|---|---|
| evaluation_group | The benchmark group to which the case belongs |
| analysis_role | The main comparison role of the case |
| mutation_group | WT, G12C, G12D, G12V, etc. |
| ligand_family | Ligand series or binding-mode family |
| pocket_module | Biological pocket region represented by the fragment |

Recommended values:

```text
evaluation_group:
  asp12_sii_p_binder_series
  switch_i_ii_groove_series
  bi2865_mutation_panel

analysis_role:
  drug_potency_series
  binding_mode_contrast
  mutation_panel

pocket_module:
  P-loop_mutation_site
  Switch-II_Nterm
  Switch-II_Cterm
  Switch-I_edge
  Nterm_Ploop_edge
  Alpha3_allosteric_region
```

---

## 13. Suggested Manuscript Description

```text
To evaluate pocket-level conformational prediction and perturbation sensitivity in KRAS, we constructed a curated benchmark consisting of three complementary structural series. The first series contains Asp12-recognizing noncovalent SII-P binders in the KRAS G12D background, enabling drug-conditioned comparison under a shared mutation context. The second series contains Switch-I/II groove or dimer-interface binders, providing a binding-mode contrast for pocket response analysis. The third series fixes the ligand BI-2865 and varies the KRAS mutation background across WT, G12C, G12D, and G12V, enabling mutation-conditioned ensemble comparison. For each structure, we selected pocket-relevant fragments covering the P-loop/mutation site, Switch-II pocket, Switch-I edge, and alpha3/allosteric region. This design allows fragment-level accuracy assessment and module-level landscape analysis of drug-, mutation-, and binding-mode-induced pocket perturbations.
```

---

## 14. Overall Conclusion

The current KRAS system and fragment selection is appropriate for the intended benchmark.

The design has several strengths:

1. It covers the key KRAS pocket regions involved in small-molecule binding.
2. It includes ligand-dependent, mutation-dependent, and binding-mode-dependent perturbations.
3. It contains a clean BI-2865 mutation panel for mutation-conditioned comparison.
4. It contains a G12D inhibitor series for drug-conditioned landscape comparison.
5. It includes Switch-I/II groove or dimer-interface binders as a binding-mode contrast.
6. The fragment-level setup is suitable for quantum sampling, while module-level aggregation improves biological interpretation.

Therefore, this benchmark can be used as a formal test set for KRAS pocket prediction and landscape perturbation analysis. The final analysis should avoid merging all systems into one global conclusion. Instead, it should compare systems within each evaluation group and report both fragment-level and module-level structural and landscape metrics.
