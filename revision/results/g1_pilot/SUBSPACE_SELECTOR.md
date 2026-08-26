# Selection by inference in the sampled subspace

Cases: 35. Subspace: up to 3,000 candidates per case.

| variant | diagonal | coupling | median picked RMSD | median rho |
|---|---|---|---:|---:|
| hybrid | energy | Hamming-1 + RMSD | 5.076 | +0.363 |
| pauli | flat | Hamming-1 | 6.314 | -0.004 |
| rmsd | flat | RMSD kernel | 5.076 | +0.361 |
| energy | energy | RMSD kernel | 5.076 | +0.363 |
| energy alone, no subspace | energy | none | 5.835 | +0.229 |

| | median RMSD |
|---|---:|
| pool best, needs the answer | 2.641 |
| best hand-crafted feature | 5.271 |
| random pick | 6.780 |

## Ground-state localisation

The participation ratio is the fraction of the subspace the ground
state actually occupies. A state spread across most of the
subspace is not selecting anything.

| variant | median participation ratio |
|---|---:|
| hybrid | 0.966 |
| pauli | 0.000 |
| rmsd | 0.966 |
| energy | 0.966 |

