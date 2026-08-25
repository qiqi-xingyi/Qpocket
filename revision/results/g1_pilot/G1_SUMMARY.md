# G1 pilot — endpoint summary

Endpoints and selector were frozen before any arm was read; see
`configs/G1_FROZEN_ENDPOINTS.md`. The primary endpoint is in-frame
CA RMSD of the representative the frozen selector ranked first.

## Cells

| arm | cells | with endpoint | no candidate | no native |
|---|---|---|---|---|
| Q | 36 | 35 | 0 | 1 |
| P | 108 | 105 | 0 | 3 |
| M | 108 | 100 | 5 | 3 |

Cells that produced no candidate at all:

- M 6GJ6_G12D_Frag-E repeat_0
- M 6GJ6_G12D_Frag-E repeat_1
- M 6GJ6_G12D_Frag-E repeat_2
- M 6GJ8_G12D_Frag-E repeat_0
- M 6GJ8_G12D_Frag-E repeat_1

These are counted, not dropped. An arm that delivers nothing on
a task has reported something about the arm.

## Primary endpoint — in-frame RMSD of the selected representative

Kabsch RMSD is shown beside it. The gap between the two is
placement error: the arms are given the anchors and the receptor
and return a structure already in that frame, and superposition
discards exactly that.

| arm | n cells | in-frame median | Kabsch median | in-frame min | in-frame max |
|---|---|---|---|---|---|
| Q | 35 | 5.805 | 4.118 | 1.354 | 13.388 |
| P | 105 | 5.701 | 4.053 | 1.157 | 14.004 |
| M | 100 | 5.557 | 4.052 | 1.728 | 13.971 |

## Per task

P and M are the median of three repeats, with the repeat spread in
brackets. Q was executed once and has no spread to report.

| task | Q | P (spread) | M (spread) |
|---|---|---|---|
| 6GJ6_G12D_Frag-B | -- | -- | -- |
| 6GJ6_G12D_Frag-D | 2.032 | 2.056 [0.334] | 2.056 [0.369] |
| 6GJ6_G12D_Frag-E | 1.354 | 1.975 [0.664] | -- |
| 6GJ6_G12D_Frag-F | 6.256 | 5.908 [0.880] | 5.244 [1.377] |
| 6GJ8_G12D_Frag-B | 13.388 | 12.810 [2.532] | 10.761 [1.292] |
| 6GJ8_G12D_Frag-D | 2.098 | 1.827 [0.127] | 1.861 [0.385] |
| 6GJ8_G12D_Frag-E | 2.117 | 1.157 [0.279] | 1.728 [0.000] |
| 6GJ8_G12D_Frag-F | 4.641 | 6.761 [2.511] | 5.343 [1.761] |
| 7RPZ_G12D_Frag-A | 5.776 | 3.975 [2.072] | 4.776 [2.134] |
| 7RPZ_G12D_Frag-B | 12.950 | 13.249 [2.518] | 12.017 [1.642] |
| 7RPZ_G12D_Frag-C | 5.805 | 5.316 [0.684] | 5.830 [0.673] |
| 7RPZ_G12D_Frag-F | 5.321 | 5.701 [2.018] | 7.110 [1.242] |
| 7RT1_G12D_Frag-A | 5.089 | 4.662 [1.061] | 4.331 [1.254] |
| 7RT1_G12D_Frag-B | 11.944 | 10.589 [2.126] | 11.057 [2.923] |
| 7RT1_G12D_Frag-C | 5.319 | 4.993 [1.339] | 5.026 [0.435] |
| 7RT1_G12D_Frag-F | 6.197 | 5.048 [1.688] | 6.080 [2.219] |
| 7RT4_G12D_Frag-A | 6.132 | 5.369 [0.948] | 4.146 [1.890] |
| 7RT4_G12D_Frag-B | 11.860 | 11.557 [1.206] | 10.626 [1.148] |
| 7RT4_G12D_Frag-C | 5.392 | 5.299 [0.932] | 5.209 [0.458] |
| 7RT4_G12D_Frag-F | 6.684 | 6.759 [1.380] | 5.754 [1.028] |
| 8AZV_WT_Frag-A | 5.721 | 5.722 [1.028] | 5.087 [1.632] |
| 8AZV_WT_Frag-B | 11.797 | 12.872 [1.984] | 11.473 [1.858] |
| 8AZV_WT_Frag-C | 5.559 | 5.029 [1.146] | 5.222 [0.566] |
| 8AZV_WT_Frag-F | 3.948 | 7.522 [0.925] | 7.244 [1.394] |
| 8AZX_G12C_Frag-A | 6.345 | 5.187 [2.296] | 4.198 [2.888] |
| 8AZX_G12C_Frag-B | 11.481 | 11.439 [2.420] | 11.860 [2.996] |
| 8AZX_G12C_Frag-C | 5.232 | 5.164 [0.533] | 5.104 [0.342] |
| 8AZX_G12C_Frag-F | 5.422 | 5.988 [1.072] | 7.152 [2.101] |
| 8AZY_G12D_Frag-A | 5.866 | 4.252 [1.292] | 4.711 [2.299] |
| 8AZY_G12D_Frag-B | 11.880 | 12.198 [0.497] | 11.767 [0.378] |
| 8AZY_G12D_Frag-C | 5.141 | 5.443 [0.480] | 5.209 [0.097] |
| 8AZY_G12D_Frag-F | 4.802 | 7.716 [1.020] | 7.148 [1.203] |
| 8AZZ_G12V_Frag-A | 5.870 | 5.019 [0.937] | 5.121 [0.311] |
| 8AZZ_G12V_Frag-B | 10.909 | 13.055 [1.451] | 11.822 [1.856] |
| 8AZZ_G12V_Frag-C | 5.983 | 5.138 [1.679] | 5.435 [1.500] |
| 8AZZ_G12V_Frag-F | 6.106 | 6.276 [3.277] | 6.526 [1.087] |

## Paired comparisons

### Q vs P

- paired tasks: 35
- median difference (Q − P): +0.093 A
- bootstrap 95% CI of the median difference: [-0.170, +0.489] A
- Wilcoxon signed-rank, two-sided: p = 0.3809 (n = 35)
- tasks where Q is closer to native: 15 / 35

A negative median difference means Q selected a representative
closer to the native fragment than P did.

### Q vs M

- paired tasks: 34
- median difference (Q − M): +0.308 A
- bootstrap 95% CI of the median difference: [+0.113, +0.749] A
- Wilcoxon signed-rank, two-sided: p = 0.0754 (n = 34)
- tasks where Q is closer to native: 11 / 34

A negative median difference means Q selected a representative
closer to the native fragment than M did.

## Secondary — oracle-best in-frame RMSD

Coverage and reachability over each arm's own candidates. Obtaining
this requires the native structure the method does not have, so it
is not predictive accuracy and is not the arms' score.

| arm | n cells | median | min | median pool size |
|---|---|---|---|---|
| Q | 35 | 1.869 | 0.564 | 111,297 |
| P | 105 | 2.602 | 0.666 | 13,620 |
| M | 100 | 2.612 | 0.666 | 13,485 |

## What the ranking costs

The gap between the selected representative and the best structure
in the same pool is the price of ranking without the native. It is
reported because a method is used through its selector.

| arm | median selected | median pool best | median gap |
|---|---|---|---|
| Q | 5.805 | 1.869 | 3.975 |
| P | 5.701 | 2.602 | 3.540 |
| M | 5.557 | 2.612 | 3.191 |

## Selected versus oracle

The two quantities differ by what they require. The oracle figure
is the best structure in the pool, found by comparing candidates
with the native fragment. The selected figure is what the frozen
native-independent selector returns, which is what a user without
the answer would obtain.

| arm | selected in-frame | selected Kabsch | pool best in-frame | pool best Kabsch |
|---|---|---|---|---|
| Q | 5.805 | 4.118 | 1.869 | 1.729 |
| P | 5.701 | 4.053 | 2.602 | 2.052 |
| M | 5.557 | 4.052 | 2.612 | 2.207 |

Across all three arms the selector costs more than three angstroms
against the best structure already present in the same pool. The
arms differ from each other by a fraction of that. On this
benchmark the ranking, not the sampler, is what limits the result.

