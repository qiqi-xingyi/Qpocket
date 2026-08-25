# Baseline comparison under one output convention

The recorded comparison scores this pipeline by the lowest RMSD
among every structure it generated, found by comparing candidates
with the native fragment, and scores each sequence-to-structure
model by its prediction. Recovering the first requires the answer;
the second does not. Placing them in one column compares a search
over a pool against a prediction.

Cases: 35. Median pool searched: 112,834 structures.

The pipeline's single-prediction column below is the representative
its own frozen selector ranks first, which is what a user without
the native fragment obtains.

| method | structures per case | median Kabsch RMSD |
|---|---:|---:|
| this pipeline, pool minimum | 112,834 | 1.292 |
| this pipeline, frozen selector | 1 | 4.118 |
| AF3 | 5 | 1.717 |
| ESMFold | 1 | 1.802 |
| OmegaFold | 1 | 4.923 |
| OpenFold | 1 | 3.152 |

## Cases won, by convention

| baseline | pool minimum wins | frozen selector wins |
|---|---:|---:|
| AF3 | 28/35 | 0/35 |
| ESMFold | 27/35 | 3/35 |
| OmegaFold | 35/35 | 23/35 |
| OpenFold | 28/35 | 4/35 |

The reported advantage over AF3, ESMFold and OpenFold does not
survive scoring this pipeline the way those models are scored. It
holds against OmegaFold, whose recorded errors are the largest of
the four.

## What this does and does not show

The pool minimum remains a real property of the sampler: a
structure close to the native is present among the generated
candidates, and on most cases it is closer than any baseline's
prediction. That is coverage, and it is worth reporting as
coverage.

What it is not is an accuracy the method can deliver. Selecting it
requires the native structure, and when the pipeline's own
selector chooses without that, the choice is several angstroms
worse and the ordering against the baselines reverses.

Two limits belong with this. The selector here is the production
basin ranking at its default; a different selector could recover
some of the gap, and nothing here shows that none can. And the
input asymmetry runs the other way: this pipeline receives the
solved receptor, both flanking anchors, and the ligand position,
while the baselines receive a sequence window. The reversal
appears despite that advantage, not in its absence.

## Per case

| case | pool | pool min | selector | AF3 | ESMFold | OmegaFold | OpenFold |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6GJ6_G12D_Frag-D | 63,157 | 0.465 | 1.346 | 0.592 | 1.832 | 4.923 | 2.550 |
| 6GJ6_G12D_Frag-E | 9,423 | 0.396 | 1.261 | 0.466 | 1.705 | 4.855 | 0.606 |
| 6GJ6_G12D_Frag-F | 112,161 | 1.257 | 4.289 | 1.703 | 1.216 | 4.080 | 2.119 |
| 6GJ8_G12D_Frag-B | 216,725 | 1.960 | 7.375 | 1.985 | 1.991 | 6.487 | 2.800 |
| 6GJ8_G12D_Frag-D | 44,619 | 0.404 | 1.538 | 1.323 | 1.574 | 5.489 | 3.366 |
| 6GJ8_G12D_Frag-E | 11,696 | 0.300 | 1.379 | 0.845 | 1.257 | 4.654 | 0.903 |
| 6GJ8_G12D_Frag-F | 106,059 | 0.958 | 3.182 | 0.974 | 1.007 | 3.715 | 1.748 |
| 7RPZ_G12D_Frag-A | 83,022 | 0.928 | 3.850 | 1.612 | 1.754 | 18.852 | 3.019 |
| 7RPZ_G12D_Frag-B | 243,328 | 1.761 | 8.393 | 2.077 | 4.217 | 6.379 | 4.150 |
| 7RPZ_G12D_Frag-C | 158,029 | 1.974 | 4.246 | 1.980 | 1.539 | 4.212 | 1.481 |
| 7RPZ_G12D_Frag-F | 111,916 | 1.304 | 4.004 | 1.998 | 1.826 | 4.315 | 3.152 |
| 7RT1_G12D_Frag-A | 83,080 | 1.089 | 2.880 | 2.566 | 1.649 | 18.936 | 2.959 |
| 7RT1_G12D_Frag-B | 240,356 | 1.701 | 7.338 | 2.052 | 4.209 | 6.387 | 4.118 |
| 7RT1_G12D_Frag-C | 157,015 | 1.808 | 4.567 | 1.421 | 1.546 | 4.213 | 1.508 |
| 7RT1_G12D_Frag-F | 117,096 | 1.085 | 4.103 | 1.626 | 1.968 | 4.410 | 3.266 |
| 7RT4_G12D_Frag-A | 80,473 | 0.752 | 4.118 | 1.531 | 1.601 | 18.687 | 3.031 |
| 7RT4_G12D_Frag-B | 232,912 | 1.786 | 8.047 | 4.168 | 4.382 | 6.499 | 4.394 |
| 7RT4_G12D_Frag-C | 169,956 | 2.021 | 4.465 | 0.688 | 1.558 | 4.271 | 1.420 |
| 7RT4_G12D_Frag-F | 111,166 | 1.120 | 4.665 | 1.717 | 2.059 | 4.553 | 3.507 |
| 8AZV_WT_Frag-A | 81,012 | 0.978 | 4.319 | 0.997 | 2.272 | 17.829 | 3.744 |
| 8AZV_WT_Frag-B | 228,131 | 1.763 | 7.352 | 3.723 | 4.376 | 6.611 | 5.143 |
| 8AZV_WT_Frag-C | 154,828 | 2.042 | 4.394 | 0.639 | 0.878 | 4.822 | 1.395 |
| 8AZV_WT_Frag-F | 113,123 | 1.292 | 3.116 | 2.229 | 2.545 | 4.901 | 3.329 |
| 8AZX_G12C_Frag-A | 82,312 | 1.086 | 3.963 | 0.661 | 1.802 | 18.504 | 3.247 |
| 8AZX_G12C_Frag-B | 229,434 | 1.849 | 7.642 | 3.727 | 4.382 | 6.597 | 5.139 |
| 8AZX_G12C_Frag-C | 155,811 | 2.084 | 3.582 | 1.345 | 1.584 | 4.844 | 1.389 |
| 8AZX_G12C_Frag-F | 112,066 | 1.122 | 3.950 | 2.249 | 2.568 | 4.912 | 3.350 |
| 8AZY_G12D_Frag-A | 80,768 | 1.053 | 3.743 | 2.485 | 1.636 | 18.772 | 3.014 |
| 8AZY_G12D_Frag-B | 227,250 | 1.845 | 6.384 | 3.727 | 4.374 | 6.614 | 5.149 |
| 8AZY_G12D_Frag-C | 154,991 | 1.970 | 4.395 | 1.973 | 1.608 | 4.858 | 1.393 |
| 8AZY_G12D_Frag-F | 112,834 | 1.245 | 3.995 | 2.315 | 2.633 | 4.977 | 3.433 |
| 8AZZ_G12V_Frag-A | 81,262 | 1.038 | 3.657 | 0.313 | 1.459 | 18.570 | 3.422 |
| 8AZZ_G12V_Frag-B | 227,518 | 1.770 | 7.499 | 3.708 | 4.359 | 6.593 | 5.123 |
| 8AZZ_G12V_Frag-C | 156,086 | 1.735 | 4.481 | 0.327 | 0.560 | 4.810 | 1.393 |
| 8AZZ_G12V_Frag-F | 111,888 | 1.343 | 3.504 | 2.227 | 2.536 | 4.906 | 3.303 |

