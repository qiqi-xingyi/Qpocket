# Learned ranker, held out by structure

Trained on eight structures and applied to the ninth, rotating
through all nine. Only held-out numbers appear here.

| | median RMSD (A) |
|---|---:|
| pool best, needs the answer | 2.578 |
| learned ranker, held out | 10.748 |
| learned ranker, on training structures | 3.803 |
| best hand-crafted feature | 5.271 |
| random pick | 6.777 |

The training-structure row is diagnostic, not a result. A large
gap between it and the held-out row means the model separated the
proteins it saw rather than the conformations.

## Per held-out case

| case | picked RMSD |
|---|---:|
| 6GJ6_G12D_Frag-D | 3.803 |
| 6GJ6_G12D_Frag-E | 3.929 |
| 6GJ6_G12D_Frag-F | 11.201 |
| 6GJ8_G12D_Frag-B | 20.180 |
| 6GJ8_G12D_Frag-D | 3.676 |
| 6GJ8_G12D_Frag-E | 4.007 |
| 6GJ8_G12D_Frag-F | 10.836 |
| 7RPZ_G12D_Frag-A | 10.569 |
| 7RPZ_G12D_Frag-B | 19.309 |
| 7RPZ_G12D_Frag-C | 11.214 |
| 7RPZ_G12D_Frag-F | 11.297 |
| 7RT1_G12D_Frag-A | 9.579 |
| 7RT1_G12D_Frag-B | 19.357 |
| 7RT1_G12D_Frag-C | 10.940 |
| 7RT1_G12D_Frag-F | 7.044 |
| 7RT4_G12D_Frag-A | 10.315 |
| 7RT4_G12D_Frag-B | 19.322 |
| 7RT4_G12D_Frag-C | 10.234 |
| 7RT4_G12D_Frag-F | 10.478 |
| 8AZV_WT_Frag-A | 10.361 |
| 8AZV_WT_Frag-B | 18.968 |
| 8AZV_WT_Frag-C | 11.338 |
| 8AZV_WT_Frag-F | 11.117 |
| 8AZX_G12C_Frag-A | 11.668 |
| 8AZX_G12C_Frag-B | 19.792 |
| 8AZX_G12C_Frag-C | 10.578 |
| 8AZX_G12C_Frag-F | 10.238 |
| 8AZY_G12D_Frag-A | 10.175 |
| 8AZY_G12D_Frag-B | 20.262 |
| 8AZY_G12D_Frag-C | 9.933 |
| 8AZY_G12D_Frag-F | 11.174 |
| 8AZZ_G12V_Frag-A | 10.388 |
| 8AZZ_G12V_Frag-B | 19.333 |
| 8AZZ_G12V_Frag-C | 9.949 |
| 8AZZ_G12V_Frag-F | 10.748 |

