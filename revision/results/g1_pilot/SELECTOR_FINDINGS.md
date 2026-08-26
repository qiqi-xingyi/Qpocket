# Selecting without the native fragment

The pipeline generates a structure close to the native and then does not
return it. Across the pilot the pool holds a candidate near 2.5 A while
the frozen selector returns one near 5.8 A. This records what was tried
to close that gap and what each attempt established.

## What was tested

Roughly twenty native-independent signals, in four families.

**Physical energy.** The production full energy and filter Hamiltonian,
and each of their components separately. Rank correlation with accuracy
sits between 0.05 and 0.17; picking the lowest-energy candidate lands
near 6.2 A against a random pick at 6.8 A. Two components turn out to
carry nothing by construction: the clash term, because the prior already
excludes clashing directions during generation, and the radius-of-gyration
term, because it is identically zero when no target radius is set.

**Ensemble statistics.** Distance to the pool mean, k-nearest-neighbour
density, spectral centrality, a pairwise model fitted to the sampled
codes, and its marginal-only null. A clean split appears here: every
statistic that seeks the *mode* performs at or below a random pick, while
those measuring distance to the *mean* carry the strongest signal found
by hand, 0.39 with the same sign in all thirty-five cases.

**Receptor and ligand context.** Packing against the receptor correlates
at 0.33. Ligand proximity correlates *negatively*, at -0.26: candidates
closer to the ligand are worse. The corridor prior pulls sampling toward
the crystallographic ligand centroid, so proximity to the ligand marks
how strongly a candidate absorbed that pull rather than whether it is
right.

**Fitted rankings.** A ridge combination of the hand-crafted features,
and a model reading per-residue geometry with its receptor and ligand
surroundings. Both held out whole structures.

## What the pattern says

Every signal that measures how *typical* a candidate is for the ensemble
tracks the sampler's bias, and the bias is the error. Mode-seeking fails
for this reason, ligand proximity fails for this reason, and the pairwise
model fails for this reason. The signals that survive are the ones tied
to constraints the prior cannot move: the receptor surface, the anchors,
and the geometry those impose.

That also explains why combining features gains so little. A ridge fit
over seven features reaches 5.16 A held out against 5.27 A for the best
single feature. The features are not independent; they are several
measurements of one thing.

## Subspace inference

Weight in the ground state of the refinement Hamiltonian was tested as a
selector, with the diagonal and each coupling removable independently.

| variant | diagonal | coupling | picked RMSD | rho | participation |
|---|---|---|---:|---:|---:|
| hybrid | energy | Hamming-1 + RMSD | 5.076 | +0.363 | 0.966 |
| pauli | flat | Hamming-1 | 6.314 | -0.004 | 0.000 |
| rmsd | flat | RMSD kernel | 5.076 | +0.361 | 0.966 |
| energy | energy | RMSD kernel | 5.076 | +0.363 | 0.966 |

Three variants return identical numbers. Removing the energy diagonal
changes nothing and removing the Hamming-1 coupling changes nothing, so
the RMSD kernel accounts for the whole result. A participation ratio of
0.966 says the ground state occupies almost the entire subspace: it is
close to uniform and is not concentrating on anything, and taking its
largest component amounts to a geometric centrality measure.

The Hamming-1 variant is degenerate for a concrete reason. Among three
thousand sampled candidates the pairs differing in exactly one bond
number 74 out of 13.5 million, and none at all for the eleven- and
fourteen-bond fragments; the mean Hamming distance between candidates
reaches 98% of its maximum. The transverse-field coupling has no support
in the sampled subspace, so the refinement's off-diagonal structure comes
entirely from the classical RMSD kernel.

## The learned ranker

| | median RMSD |
|---|---:|
| pool best, needs the answer | 2.578 |
| learned ranker, training structures | 3.803 |
| best hand-crafted feature | 5.271 |
| random pick | 6.777 |
| learned ranker, held-out structures | 10.748 |

The two rows for the same model are the result. On structures it trained
on it reaches 3.80 A, better than every hand-crafted signal and close to
the pool best; on a structure it has not seen it falls below a random
pick.

This is not evidence that the approach fails. It is evidence that the
information is present and extractable from per-residue geometry and its
surroundings -- a scalar summary loses what a model reading the structure
recovers -- and that eight training structures are too few to learn it in
a form that transfers. Nine structures cannot support a model that must
generalise across protein context; comparable quality-assessment models
are trained on thousands.

## Where this leaves the selector

Ranking is the binding constraint on this benchmark, not sampling. The
gap between the pool best and the selected representative exceeds three
angstroms for every arm, and the arms differ from each other by a
fraction of that.

Closing it is a training-set problem before it is a modelling problem. It
needs many diverse systems with known answers, which is a separate study
and cannot be assembled from nine structures.

Until then the defensible product is the ensemble rather than a single
structure. A pool that reliably contains a near-native conformation is
useful to ensemble docking, which consumes many receptor conformations by
design; a single prediction several angstroms off is not.
