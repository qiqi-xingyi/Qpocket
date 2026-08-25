# G1 KRAS pilot — frozen endpoints

This specification fixes how the G1 arms are compared. It is written to
be settled before any arm's results are read, because a selector or an
endpoint chosen after seeing results is not an endpoint but a fitted
parameter.

## Selector

The primary prediction for a cell is the representative of the basin
ranked first by the production selector:

```
basin_score = best_refined_score − 0.5 · log(basin_weight + 1e-12)
```

ascending, rank 0 first. This is `ras_folding.postprocess.selector.rank_basins`
at its production default, unchanged. Freezing the default rather than
choosing a value for this comparison matters twice over: it is the
setting the hardware arm was produced under, and a bonus tuned here would
be a parameter fitted to the comparison it is supposed to adjudicate.

The selector reads a refined energy and a basin weight. It reads no
native coordinates, no reference structure, and no RMSD.

## Where native coordinates may enter

Native coordinates score a structure that has already been chosen. They
do not enter generation, tuning, stopping, ranking, or representative
selection in any arm.

The consequence worth stating plainly: a cell's primary number is the
error of whatever the frozen selector picked, not the error of the best
structure the arm produced. An arm that samples an excellent conformation
but ranks it poorly is scored on the ranking, because that is what a user
of the method would obtain.

## Primary endpoint

In-frame Cα RMSD, no superposition, between the rank-0 representative and
the native fragment.

Superposition is withheld deliberately. Every arm receives the flanking
anchor coordinates and the surrounding receptor, and produces a structure
already placed in that frame. A Kabsch fit would discard the placement
those inputs determine and score shape alone, crediting the method for
information it was handed. In-frame RMSD is reported as the primary
number; Kabsch RMSD is reported beside it, and the gap between them is
itself informative about placement error.

## Secondary endpoints

**Fixed-budget oracle-best RMSD.** The lowest RMSD anywhere in the
candidate pool. This measures reachability and coverage — whether an arm
put a good conformation in the pool at all — and is not predictive
accuracy, because obtaining it requires the native structure the method
does not have. It is never reported as the method's accuracy.

**Acceptance and coverage counts.** Valid rate, accepted count, distinct
states, and for arm M the per-move acceptance rates. These characterise
the samplers and do not by themselves rank them.

## Budget matching

Both definitions are reported for every comparison:

- **proposal-matched** — equal proposals offered to the downstream, where
  one prior rollout attempt or one Metropolis proposal matches one shot;
- **valid-unique-matched** — equal valid distinct candidates delivered to
  the downstream.

The two differ by the valid rate, near 14% for the prior arm, so they are
not interchangeable and neither alone is sufficient.

## Repeats and what they permit

Arms P and M have three independent repeats per task, seeded
deterministically from `(master seed, arm, task, repeat)`. Arm Q has one:
it was executed once on hardware.

Within-arm variability is therefore available for P and M and unavailable
for Q. Any interval, test, or error bar that requires it is reported for
P and M only, and arm Q is shown as a single observation rather than
given a spread it does not have. Candidate, shot, and seed counts are not
biological replicates and are not treated as such.

## Conclusions this pilot can support

The comparison of interest is Q against P: whether hardware-executed
circuit sampling contributes anything beyond the prior it was built from.
The permitted readings are fixed in advance:

- Q better than P on the primary endpoint — a measurable sampler
  contribution. Not quantum advantage, not speedup, not classical
  intractability.
- Q differs in distribution but not on the primary endpoint — a
  distributional transformation with no demonstrated benefit.
- Q indistinguishable from P — no detected contribution from hardware
  execution.
- P or M better than Q — the classical result stands, and hardware-utility
  or superiority wording is removed from the manuscript.

A null or negative result is an outcome of this pilot, not a failure of
it. It changes the claim. It does not license changing the endpoint, the
selector, the matching rule, or the arm set afterwards.

## What this pilot cannot settle

Nine KRAS systems and thirty-six fragments do not establish generality
across protein families, therapeutic prediction, or cryptic-pocket
discovery. The pilot compares samplers under one fixed local task.

Input-matched comparison against established structure-conditioned local
methods is a separate arm and is not part of G1.
