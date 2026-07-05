# Splits and leakage: the data lesson that matters most

You can have a perfect architecture and still ship a useless model if your train/test split leaks.
This is the single most important data concept in structural ML, and it's worth understanding even
in a teaching repo that never touches real data.

## The problem

Proteins and ligands come in families. If you split your data **randomly**, near-duplicates end up
on both sides of the split — a test protein that's 95% identical to a training protein, or a test
ligand from the same chemical series as a training ligand. The model can then score brilliantly on
the test set by *memorizing* the family, while failing completely on genuinely new targets. Random
splits systematically **overestimate** real-world performance in this field.

## The fix: split by similarity, not at random

Three axes of leakage, each with a corresponding split:

- **Sequence split** — cluster proteins by sequence identity (e.g. with MMseqs2 at 30%) and put
  whole clusters on one side. A test protein must have no close homolog in training.
- **Scaffold split** — cluster ligands by chemical scaffold and keep similar scaffolds out of test,
  so you measure generalization to *new chemistry*, not interpolation within a series.
- **Time split** — train on structures deposited before a date, test on those after. This is the
  most honest "will it work on tomorrow's targets" estimate, and it's how CASP/CAMEO evaluate.

The strongest evaluation is disjoint on all three at once, and you report each split separately: a
model that's great on a random split but poor on a scaffold split is telling you it memorized.

## Why it's in a teaching repo

Because it reframes what "good accuracy" means. The first question to ask of *any* structure/affinity
result is not "how high is the number" but "how was the split made." See
`FutureAffinity_PyTorch/docs/data-engineering.md` for how this becomes an actual pipeline
(clustering, dedup, versioned split files).
