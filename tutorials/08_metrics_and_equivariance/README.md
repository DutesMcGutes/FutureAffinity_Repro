# Chapter 08: Metrics and Why Rotation Matters

This chapter ties together two ideas that turn out to be the same idea: **how you evaluate a
structure** and **why orientation shouldn't matter**.

Take a predicted structure and rotate the whole thing 90° -- a rigid motion that changes nothing
about its actual geometry. Then score it two ways:

- **Naive coordinate MSE** (compare atom positions directly): blows up, because every atom moved.
- **lDDT** (compare *distances between atoms*): unchanged at 100, because internal distances are
  invariant to rotation.

That contrast is the whole lesson. The naive metric is fooled by something that doesn't matter; the
good metric isn't. And it's the same reason the real model (`FutureAffinity_PyTorch`) gets
rotation-robustness by *centering + random-rotation augmentation* rather than by trusting absolute
coordinates: internal geometry is what's real, the frame is arbitrary. See `docs/equivariance.md`
and `docs/metrics.md`.

## Student Task

Open `exercise.py` and study the TODO-annotated functions:

1. `rotate_z` -- rotate the whole structure about the z-axis (a rigid motion).
2. `naive_coordinate_mse` -- the un-aligned metric that gets fooled.
3. `compute_lddt` -- the superposition-free metric that doesn't.

Then run it and watch the two metrics disagree under rotation:

```bash
python tutorials/08_metrics_and_equivariance/exercise.py
```

To compare against the reference implementation:

```bash
python scripts/generate_test_results.py 08_metrics_and_equivariance
```
