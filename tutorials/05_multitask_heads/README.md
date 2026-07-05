# Chapter 05: Multi-Task Heads

Structure isn't the only thing worth predicting from a trunk representation. This chapter builds
three more "heads" that all read from the same kind of representation the trunk produces:

- **Confidence (lDDT)** -- a real, well-defined metric (not a stand-in) computed from two sets of
  coordinates: what fraction of nearby-pair distances survive within tolerance. This is what
  AlphaFold's pLDDT is trained to predict.
- **Contacts** -- a real distance-thresholded contact map.
- **Affinity** -- a toy `E(protein, ligand, conformation)`: notice it's a function of *which
  coordinates* you pass in, not just which atoms are present. That's the point: affinity here is
  one observable of a specific conformation, not a lookup keyed on sequence/ligand identity alone.

None of these are learned yet -- same "real computation, no learned weights" idea as the earlier
chapters. The payoff of building them this way is that every one of them slots into
FutureAffinity_PyTorch's multi-task training loop with the same interface: given a structure (real or
sampled), produce one more observable of it.

## Student Task

Open `exercise.py` and study the TODO-annotated functions:

1. `compute_lddt` -- per-atom lDDT from two sets of coordinates.
2. `contact_map` -- a distance-thresholded contact map.
3. `predict_affinity` -- a toy conformation-dependent affinity score.

Then run:

```bash
python tutorials/05_multitask_heads/exercise.py
```

To compare against the reference implementation (which also adds a ΔΔG head):

```bash
python scripts/generate_test_results.py 05_multitask_heads
```
