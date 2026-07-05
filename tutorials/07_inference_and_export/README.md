# Chapter 07: Inference and Export

The capstone chapter. Chapters 01-06 each built one piece in isolation (features, embeddings, the
trunk, diffusion + ensembles, multi-task heads, synthetic supervision); this one chains the
finished library pieces into a single end-to-end call and writes a real mmCIF file.

The shape here is deliberate, and it's the same shape `predict()` has in FutureAffinity_PyTorch's
`inference/predict.py`: sequence(+ligand) in; a *structure ensemble* (not one structure) +
per-token confidence + a contact map + an affinity estimate + a real uncertainty number out. A
single point-estimate structure would hide exactly the information (how confident is this,
really?) that makes a prediction useful to act on.

## Student Task

Open `exercise.py` and complete `run_futureaffinity`:

1. Parse the input and build features.
2. Run a diffusion ensemble (not a single sample).
3. Compute per-atom uncertainty from that ensemble.
4. Run the confidence, contact, and affinity heads on a representative structure.
5. Export the representative structure to mmCIF.

Then run:

```bash
python tutorials/07_inference_and_export/exercise.py
```

That writes `examples/outputs/exercise_output.cif`. To compare the numeric summary against the
reference implementation:

```bash
python scripts/generate_test_results.py 07_inference_and_export
```
