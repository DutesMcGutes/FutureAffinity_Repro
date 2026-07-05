# FutureAffinity Reproduction

A from-scratch, educational walkthrough of a co-folding + affinity foundation model, written in
plain, dependency-free Python. The goal is to actually understand how the pieces fit together:
input parsing (protein *and* ligand), featurization, embeddings, a Pairformer-style trunk, a
diffusion-style ensemble sampler, multi-task heads (confidence, contacts, affinity, ΔΔG), and
synthetic-supervision pretraining -- instead of just reading about the ideas and nodding along.

This extends the two-chapter-per-concept style of an AlphaFold3 protein-only reproduction to the
FutureAffinity project brief: **structure prediction is becoming one task among many**, and the
richest source of supervision isn't scarce affinity labels, it's everything else (structures,
sequences, synthetic docking, ensembles) a model can learn from jointly.

```text
co-folding JSON (protein + ligand) -> tokens/atoms -> embeddings -> trunk -> ensemble -> heads -> mmCIF
```

One thing to be upfront about: there are no learned weights anywhere in here. See
`docs/limitations.md` before reading anything into the numbers it produces.

## Why this exists

This tries to sit between "official, production-grade inference code" and "a scattered notebook
that explains one piece in isolation": readable enough to actually learn from, structured enough
that you could build on it, and honest about what it can't do yet. The paired repo,
`FutureAffinity_PyTorch`, is the real, trainable version of the same architecture.

## What's here

- a co-folding JSON parser (protein sequences *and* an optional ligand SMILES)
- a feature builder producing one token stream for both protein residues and ligand atoms
- deterministic token and pair embeddings, including a stand-in protein-language-model contribution
- a dependency-free Pairformer-style trunk, with both a uniform-averaging triangle update and a
  real (if tiny) softmax-attention triangle update, side by side
- a diffusion-style coordinate sampler, extended to draw an *ensemble* of samples and report
  per-atom uncertainty from their spread
- real (if untrained) multi-task heads: lDDT confidence, a distance-thresholded contact map, a
  conformation-dependent affinity score, and a proximity-weighted ΔΔG estimate
- a dependency-free toy docking engine for synthetic-supervision pretraining
- a minimal mmCIF writer (ligand atoms as `HETATM`)
- an end-to-end pipeline chaining all of the above together
- a runnable demo script and a full test suite
- seven complete tutorial chapters, each with a working exercise and a reference solution

## Quickstart

From the repo root:

```bash
python scripts/run_demo.py
```

That writes `examples/outputs/aspirin_lysozyme_demo.cif`. To run the tests:

```bash
python -m unittest discover tests
```

Everything here runs on the Python standard library. No torch, no numpy, nothing to install first.
The package itself is named `futureaffinity_reproduction`.

## Working through the tutorials

Start with chapter one:

```bash
python tutorials/01_feature_extraction/exercise.py
python scripts/generate_test_results.py 01_feature_extraction
```

| Chapter | Topic | Status |
| --- | --- | --- |
| 01 | Feature extraction (protein + ligand) | Ready |
| 02 | Input embedding | Ready |
| 03 | Pairformer (averaging vs. real attention) | Ready |
| 04 | Diffusion (single sample vs. ensemble) | Ready |
| 05 | Multi-task heads | Ready |
| 06 | Synthetic supervision | Ready |
| 07 | Inference and export | Ready |

Each chapter has its own README with the real walkthrough; this top-level one is just the map.

## Weight policy

No model weights live in this repo, and none ever will. See docs/data-and-weights.md.

## Before you read too much into the output

Check docs/model-card.md and docs/limitations.md first. Short version: the pipeline runs end to
end, but nothing in it is trained.

## Sources and inspiration

See docs/references.md.
