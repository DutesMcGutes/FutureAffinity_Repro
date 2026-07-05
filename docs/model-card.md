# Model Card

**Model**: none. FutureAffinity Reproduction contains no trained model and no model weights.

**Intended use**: education. Understanding how a co-folding + affinity foundation model's pieces
(feature extraction, embeddings, a Pairformer-style trunk, diffusion-style structure sampling,
multi-task heads, synthetic supervision) fit together and communicate, by running each piece
directly on plain Python data.

**Out-of-scope use**: any structural, affinity, or biological conclusion. Predicted coordinates
come from a deterministic demo generator, not a trained network; predicted lDDT/contacts/affinity/
ΔΔG come from real formulas applied to those demo coordinates, not from learned weights. See
docs/limitations.md.

**Training data**: none used, none required to run this repository.

**Evaluation**: `tests/test_tutorials.py` checks that each chapter's reference solution produces
exactly the expected (frozen, deterministic) output on `examples/aspirin_lysozyme.json`;
`tests/test_pipeline.py` checks shape and invariant properties (symmetry, self-consistency,
sortedness) of each stage directly. Neither is a measure of predictive accuracy.
