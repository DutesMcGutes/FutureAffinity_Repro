# Data and Weights

FutureAffinity Reproduction does not include any real structure, affinity, or protein-language-model
weights or datasets.

The demo pipeline (Chapters 01-07, `scripts/run_demo.py`) does not use protected weights. It
produces deterministic structure-like coordinates and toy-but-real head outputs (lDDT, contacts,
a physical-distance affinity score) so contributors can test the full pipeline without any
restricted or bulk-download assets.

If you want real data or weights to compare against:

- **AlphaFold 3 weights**: get them directly from Google DeepMind, under their terms. This repo
  never has and never will bundle them.
- **PDBbind / BindingDB / ESM2**: see `FutureAffinity_PyTorch/docs/data-and-weights.md` in the paired
  implementation repo -- that's where real dataset loaders and download instructions live. This
  reproduction repo intentionally stays dependency-light and works entirely on synthetic/toy data.
