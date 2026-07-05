# Tutorials

Each chapter has the same shape:

- Read the chapter README.
- Open the exercise module.
- Replace the TODO sections.
- Run the exercise script directly, and compare against `solutions/`.
- Run `python scripts/generate_test_results.py <chapter>` to compare against the frozen expected output.

Chapters 01-04 extend the AlphaFold3 reproduction arc (features, embeddings, trunk, diffusion) to
co-folding (protein + ligand) and structural ensembles. Chapters 05-07 are new: multi-task heads,
synthetic supervision, and end-to-end inference + export -- the ideas from the FutureAffinity project
brief about using every available signal, not just scarce affinity labels.

## Chapter Status

| Chapter | Topic | Status |
| --- | --- | --- |
| 01 | Feature extraction (protein + ligand) | Ready |
| 02 | Input embedding | Ready |
| 03 | Pairformer (averaging vs. real attention) | Ready |
| 04 | Diffusion (single sample vs. ensemble) | Ready |
| 05 | Multi-task heads (confidence, contacts, affinity) | Ready |
| 06 | Synthetic supervision (toy docking) | Ready |
| 07 | Inference and export | Ready |
