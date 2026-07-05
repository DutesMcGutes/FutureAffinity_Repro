# Roadmap

This repo's scope is deliberately fixed: teach the architecture and data flow of a co-folding +
multi-task foundation model using plain Python, no trained weights, no external data. It isn't
meant to grow into a trainable system -- that's what `FutureAffinity_PyTorch` is for.

Reasonable extensions that stay within that scope, roughly in order of how much they'd add:

1. **A confidence-calibration chapter**: bin the real lDDT calculation from Chapter 05 into the
   discrete bins a real pLDDT head would predict, and show what cross-entropy training against
   those bins would optimize (still without training anything).
2. **A templates/MSA-hint chapter**: a toy "evolutionary" pair bias (e.g. co-occurrence counts
   from a handful of homologous toy sequences) feeding into the pair representation, as a
   stand-in for what a real MSA-based prior would contribute.
3. **A second worked example** beyond aspirin + lysozyme (e.g. a small peptide ligand, or a
   two-chain protein-protein complex) to make clear the pipeline isn't overfit to one input shape.
4. **An `ml` extra that's actually used**: `pyproject.toml` already declares an optional `torch`
   dependency; nothing currently uses it. A natural next step would be a bonus chapter that swaps
   one hand-written stand-in (e.g. `vocab_embedding`) for a real, tiny, trainable `nn.Embedding`,
   as a bridge to `FutureAffinity_PyTorch`.

None of the above is scheduled.
