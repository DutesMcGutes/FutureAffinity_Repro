# Sources and Inspiration

- Google DeepMind's AlphaFold3 inference pipeline: https://github.com/google-deepmind/alphafold3
- Jumper et al. (2021), "Highly accurate protein structure prediction with AlphaFold" (Nature) --
  the triangular multiplicative update and triangular attention algorithms this repo's
  `trunk/pairformer.py` reproduces the communication pattern of, at toy scale.
- Abramson et al. (2024), "Accurate structure prediction of biomolecular interactions with
  AlphaFold3" (Nature) -- the co-folding (protein + ligand, one token per polymer residue / one
  token per ligand heavy atom) framing this repo extends the original protein-only reproduction to.
- Karras et al. (2022), "Elucidating the Design Space of Diffusion-Based Generative Models" -- the
  general shape (noise schedule, forward noise, reverse denoise) `generator/diffusion.py`'s toy
  sampler follows, at a much smaller scale than the real EDM-style module in `FutureAffinity_PyTorch`.
- PDBbind, BindingDB, ESM2 -- the real data sources this reproduction deliberately doesn't use (see
  docs/data-and-weights.md), but which motivate the multi-task/synthetic-supervision framing in
  Chapters 05-06 and are used for real in `FutureAffinity_PyTorch`.
