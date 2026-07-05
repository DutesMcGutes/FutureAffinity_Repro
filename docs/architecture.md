# Architecture

FutureAffinity Reproduction is organized as a pipeline, extending the AlphaFold3-reproduction pattern
from protein-only folding to protein+ligand co-folding and multi-task prediction:

1. `io`: parse a co-folding JSON (protein sequences and/or a ligand SMILES) into typed entities.
2. `chem`: residue, backbone-atom, and ligand-element vocabularies, plus a bag-of-elements SMILES reader.
3. `features`: convert entities into a single token/atom stream, with an `is_ligand` flag telling
   protein tokens and ligand tokens apart -- everything downstream operates on that one stream.
4. `features.embedding`: deterministic token/pair embeddings (vocabulary lookup + positional
   encoding + a stand-in protein-language-model contribution).
5. `trunk`: a dependency-free Pairformer-style trunk, with *two* triangle-update variants --
   uniform averaging (`triangle_update`) and real softmax attention (`triangle_attention_update`)
   -- so the difference between "blend everything" and "learn what to weight" is visible directly.
6. `generator`: a diffusion-style coordinate sampler, extended with `run_diffusion_ensemble` +
   `per_atom_uncertainty` -- several independent samples instead of one point estimate.
7. `heads`: confidence (a real lDDT calculation), contacts (a real distance-thresholded map), and
   affinity/ΔΔG (toy but real, conformation-dependent physical-distance scores) -- no learned
   weights, but genuine computations, not placeholders.
8. `datasources`: a dependency-free toy docking energy + pose generator, for synthetic-supervision
   pretraining.
9. `export`: a minimal mmCIF writer, with ligand atoms written as `HETATM`.

Every stage chains into the next (see Chapter 07), producing a runnable end-to-end pipeline: input
JSON -> tokens/atoms -> embeddings -> trunk representations -> a structural ensemble -> per-token
confidence + contacts + affinity -> an exported mmCIF file. None of it is trained -- the trunk uses
fixed averaging or an untrained attention formula instead of learned projections, and the diffusion
sampler denoises toward deterministic demo coordinates instead of a learned prediction. They exist
to teach the architecture's data flow and communication pattern (including where a real,
trainable, multi-head version of each piece lives in `FutureAffinity_PyTorch`), not to produce
accurate structures or affinities.
