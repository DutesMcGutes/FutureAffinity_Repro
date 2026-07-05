# Limitations

FutureAffinity Reproduction is not currently a scientifically accurate structure or affinity
predictor, and it isn't meant to become one -- that's what `FutureAffinity_PyTorch` is for. This repo
teaches the architecture and data flow.

Specific simplifications, so they're not mistaken for bugs:

- **Nothing is learned.** The trunk uses fixed averaging or an untrained attention formula
  instead of learned projections; the diffusion sampler denoises toward deterministic demo
  coordinates instead of a learned prediction; every head computes a real formula (lDDT, a
  distance-thresholded contact map, a physical-distance affinity/ΔΔG score) but with no trained
  weights behind it.
- **One coordinate generator, atom-level.** Unlike `FutureAffinity_PyTorch` (one coordinate per
  *token*), this reproduction's demo coordinate generator assigns one coordinate per *atom*
  (matching the original AlphaFold3-reproduction pattern it extends), so protein backbone atoms
  (N, CA, C, O) each get their own position while ligand tokens get exactly one atom each.
- **Ligands have no bonds.** The SMILES reader (`chem/residues.py:parse_smiles_elements`) reads
  atom composition only, not connectivity, exactly like the corresponding reader in
  `FutureAffinity_PyTorch`.
- **The toy docking energy is a Lennard-Jones-style formula**, not a real force field -- useful as
  a cheap, repeatable synthetic signal, not as ground truth.
- **ΔΔG assumes single point mutations** and treats each mutated position's contribution as
  independent and additive, which is a real simplification real ΔΔG prediction has to grapple with.

None of this should be used for real structural or affinity conclusions. See
`FutureAffinity_PyTorch/docs/roadmap.md` for what a real-scale, trainable version of these ideas
requires.
