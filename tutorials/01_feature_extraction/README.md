# Chapter 01: Feature Extraction

This chapter turns an AF3-style co-folding JSON (protein sequences *and* an optional ligand
SMILES) into a flat stream of tokens and atoms -- the same first step every co-folding model
takes, whether it's this reproduction or the real FutureAffinity_PyTorch implementation.

The key idea carried through the rest of this tutorial: **a ligand isn't a special case bolted
onto a protein folding model, it's just another kind of token.** One token per protein residue,
one token per ligand heavy atom, both living in the same token stream, distinguished only by an
`is_ligand` flag. Everything downstream (embeddings, the trunk, diffusion, every head) operates on
that stream without needing to know which parts came from a protein and which came from a ligand.

## Student Task

Open `exercise.py` and study the TODO-annotated functions:

1. `validate_sequence` -- normalize and validate a protein sequence.
2. `parse_smiles_elements` -- read a ligand SMILES string as a bag of atom elements.
3. `build_tokens` -- create one token per protein residue and one per ligand atom.
4. `build_atoms` -- create demo backbone atoms for protein tokens, one atom for ligand tokens.
5. `build_pair_hints` -- encode same-chain proximity as a pair matrix.

Then run:

```bash
python tutorials/01_feature_extraction/exercise.py
```

To compare against the reference implementation:

```bash
python scripts/generate_test_results.py 01_feature_extraction
```
