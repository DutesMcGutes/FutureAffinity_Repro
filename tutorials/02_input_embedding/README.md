# Chapter 02: Input Embedding

Turns the token stream from Chapter 01 into fixed-size vectors: a positional signal, an identity
lookup (one shared vocabulary for amino acids and ligand elements), and a stand-in for what a
pretrained protein-language-model embedding would contribute.

None of this is learned yet -- these are deterministic functions standing in for what an
embedding table and a real PLM would provide, so the rest of the pipeline has real fixed-size
vectors to operate on. The point of this chapter is the *shape* of the input embedding (three
signals summed together), not the specific numbers.

## Student Task

Open `exercise.py` and study the TODO-annotated functions:

1. `sinusoidal_position_encoding` -- the classic alternating sin/cos positional encoding.
2. `vocab_embedding` -- a deterministic per-identity vector, shared across residues and ligand elements.
3. `stand_in_plm_embedding` -- a stand-in for a real ESM2-style embedding (zero for ligand tokens).
4. `token_embedding` -- combine all three into one vector per token.

Then run:

```bash
python tutorials/02_input_embedding/exercise.py
```

To compare against the reference implementation:

```bash
python scripts/generate_test_results.py 02_input_embedding
```
