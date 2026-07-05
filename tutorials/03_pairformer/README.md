# Chapter 03: Pairformer

Builds a tiny, dependency-free stand-in for AlphaFold's Pairformer trunk, and then shows the one
change that turns it into something closer to the real thing. It keeps the real architectural
shape:

- a token-to-pair update that blends each pair vector with its two owning tokens
- a triangle update that routes pair (i, j) through every third token k, so evidence can
  propagate across the whole token graph in one block
- a pair-to-token update that folds pair information back into each token

`triangle_update` replaces every learned projection with plain elementwise averaging (as in the
AlphaFold3 reproduction this builds on) -- enough to see the *communication pattern*, not to
reach real representations. `triangle_attention_update` then swaps that uniform averaging for
real (if tiny, single-head) softmax attention: instead of blending every third token k equally,
it weighs each k by how much token j already agrees with it. That's the actual idea behind
AlphaFold's triangle attention -- see FutureAffinity_PyTorch's `model/pairformer.py` for the real,
learned, multi-head version with query/key/value projections.

## Student Task

Open `exercise.py` and study the TODO-annotated sections:

1. Implement the token-to-pair update.
2. Implement the triangle update (uniform averaging over k).
3. Implement the triangle *attention* update (softmax-weighted over k).
4. Implement the pair-to-token update.
5. Chain them into `run_pairformer`, with a flag to pick which triangle variant to use.

Then run:

```bash
python tutorials/03_pairformer/exercise.py
```

To compare against the reference implementation:

```bash
python scripts/generate_test_results.py 03_pairformer
```
