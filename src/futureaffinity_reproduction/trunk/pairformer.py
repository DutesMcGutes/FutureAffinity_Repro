"""A tiny, dependency-free stand-in for AlphaFold's Pairformer trunk.

Keeps the real architectural shape -- a token-to-pair update, a triangle
update that routes pair (i, j) through every third token k, and a
pair-to-token update -- with plain elementwise averaging standing in for
every learned projection, so the block runs on nothing but the standard
library. `triangle_attention_update` then swaps the triangle update's
averaging for real (if tiny) softmax attention, to show what changes when
"blend everything together" becomes "learn what to weight."

See FutureAffinity_PyTorch's `model/pairformer.py` for the real, trainable,
multi-head version of both of these ideas.
"""

from __future__ import annotations

import math
from typing import Tuple

Vector = Tuple[float, ...]


def _average(*vectors: Vector) -> Vector:
    dim = len(vectors[0])
    return tuple(sum(vector[d] for vector in vectors) / len(vectors) for d in range(dim))


def _dot(a: Vector, b: Vector) -> float:
    return sum(x * y for x, y in zip(a, b))


def token_to_pair_update(
    tokens: tuple[Vector, ...], pairs: tuple[tuple[Vector, ...], ...]
) -> tuple[tuple[Vector, ...], ...]:
    """Mix each pair vector with the two token vectors it connects."""
    n = len(tokens)
    return tuple(tuple(_average(pairs[i][j], tokens[i], tokens[j]) for j in range(n)) for i in range(n))


def triangle_update(pairs: tuple[tuple[Vector, ...], ...]) -> tuple[tuple[Vector, ...], ...]:
    """Route pair (i, j) through every third token k, AlphaFold-triangle-style.

    Pair (i, k) and pair (k, j) are multiplied together and averaged over all
    k, then blended back into pair (i, j) -- how a single block lets distant
    tokens influence each other through a shared neighbor.
    """
    n = len(pairs)
    dim = len(pairs[0][0])
    updated = [[tuple()] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            mixed = [0.0] * dim
            for k in range(n):
                left, right = pairs[i][k], pairs[k][j]
                for d in range(dim):
                    mixed[d] += left[d] * right[d]
            mixed = tuple(value / n for value in mixed)
            updated[i][j] = _average(pairs[i][j], mixed)
    return tuple(tuple(row) for row in updated)


def triangle_attention_update(pairs: tuple[tuple[Vector, ...], ...]) -> tuple[tuple[Vector, ...], ...]:
    """The same triangle communication pattern as `triangle_update`, but with real attention.

    For a fixed row i, token j attends over token k: instead of averaging
    pair[i, k] and pair[k, j] uniformly over every k (what `triangle_update`
    does), it computes an attention weight `softmax_k(pair[i, j] . pair[i, k])`
    and takes a *weighted* combination of pair[k, j] -- so tokens the row
    already agrees with contribute more. That's the essential idea behind
    `TriangleAttention` in FutureAffinity_PyTorch's real trunk, minus the learned
    query/key/value projections and multiple heads.
    """
    n = len(pairs)
    updated = [[tuple()] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            scores = [_dot(pairs[i][j], pairs[i][k]) for k in range(n)]
            max_score = max(scores)
            exp_scores = [math.exp(score - max_score) for score in scores]
            total = sum(exp_scores)
            weights = [value / total for value in exp_scores]

            dim = len(pairs[0][0])
            attended = [0.0] * dim
            for k, weight in enumerate(weights):
                for d in range(dim):
                    attended[d] += weight * pairs[k][j][d]

            updated[i][j] = _average(pairs[i][j], tuple(attended))
    return tuple(tuple(row) for row in updated)


def pair_to_token_update(
    tokens: tuple[Vector, ...], pairs: tuple[tuple[Vector, ...], ...]
) -> tuple[Vector, ...]:
    """Fold the outgoing pair row for token i back into token i."""
    return tuple(_average(tokens[i], _average(*pairs[i])) for i in range(len(tokens)))


def run_pairformer(
    tokens: tuple[Vector, ...], pairs: tuple[tuple[Vector, ...], ...], num_blocks: int = 2
) -> tuple[tuple[Vector, ...], tuple[tuple[Vector, ...], ...]]:
    """Alternate token->pair, triangle, and pair->token updates."""
    for _ in range(num_blocks):
        pairs = token_to_pair_update(tokens, pairs)
        pairs = triangle_update(pairs)
        tokens = pair_to_token_update(tokens, pairs)
    return tokens, pairs


def run_pairformer_with_attention(
    tokens: tuple[Vector, ...], pairs: tuple[tuple[Vector, ...], ...], num_blocks: int = 2
) -> tuple[tuple[Vector, ...], tuple[tuple[Vector, ...], ...]]:
    """Same as `run_pairformer`, but using `triangle_attention_update` instead of `triangle_update`."""
    for _ in range(num_blocks):
        pairs = token_to_pair_update(tokens, pairs)
        pairs = triangle_attention_update(pairs)
        tokens = pair_to_token_update(tokens, pairs)
    return tokens, pairs
