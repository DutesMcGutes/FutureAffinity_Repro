from __future__ import annotations

import math
from typing import Tuple

Vector = Tuple[float, ...]

DIM = 4
NUM_TOKENS = 5


def seed_token_vector(index: int, dim: int = DIM) -> Vector:
    """A stand-in for a real token embedding: distinct, deterministic, tiny."""
    return tuple((index + 1) * 0.1 + component * 0.01 for component in range(dim))


def seed_pair_vector(i: int, j: int, dim: int = DIM) -> Vector:
    """A stand-in for a real pair embedding: closer tokens start more alike."""
    base = 1.0 / (1.0 + abs(i - j))
    return tuple(base + component * 0.01 for component in range(dim))


def _average(*vectors: Vector) -> Vector:
    dim = len(vectors[0])
    return tuple(sum(vector[d] for vector in vectors) / len(vectors) for d in range(dim))


def _dot(a: Vector, b: Vector) -> float:
    return sum(x * y for x, y in zip(a, b))


def token_to_pair_update(
    tokens: tuple[Vector, ...], pairs: tuple[tuple[Vector, ...], ...]
) -> tuple[tuple[Vector, ...], ...]:
    """TODO: Mix each pair vector with the two token vectors it connects."""
    n = len(tokens)
    return tuple(tuple(_average(pairs[i][j], tokens[i], tokens[j]) for j in range(n)) for i in range(n))


def triangle_update(pairs: tuple[tuple[Vector, ...], ...]) -> tuple[tuple[Vector, ...], ...]:
    """TODO: Route pair (i, j) through every third token k, AlphaFold-triangle-style.

    Pair (i, k) and pair (k, j) are multiplied together and averaged over all
    k, then blended back into pair (i, j). This is how a single block lets
    distant tokens influence each other through shared neighbors.
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
    """TODO: The same triangle pattern, but weight each k by real (softmax) attention instead
    of averaging uniformly over every k.

    For a fixed row i, token j attends over token k with weight
    `softmax_k(pair[i,j] . pair[i,k])`, then takes a weighted combination of
    pair[k, j]. This is the essential idea behind real triangle attention --
    tokens the row already agrees with contribute more -- minus the learned
    query/key/value projections and multiple heads a trained version would add.
    """
    n = len(pairs)
    dim = len(pairs[0][0])
    updated = [[tuple()] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            scores = [_dot(pairs[i][j], pairs[i][k]) for k in range(n)]
            max_score = max(scores)
            exp_scores = [math.exp(score - max_score) for score in scores]
            total = sum(exp_scores)
            weights = [value / total for value in exp_scores]

            attended = [0.0] * dim
            for k, weight in enumerate(weights):
                for d in range(dim):
                    attended[d] += weight * pairs[k][j][d]

            updated[i][j] = _average(pairs[i][j], tuple(attended))
    return tuple(tuple(row) for row in updated)


def pair_to_token_update(
    tokens: tuple[Vector, ...], pairs: tuple[tuple[Vector, ...], ...]
) -> tuple[Vector, ...]:
    """TODO: Fold the outgoing pair row for token i back into token i."""
    return tuple(_average(tokens[i], _average(*pairs[i])) for i in range(len(tokens)))


def run_pairformer(
    tokens: tuple[Vector, ...], pairs: tuple[tuple[Vector, ...], ...], num_blocks: int = 2, use_attention: bool = False
) -> tuple[tuple[Vector, ...], tuple[tuple[Vector, ...], ...]]:
    """TODO: Alternate token->pair, triangle, and pair->token updates."""
    triangle_fn = triangle_attention_update if use_attention else triangle_update
    for _ in range(num_blocks):
        pairs = token_to_pair_update(tokens, pairs)
        pairs = triangle_fn(pairs)
        tokens = pair_to_token_update(tokens, pairs)
    return tokens, pairs


def main() -> None:
    tokens = tuple(seed_token_vector(i) for i in range(NUM_TOKENS))
    pairs = tuple(tuple(seed_pair_vector(i, j) for j in range(NUM_TOKENS)) for i in range(NUM_TOKENS))

    averaged_tokens, _ = run_pairformer(tokens, pairs, use_attention=False)
    attended_tokens, _ = run_pairformer(tokens, pairs, use_attention=True)

    print(f"tokens={len(averaged_tokens)} dim={len(averaged_tokens[0])}")
    print(f"token[0] before                = {tokens[0]}")
    print(f"token[0] after (averaging)     = {averaged_tokens[0]}")
    print(f"token[0] after (attention)     = {attended_tokens[0]}")
    print(f"averaging and attention differ: {averaged_tokens[0] != attended_tokens[0]}")


if __name__ == "__main__":
    main()
