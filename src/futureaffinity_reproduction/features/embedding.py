"""Deterministic token and pair embeddings for the input embedding stage.

There are no learned weights yet. The vocabulary, positional, and stand-in
protein-language-model signals below stand in for what an embedding table,
a positional encoding, and a pretrained PLM would provide, so the rest of
the pipeline (trunk, generator, heads) has real fixed-size vectors to
operate on instead of raw residue/element codes.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from futureaffinity_reproduction.chem.residues import LIGAND_ELEMENTS, STANDARD_AMINO_ACIDS
from futureaffinity_reproduction.features.builder import FeatureBatch, TokenRecord

EMBED_DIM = 8

AMINO_ACID_VOCAB = tuple(sorted(STANDARD_AMINO_ACIDS))
LIGAND_ELEMENT_VOCAB = LIGAND_ELEMENTS


@dataclass(frozen=True)
class EmbeddingBatch:
    token_embeddings: tuple[tuple[float, ...], ...]
    pair_embeddings: tuple[tuple[tuple[float, ...], ...], ...]


def sinusoidal_position_encoding(position: int, dim: int) -> tuple[float, ...]:
    values: list[float] = []
    for index in range(dim):
        exponent = (index - index % 2) / dim
        frequency = 1.0 / (10000.0**exponent)
        angle = position * frequency
        values.append(math.sin(angle) if index % 2 == 0 else math.cos(angle))
    return tuple(values)


def vocab_embedding(token: TokenRecord, dim: int) -> tuple[float, ...]:
    """One embedding table shared by residues and ligand elements (offset apart), same idea
    FutureAffinity_PyTorch's `FutureAffinityConfig.vocab_size` uses for its real embedding table."""
    if token.is_ligand:
        vocab_index = len(AMINO_ACID_VOCAB) + LIGAND_ELEMENT_VOCAB.index(token.residue_code) + 1
    else:
        vocab_index = AMINO_ACID_VOCAB.index(token.residue_code) + 1
    return tuple(math.sin(vocab_index * (component + 1) * 0.3) for component in range(dim))


def stand_in_plm_embedding(token: TokenRecord, dim: int) -> tuple[float, ...]:
    """A stand-in for a pretrained protein-language-model embedding (e.g. ESM2).

    Real PLM embeddings carry evolutionary/contextual signal beyond one-hot
    identity; this deterministic function only mimics the *shape* of that
    contribution (a per-residue vector, zero for ligand tokens, which have
    no PLM of their own) -- see FutureAffinity_PyTorch's `data/esm_embeddings.py`
    for where a real ESM2 embedding would actually be computed and plugged in.
    """
    if token.is_ligand:
        return tuple(0.0 for _ in range(dim))
    vocab_index = AMINO_ACID_VOCAB.index(token.residue_code) + 1
    return tuple(
        math.sin(vocab_index * (component + 7) * 0.19 + token.residue_index * 0.05) for component in range(dim)
    )


def token_embedding(token: TokenRecord, dim: int = EMBED_DIM) -> tuple[float, ...]:
    position = sinusoidal_position_encoding(token.residue_index, dim)
    vocab = vocab_embedding(token, dim)
    plm = stand_in_plm_embedding(token, dim)
    return tuple(p + v + m for p, v, m in zip(position, vocab, plm))


def pair_embedding(left: TokenRecord, right: TokenRecord, dim: int = EMBED_DIM) -> tuple[float, ...]:
    same_chain = 1.0 if left.chain_id == right.chain_id else 0.0
    offset = left.residue_index - right.residue_index
    relative_position = sinusoidal_position_encoding(offset, dim)
    distance_hint = 1.0 / (1.0 + abs(offset))
    return tuple(same_chain * distance_hint * component for component in relative_position)


def build_embeddings(features: FeatureBatch, dim: int = EMBED_DIM) -> EmbeddingBatch:
    token_embeddings = tuple(token_embedding(token, dim) for token in features.tokens)
    pair_embeddings = tuple(
        tuple(pair_embedding(left, right, dim) for right in features.tokens)
        for left in features.tokens
    )
    return EmbeddingBatch(token_embeddings=token_embeddings, pair_embeddings=pair_embeddings)
