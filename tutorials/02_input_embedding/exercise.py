from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

EMBED_DIM = 8
AMINO_ACID_VOCAB = tuple(sorted("ACDEFGHIKLMNPQRSTVWY"))
LIGAND_ELEMENT_VOCAB = ("C", "N", "O", "S", "P", "F", "CL", "BR", "I")


@dataclass(frozen=True)
class TutorialToken:
    residue_index: int
    residue_code: str  # amino acid letter, or an element symbol if is_ligand
    is_ligand: bool = False


def sinusoidal_position_encoding(position: int, dim: int = EMBED_DIM) -> tuple[float, ...]:
    """TODO: Alternate sin/cos at geometrically spaced frequencies (the classic Transformer encoding)."""
    values = []
    for index in range(dim):
        exponent = (index - index % 2) / dim
        frequency = 1.0 / (10000.0**exponent)
        angle = position * frequency
        values.append(math.sin(angle) if index % 2 == 0 else math.cos(angle))
    return tuple(values)


def vocab_embedding(token: TutorialToken, dim: int = EMBED_DIM) -> tuple[float, ...]:
    """TODO: Look up a deterministic vector for this token's identity.

    One shared vocabulary for residues and ligand elements (offset apart) --
    the same layout FutureAffinityConfig.vocab_size uses in the real
    implementation for its (learned) embedding table.
    """
    if token.is_ligand:
        vocab_index = len(AMINO_ACID_VOCAB) + LIGAND_ELEMENT_VOCAB.index(token.residue_code) + 1
    else:
        vocab_index = AMINO_ACID_VOCAB.index(token.residue_code) + 1
    return tuple(math.sin(vocab_index * (component + 1) * 0.3) for component in range(dim))


def stand_in_plm_embedding(token: TutorialToken, dim: int = EMBED_DIM) -> tuple[float, ...]:
    """TODO: A stand-in for a pretrained protein-language-model embedding (zero for ligand tokens).

    Real PLM embeddings (e.g. ESM2) carry evolutionary/contextual signal a
    one-hot identity lookup can't. This function only mimics the *shape* of
    that contribution -- see FutureAffinity_PyTorch's data/esm_embeddings.py for
    where a real one would be computed.
    """
    if token.is_ligand:
        return tuple(0.0 for _ in range(dim))
    vocab_index = AMINO_ACID_VOCAB.index(token.residue_code) + 1
    return tuple(
        math.sin(vocab_index * (component + 7) * 0.19 + token.residue_index * 0.05) for component in range(dim)
    )


def token_embedding(token: TutorialToken, dim: int = EMBED_DIM) -> tuple[float, ...]:
    """TODO: Combine position, identity, and PLM stand-in into one token vector."""
    position = sinusoidal_position_encoding(token.residue_index, dim)
    vocab = vocab_embedding(token, dim)
    plm = stand_in_plm_embedding(token, dim)
    return tuple(p + v + m for p, v, m in zip(position, vocab, plm))


def main() -> None:
    tokens = [
        TutorialToken(residue_index=1, residue_code="K", is_ligand=False),
        TutorialToken(residue_index=2, residue_code="V", is_ligand=False),
        TutorialToken(residue_index=1, residue_code="C", is_ligand=True),
        TutorialToken(residue_index=2, residue_code="O", is_ligand=True),
    ]
    embeddings = [token_embedding(token) for token in tokens]
    for token, embedding in zip(tokens, embeddings):
        kind = "ligand" if token.is_ligand else "protein"
        print(f"{kind} token {token.residue_code}: {[round(v, 4) for v in embedding]}")


if __name__ == "__main__":
    main()
