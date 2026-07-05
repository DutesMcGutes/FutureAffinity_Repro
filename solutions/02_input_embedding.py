from __future__ import annotations

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from futureaffinity_reproduction.features.builder import build_features
from futureaffinity_reproduction.features.embedding import build_embeddings
from futureaffinity_reproduction.io.schema import parse_fold_input


def summarize_input_embedding(payload: dict) -> dict:
    fold_input = parse_fold_input(payload)
    features = build_features(fold_input)
    embeddings = build_embeddings(features)

    return {
        "num_tokens": len(embeddings.token_embeddings),
        "embed_dim": len(embeddings.token_embeddings[0]),
        "first_protein_token_embedding_rounded": [round(v, 4) for v in embeddings.token_embeddings[0]],
        "last_ligand_token_embedding_rounded": [round(v, 4) for v in embeddings.token_embeddings[-1]],
        "pair_embedding_shape": [len(embeddings.pair_embeddings), len(embeddings.pair_embeddings[0])],
    }


def main() -> None:
    payload = json.loads((ROOT / "examples" / "aspirin_lysozyme.json").read_text(encoding="utf-8"))
    print(json.dumps(summarize_input_embedding(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
