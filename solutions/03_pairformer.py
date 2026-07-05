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
from futureaffinity_reproduction.trunk.pairformer import run_pairformer, run_pairformer_with_attention


def summarize_pairformer(payload: dict, num_blocks: int = 2) -> dict:
    fold_input = parse_fold_input(payload)
    features = build_features(fold_input)
    embeddings = build_embeddings(features)

    averaged_tokens, _ = run_pairformer(embeddings.token_embeddings, embeddings.pair_embeddings, num_blocks=num_blocks)
    attended_tokens, _ = run_pairformer_with_attention(
        embeddings.token_embeddings, embeddings.pair_embeddings, num_blocks=num_blocks
    )

    return {
        "num_tokens": len(averaged_tokens),
        "token_dim": len(averaged_tokens[0]),
        "num_blocks": num_blocks,
        "averaging_first_token_rounded": [round(v, 4) for v in averaged_tokens[0]],
        "attention_first_token_rounded": [round(v, 4) for v in attended_tokens[0]],
        "averaging_and_attention_differ": averaged_tokens[0] != attended_tokens[0],
    }


def main() -> None:
    payload = json.loads((ROOT / "examples" / "aspirin_lysozyme.json").read_text(encoding="utf-8"))
    print(json.dumps(summarize_pairformer(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
