from __future__ import annotations

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from futureaffinity_reproduction.features.builder import build_features
from futureaffinity_reproduction.io.schema import parse_fold_input


def summarize_feature_extraction(payload: dict) -> dict:
    fold_input = parse_fold_input(payload)
    features = build_features(fold_input)

    protein_tokens = [token for token in features.tokens if not token.is_ligand]
    ligand_tokens = [token for token in features.tokens if token.is_ligand]

    return {
        "num_tokens": len(features.tokens),
        "num_atoms": len(features.atoms),
        "num_protein_tokens": len(protein_tokens),
        "num_ligand_tokens": len(ligand_tokens),
        "first_protein_token": {
            "residue_name": protein_tokens[0].residue_name,
            "chain_id": protein_tokens[0].chain_id,
        },
        "first_ligand_token": (
            {"residue_name": ligand_tokens[0].residue_name, "element": ligand_tokens[0].residue_code}
            if ligand_tokens
            else None
        ),
    }


def main() -> None:
    payload = json.loads((ROOT / "examples" / "aspirin_lysozyme.json").read_text(encoding="utf-8"))
    print(json.dumps(summarize_feature_extraction(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
