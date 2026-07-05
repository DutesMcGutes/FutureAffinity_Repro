from __future__ import annotations

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from futureaffinity_reproduction.features.builder import build_features
from futureaffinity_reproduction.generator.diffusion import run_diffusion
from futureaffinity_reproduction.heads.affinity import predict_affinity
from futureaffinity_reproduction.heads.confidence import compute_lddt
from futureaffinity_reproduction.heads.contacts import contact_map
from futureaffinity_reproduction.heads.ddg import predict_ddg
from futureaffinity_reproduction.io.schema import parse_fold_input


def summarize_multitask_heads(payload: dict) -> dict:
    fold_input = parse_fold_input(payload)
    features = build_features(fold_input)
    coords = run_diffusion(features, seed=1, num_steps=6)

    lddt = compute_lddt(coords, coords)  # identical structures: every score should be 100
    contacts = contact_map(coords)
    num_contacts = sum(sum(row) for row in contacts)
    affinity = predict_affinity(features, coords)

    wildtype_sequence = fold_input.proteins[0].sequence
    mutated_residue = "A" if wildtype_sequence[0] != "A" else "G"
    mutant_sequence = mutated_residue + wildtype_sequence[1:]
    ddg = predict_ddg(features, coords, mutant_sequence)

    return {
        "mean_lddt_identical_structures": round(sum(lddt) / len(lddt), 4),
        "num_predicted_contacts": num_contacts,
        "contact_density_rounded": round(num_contacts / (len(contacts) ** 2), 4),
        "affinity_rounded": round(affinity, 4),
        "ddg_rounded": round(ddg, 4),
    }


def main() -> None:
    payload = json.loads((ROOT / "examples" / "aspirin_lysozyme.json").read_text(encoding="utf-8"))
    print(json.dumps(summarize_multitask_heads(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
