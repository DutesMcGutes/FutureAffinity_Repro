from __future__ import annotations

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from futureaffinity_reproduction.features.builder import build_features
from futureaffinity_reproduction.generator.diffusion import per_atom_uncertainty, run_diffusion, run_diffusion_ensemble
from futureaffinity_reproduction.io.schema import parse_fold_input


def summarize_diffusion(payload: dict, seed: int = 1, num_steps: int = 6) -> dict:
    fold_input = parse_fold_input(payload)
    features = build_features(fold_input)

    single = run_diffusion(features, seed=seed, num_steps=num_steps)
    ensemble = run_diffusion_ensemble(features, seeds=(1, 2, 3, 4, 5), num_steps=num_steps)
    uncertainty = per_atom_uncertainty(ensemble)

    return {
        "num_atoms": len(single),
        "num_ensemble_samples": len(ensemble),
        "first_atom_single_rounded": [round(single[0].x, 4), round(single[0].y, 4), round(single[0].z, 4)],
        "mean_uncertainty_rounded": round(sum(uncertainty) / len(uncertainty), 4),
        "max_uncertainty_rounded": round(max(uncertainty), 4),
    }


def main() -> None:
    payload = json.loads((ROOT / "examples" / "aspirin_lysozyme.json").read_text(encoding="utf-8"))
    print(json.dumps(summarize_diffusion(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
