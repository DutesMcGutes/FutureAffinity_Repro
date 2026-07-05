from __future__ import annotations

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from futureaffinity_reproduction.datasources.mock_docking import dock
from futureaffinity_reproduction.features.builder import build_features
from futureaffinity_reproduction.generator.diffusion import run_diffusion
from futureaffinity_reproduction.io.schema import parse_fold_input


def summarize_synthetic_supervision(payload: dict, seed: int = 0, num_candidates: int = 8, num_poses: int = 3) -> dict:
    fold_input = parse_fold_input(payload)
    features = build_features(fold_input)
    coords = run_diffusion(features, seed=1, num_steps=6)

    protein_coords = tuple(coord for atom, coord in zip(features.atoms, coords) if not atom.is_ligand)
    ligand_coords = tuple(coord for atom, coord in zip(features.atoms, coords) if atom.is_ligand)

    poses = dock(protein_coords, ligand_coords, num_poses=num_poses, num_candidates=num_candidates, seed=seed)
    energies = [round(energy, 4) for _, energy in poses]

    return {
        "num_poses_requested": num_poses,
        "num_poses_returned": len(poses),
        "energies_sorted_ascending": energies,
        "best_energy": energies[0] if energies else None,
        "is_sorted_ascending": energies == sorted(energies),
    }


def main() -> None:
    payload = json.loads((ROOT / "examples" / "aspirin_lysozyme.json").read_text(encoding="utf-8"))
    print(json.dumps(summarize_synthetic_supervision(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
