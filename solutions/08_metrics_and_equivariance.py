from __future__ import annotations

from pathlib import Path
import json
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from futureaffinity_reproduction.features.builder import build_features
from futureaffinity_reproduction.generator.diffusion import run_diffusion
from futureaffinity_reproduction.generator.tiny import AtomCoordinate
from futureaffinity_reproduction.heads.confidence import compute_lddt
from futureaffinity_reproduction.io.schema import parse_fold_input


def rotate_z(coords: tuple[AtomCoordinate, ...], degrees: float) -> tuple[AtomCoordinate, ...]:
    """Rotate a whole structure about the z-axis -- a rigid motion that changes no internal geometry."""
    angle = math.radians(degrees)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    return tuple(
        AtomCoordinate(
            atom_index=atom.atom_index,
            x=atom.x * cos_a - atom.y * sin_a,
            y=atom.x * sin_a + atom.y * cos_a,
            z=atom.z,
        )
        for atom in coords
    )


def naive_coordinate_mse(a: tuple[AtomCoordinate, ...], b: tuple[AtomCoordinate, ...]) -> float:
    """Mean squared coordinate difference -- with NO alignment. This is the "wrong" metric."""
    total = sum((p.x - q.x) ** 2 + (p.y - q.y) ** 2 + (p.z - q.z) ** 2 for p, q in zip(a, b))
    return total / len(a)


def summarize_metrics_and_equivariance(payload: dict) -> dict:
    """The teaching payoff: a rigid rotation wrecks naive coordinate MSE but leaves lDDT untouched,
    because lDDT is superposition-free -- it scores internal geometry, which the rotation preserved.
    That is exactly why the field uses metrics like lDDT, and why the model gets rotation-robustness
    from centering + augmentation rather than from absolute coordinates (see docs/equivariance.md)."""
    fold_input = parse_fold_input(payload)
    features = build_features(fold_input)
    coords = run_diffusion(features, seed=1, num_steps=6)
    rotated = rotate_z(coords, 90.0)

    return {
        "num_atoms": len(coords),
        "naive_mse_identity": round(naive_coordinate_mse(coords, coords), 4),
        "naive_mse_after_rotation": round(naive_coordinate_mse(coords, rotated), 4),
        "mean_lddt_identity": round(sum(compute_lddt(coords, coords)) / len(coords), 4),
        "mean_lddt_after_rotation": round(sum(compute_lddt(rotated, coords)) / len(coords), 4),
        "naive_mse_changes_under_rotation": naive_coordinate_mse(coords, rotated) > 1.0,
        "lddt_invariant_under_rotation": abs(
            sum(compute_lddt(rotated, coords)) / len(coords) - 100.0
        ) < 1e-6,
    }


def main() -> None:
    payload = json.loads((ROOT / "examples" / "aspirin_lysozyme.json").read_text(encoding="utf-8"))
    print(json.dumps(summarize_metrics_and_equivariance(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
