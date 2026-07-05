from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import math

_LDDT_THRESHOLDS = (0.5, 1.0, 2.0, 4.0)


@dataclass(frozen=True)
class Atom:
    index: int
    x: float
    y: float
    z: float


def _distance(a: Atom, b: Atom) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def rotate_z(atoms: Tuple[Atom, ...], degrees: float) -> Tuple[Atom, ...]:
    """TODO: Rotate the whole structure about the z-axis by `degrees` (a rigid motion)."""
    angle = math.radians(degrees)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    return tuple(
        Atom(index=a.index, x=a.x * cos_a - a.y * sin_a, y=a.x * sin_a + a.y * cos_a, z=a.z) for a in atoms
    )


def naive_coordinate_mse(a: Tuple[Atom, ...], b: Tuple[Atom, ...]) -> float:
    """TODO: Mean squared coordinate difference with NO alignment (the metric that gets fooled)."""
    total = sum((p.x - q.x) ** 2 + (p.y - q.y) ** 2 + (p.z - q.z) ** 2 for p, q in zip(a, b))
    return total / len(a)


def compute_lddt(pred: Tuple[Atom, ...], true: Tuple[Atom, ...], radius: float = 15.0) -> Tuple[float, ...]:
    """TODO: Per-atom lDDT -- superposition-free, so it scores internal geometry only.

    Because it only ever looks at *distances between atoms* (never absolute positions), a rigid
    rotation of the whole structure leaves every score unchanged. That's the property to observe.
    """
    n = len(true)
    true_d = [[_distance(true[i], true[j]) for j in range(n)] for i in range(n)]
    pred_d = [[_distance(pred[i], pred[j]) for j in range(n)] for i in range(n)]
    scores = []
    for i in range(n):
        preserved, neighbors = 0.0, 0
        for j in range(n):
            if i == j or true_d[i][j] >= radius:
                continue
            neighbors += 1
            diff = abs(pred_d[i][j] - true_d[i][j])
            preserved += sum(1.0 for t in _LDDT_THRESHOLDS if diff < t) / len(_LDDT_THRESHOLDS)
        scores.append((preserved / neighbors * 100.0) if neighbors else 100.0)
    return tuple(scores)


def main() -> None:
    import random

    rng = random.Random(0)
    structure = tuple(Atom(index=i, x=rng.gauss(0, 5), y=rng.gauss(0, 5), z=rng.gauss(0, 5)) for i in range(12))
    rotated = rotate_z(structure, 90.0)

    print(f"naive MSE, identity           : {naive_coordinate_mse(structure, structure):.4f}")
    print(f"naive MSE, after 90d rotation : {naive_coordinate_mse(structure, rotated):.4f}   <- blows up")
    lddt_same = sum(compute_lddt(structure, structure)) / len(structure)
    lddt_rot = sum(compute_lddt(rotated, structure)) / len(structure)
    print(f"mean lDDT, identity           : {lddt_same:.2f}")
    print(f"mean lDDT, after 90d rotation : {lddt_rot:.2f}   <- unchanged (superposition-free)")


if __name__ == "__main__":
    main()
