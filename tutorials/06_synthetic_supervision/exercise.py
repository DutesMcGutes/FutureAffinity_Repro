from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import math
import random

_LJ_EPSILON = 0.20
_LJ_SIGMA = 3.5
_MIN_DISTANCE = 0.5


@dataclass(frozen=True)
class Atom:
    index: int
    x: float
    y: float
    z: float


def _distance(a: Atom, b: Atom) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def toy_pairwise_energy(receptor: Tuple[Atom, ...], ligand: Tuple[Atom, ...]) -> float:
    """TODO: A Lennard-Jones-style toy energy: lower = a more favorable (less clashing) pose.

    Not a real force field -- just enough physics-shaped structure (a
    repulsive wall at short range, an attractive well at moderate range) to
    generate a *plausible*, cheap, and infinitely repeatable energy signal.
    """
    total = 0.0
    for receptor_atom in receptor:
        for ligand_atom in ligand:
            distance = max(_distance(receptor_atom, ligand_atom), _MIN_DISTANCE)
            total += _LJ_EPSILON * ((_LJ_SIGMA / distance) ** 12 - 2.0 * (_LJ_SIGMA / distance) ** 6)
    return total


def _rotate_and_translate(
    template: Tuple[Atom, ...], angle_x: float, angle_y: float, angle_z: float, translation: Tuple[float, float, float]
) -> Tuple[Atom, ...]:
    cx, sx, cy, sy, cz, sz = math.cos(angle_x), math.sin(angle_x), math.cos(angle_y), math.sin(angle_y), math.cos(angle_z), math.sin(angle_z)

    def rotate(x: float, y: float, z: float) -> tuple[float, float, float]:
        y, z = y * cx - z * sx, y * sx + z * cx
        x, z = x * cy + z * sy, -x * sy + z * cy
        x, y = x * cz - y * sz, x * sz + y * cz
        return x, y, z

    tx, ty, tz = translation
    rotated = []
    for atom in template:
        x, y, z = rotate(atom.x, atom.y, atom.z)
        rotated.append(Atom(index=atom.index, x=x + tx, y=y + ty, z=z + tz))
    return tuple(rotated)


def dock(
    receptor: Tuple[Atom, ...], ligand_template: Tuple[Atom, ...], num_poses: int = 1, num_candidates: int = 8, seed: int = 0
) -> list[tuple[Tuple[Atom, ...], float]]:
    """TODO: Generate `num_candidates` random rigid-body poses around the receptor's centroid,
    and return the `num_poses` lowest-energy ones.

    This is the "generate billions of noisy labels" idea in miniature: every
    call produces a plausible pose + energy pair, cheaply and at whatever
    scale you want, to pretrain an affinity head before ever touching scarce
    real measurements. See FutureAffinity_PyTorch's `datasources/mock_docking.py`
    for the tensorized version, and `vina_adapter.py`/`openmm_adapter.py` for
    where *real* physics would plug into the same interface.
    """
    rng = random.Random(seed)
    centroid_x = sum(atom.x for atom in receptor) / len(receptor)
    centroid_y = sum(atom.y for atom in receptor) / len(receptor)
    centroid_z = sum(atom.z for atom in receptor) / len(receptor)

    candidates = []
    for _ in range(num_candidates):
        angles = (rng.uniform(0, 2 * math.pi) for _ in range(3))
        translation = (
            centroid_x + rng.gauss(0.0, 2.0),
            centroid_y + rng.gauss(0.0, 2.0),
            centroid_z + rng.gauss(0.0, 2.0),
        )
        pose = _rotate_and_translate(ligand_template, *angles, translation)
        candidates.append((pose, toy_pairwise_energy(receptor, pose)))

    candidates.sort(key=lambda candidate: candidate[1])
    return candidates[:num_poses]


def main() -> None:
    receptor = tuple(Atom(index=i, x=float(i) * 3.8, y=0.0, z=0.0) for i in range(6))
    ligand_template = (Atom(index=0, x=0.0, y=0.0, z=0.0), Atom(index=1, x=1.5, y=0.0, z=0.0), Atom(index=2, x=0.0, y=1.5, z=0.0))

    poses = dock(receptor, ligand_template, num_poses=3, num_candidates=12, seed=0)
    for rank, (_, energy) in enumerate(poses, start=1):
        print(f"pose {rank}: energy={energy:.4f}")
    print(f"poses sorted ascending: {[round(e, 4) for _, e in poses] == sorted(round(e, 4) for _, e in poses)}")


if __name__ == "__main__":
    main()
