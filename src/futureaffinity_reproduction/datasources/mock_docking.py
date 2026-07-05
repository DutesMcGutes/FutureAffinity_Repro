"""A dependency-free toy docking energy + pose generator.

Not a real force field -- a Lennard-Jones + toy-electrostatics energy, same
formula as FutureAffinity_PyTorch's `datasources/mock_docking.py`, just on plain
tuples instead of tensors. Exists to generate cheap, plentiful *synthetic*
supervision for pretraining an affinity head before fine-tuning on scarce
real measurements (the idea Chapter 06 walks through).
"""

from __future__ import annotations

import math
import random

from futureaffinity_reproduction.generator.tiny import AtomCoordinate

_LJ_EPSILON = 0.20
_LJ_SIGMA = 3.5
_MIN_DISTANCE = 0.5


def _distance(a: AtomCoordinate, b: AtomCoordinate) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def toy_pairwise_energy(
    receptor: tuple[AtomCoordinate, ...], ligand: tuple[AtomCoordinate, ...]
) -> float:
    """Lower = a more favorable (less clashing) pose."""
    total = 0.0
    for receptor_atom in receptor:
        for ligand_atom in ligand:
            distance = max(_distance(receptor_atom, ligand_atom), _MIN_DISTANCE)
            total += _LJ_EPSILON * ((_LJ_SIGMA / distance) ** 12 - 2.0 * (_LJ_SIGMA / distance) ** 6)
    return total


def _rotate_and_translate(
    template: tuple[AtomCoordinate, ...], angle_x: float, angle_y: float, angle_z: float, translation: tuple[float, float, float]
) -> tuple[AtomCoordinate, ...]:
    cx, sx = math.cos(angle_x), math.sin(angle_x)
    cy, sy = math.cos(angle_y), math.sin(angle_y)
    cz, sz = math.cos(angle_z), math.sin(angle_z)

    def rotate(x: float, y: float, z: float) -> tuple[float, float, float]:
        # Rx then Ry then Rz, composed by hand (no matrix library needed)
        y, z = y * cx - z * sx, y * sx + z * cx
        x, z = x * cy + z * sy, -x * sy + z * cy
        x, y = x * cz - y * sz, x * sz + y * cz
        return x, y, z

    tx, ty, tz = translation
    rotated = []
    for atom in template:
        x, y, z = rotate(atom.x, atom.y, atom.z)
        rotated.append(AtomCoordinate(atom_index=atom.atom_index, x=x + tx, y=y + ty, z=z + tz))
    return tuple(rotated)


def dock(
    receptor: tuple[AtomCoordinate, ...],
    ligand_template: tuple[AtomCoordinate, ...],
    num_poses: int = 1,
    num_candidates: int = 8,
    seed: int = 0,
) -> list[tuple[tuple[AtomCoordinate, ...], float]]:
    """Generates `num_candidates` random rigid-body poses of the ligand around the receptor's
    centroid and returns the `num_poses` lowest-energy ones, as (coordinates, energy) pairs."""
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
