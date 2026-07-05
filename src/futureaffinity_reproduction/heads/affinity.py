"""A toy but real physical-distance-based affinity score: E(protein, ligand, conformation).

No learned weights -- this is the same "pool the interface, turn it into one
number" shape as FutureAffinity_PyTorch's `heads/affinity.py:AffinityHead`,
minus the learned projections, computed directly from a specific set of
coordinates (so it genuinely depends on *which* conformation you pass in,
not just which residues/atoms are present).
"""

from __future__ import annotations

import math

from futureaffinity_reproduction.features.builder import FeatureBatch
from futureaffinity_reproduction.generator.tiny import AtomCoordinate


def _distance(a: AtomCoordinate, b: AtomCoordinate) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def predict_affinity(features: FeatureBatch, coordinates: tuple[AtomCoordinate, ...]) -> float:
    """Higher = tighter predicted binder (loosely, a pKd-like scale). 0.0 if there's no ligand."""
    coord_by_atom_index = {coord.atom_index: coord for coord in coordinates}
    protein_coords = [coord_by_atom_index[atom.atom_index] for atom in features.atoms if not atom.is_ligand]
    ligand_coords = [coord_by_atom_index[atom.atom_index] for atom in features.atoms if atom.is_ligand]

    if not protein_coords or not ligand_coords:
        return 0.0

    nearest_distances = [min(_distance(ligand_atom, p) for p in protein_coords) for ligand_atom in ligand_coords]
    mean_nearest_distance = sum(nearest_distances) / len(nearest_distances)
    return 10.0 - mean_nearest_distance / 2.0
