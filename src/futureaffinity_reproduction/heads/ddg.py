"""A toy DeltaDeltaG estimate: one additive contribution per mutated protein position,
weighted by how close that position is to the ligand pocket in a given conformation.

Mirrors the shape of FutureAffinity_PyTorch's `heads/ddg.py:DDGHead` (only *changed* positions
contribute; contributions are pocket-proximity-weighted) without any learned weights --
the "direction" of each contribution here comes from a deterministic hash of the two residue
codes, not a trained preference.
"""

from __future__ import annotations

import math

from futureaffinity_reproduction.features.builder import FeatureBatch
from futureaffinity_reproduction.generator.tiny import AtomCoordinate

_FAR_FROM_POCKET_DISTANCE = 20.0


def _distance(a: AtomCoordinate, b: AtomCoordinate) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def predict_ddg(features: FeatureBatch, coordinates: tuple[AtomCoordinate, ...], mutant_sequence: str) -> float:
    protein_tokens = [token for token in features.tokens if not token.is_ligand]
    if len(mutant_sequence) != len(protein_tokens):
        raise ValueError(
            f"mutant_sequence has {len(mutant_sequence)} residues, expected {len(protein_tokens)} "
            "(one per protein token; point mutations only)."
        )

    coord_by_atom_index = {coord.atom_index: coord for coord in coordinates}
    ca_by_token_index = {
        atom.token_index: coord_by_atom_index[atom.atom_index]
        for atom in features.atoms
        if not atom.is_ligand and atom.atom_name == "CA"
    }
    ligand_coords = [coord_by_atom_index[atom.atom_index] for atom in features.atoms if atom.is_ligand]

    total = 0.0
    for token, mutant_code in zip(protein_tokens, mutant_sequence.upper()):
        if token.residue_code == mutant_code:
            continue

        token_coord = ca_by_token_index[token.token_index]
        nearest_to_pocket = (
            min(_distance(token_coord, ligand_atom) for ligand_atom in ligand_coords)
            if ligand_coords
            else _FAR_FROM_POCKET_DISTANCE
        )
        proximity_weight = 1.0 / (1.0 + nearest_to_pocket / 5.0)

        # deterministic (not random) +/- direction from the two residue codes, so re-running
        # this function on the same mutation always gives the same answer
        direction = 1.0 if (ord(mutant_code) + ord(token.residue_code)) % 2 == 0 else -1.0
        total += direction * proximity_weight * 2.0

    return total
