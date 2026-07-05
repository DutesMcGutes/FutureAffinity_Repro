"""Deterministic demo coordinate generator.

This is not a trained predictor. It gives the project an end-to-end runnable
pipeline while the Pairformer and diffusion modules are developed. Ligand
atoms get their own "chain" offset exactly like a second protein chain would,
so a ligand ends up placed near-but-distinct from the protein instead of
overlapping it.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from futureaffinity_reproduction.features.builder import FeatureBatch


@dataclass(frozen=True)
class AtomCoordinate:
    atom_index: int
    x: float
    y: float
    z: float


def generate_demo_coordinates(features: FeatureBatch, seed: int = 1) -> tuple[AtomCoordinate, ...]:
    coordinates: list[AtomCoordinate] = []
    chain_offsets = _chain_offsets(features)

    for atom in features.atoms:
        angle = atom.residue_index * 1.75 + seed * 0.013
        radius = 2.3
        chain_offset = chain_offsets[atom.chain_id]
        atom_offset = _atom_offset(atom.atom_name)
        x = chain_offset + radius * math.cos(angle) + atom_offset[0]
        y = radius * math.sin(angle) + atom_offset[1]
        z = atom.residue_index * 1.45 + atom_offset[2]
        coordinates.append(AtomCoordinate(atom_index=atom.atom_index, x=x, y=y, z=z))

    return tuple(coordinates)


def _chain_offsets(features: FeatureBatch) -> dict[str, float]:
    chain_ids = []
    for token in features.tokens:
        if token.chain_id not in chain_ids:
            chain_ids.append(token.chain_id)
    return {chain_id: index * 8.0 for index, chain_id in enumerate(chain_ids)}


def _atom_offset(atom_name: str) -> tuple[float, float, float]:
    offsets = {
        "N": (-0.52, 0.12, -0.36),
        "CA": (0.0, 0.0, 0.0),
        "C": (0.58, -0.08, 0.42),
        "O": (0.92, -0.28, 0.72),
    }
    return offsets.get(atom_name, (0.0, 0.0, 0.0))
