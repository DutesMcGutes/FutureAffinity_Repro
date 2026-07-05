"""Minimal mmCIF writer for demo outputs. Ligand atoms are written as HETATM, matching
real mmCIF/PDB convention, so a co-folded output distinguishes polymer from ligand."""

from __future__ import annotations

from futureaffinity_reproduction.features.builder import FeatureBatch
from futureaffinity_reproduction.generator.tiny import AtomCoordinate


def to_mmcif(features: FeatureBatch, coordinates: tuple[AtomCoordinate, ...]) -> str:
    coordinate_by_atom = {coord.atom_index: coord for coord in coordinates}
    lines = [
        f"data_{_safe_name(features.name)}",
        "#",
        "loop_",
        "_atom_site.group_PDB",
        "_atom_site.id",
        "_atom_site.type_symbol",
        "_atom_site.label_atom_id",
        "_atom_site.label_comp_id",
        "_atom_site.label_asym_id",
        "_atom_site.label_seq_id",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
        "_atom_site.occupancy",
        "_atom_site.B_iso_or_equiv",
        "_atom_site.pdbx_PDB_model_num",
    ]

    for atom in features.atoms:
        coord = coordinate_by_atom[atom.atom_index]
        lines.append(
            " ".join(
                [
                    "HETATM" if atom.is_ligand else "ATOM",
                    str(atom.atom_index + 1),
                    atom.element,
                    atom.atom_name,
                    atom.residue_name,
                    atom.chain_id,
                    str(atom.residue_index),
                    f"{coord.x:.3f}",
                    f"{coord.y:.3f}",
                    f"{coord.z:.3f}",
                    "1.00",
                    "50.00",
                    "1",
                ]
            )
        )

    lines.append("#")
    return "\n".join(lines) + "\n"


def _safe_name(name: str) -> str:
    return "".join(char if char.isalnum() or char in ("_", "-") else "_" for char in name)
