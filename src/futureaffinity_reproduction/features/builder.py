"""Feature construction for the FutureAffinity reproduction's co-folding vertical slice.

Extends the protein-only AlphaFold3 reproduction pattern with ligands: one
token (and, for this simplified demo, one atom) per ligand heavy atom,
marked with `is_ligand=True` so the rest of the pipeline can tell protein
tokens and ligand tokens apart.
"""

from __future__ import annotations

from dataclasses import dataclass

from futureaffinity_reproduction.chem.residues import BACKBONE_ATOMS, THREE_LETTER_CODES
from futureaffinity_reproduction.io.schema import FoldInput


@dataclass(frozen=True)
class TokenRecord:
    token_index: int
    chain_id: str
    residue_index: int
    residue_name: str
    residue_code: str
    is_ligand: bool = False


@dataclass(frozen=True)
class AtomRecord:
    atom_index: int
    token_index: int
    chain_id: str
    residue_index: int
    residue_name: str
    atom_name: str
    element: str
    is_ligand: bool = False


@dataclass(frozen=True)
class FeatureBatch:
    name: str
    tokens: tuple[TokenRecord, ...]
    atoms: tuple[AtomRecord, ...]
    token_mask: tuple[int, ...]
    atom_mask: tuple[int, ...]


def build_features(fold_input: FoldInput) -> FeatureBatch:
    tokens: list[TokenRecord] = []
    atoms: list[AtomRecord] = []

    for protein in fold_input.proteins:
        for local_index, residue_code in enumerate(protein.sequence, start=1):
            residue_name = THREE_LETTER_CODES[residue_code]
            token = TokenRecord(
                token_index=len(tokens),
                chain_id=protein.chain_id,
                residue_index=local_index,
                residue_name=residue_name,
                residue_code=residue_code,
                is_ligand=False,
            )
            tokens.append(token)

            for atom_name in BACKBONE_ATOMS:
                atoms.append(
                    AtomRecord(
                        atom_index=len(atoms),
                        token_index=token.token_index,
                        chain_id=protein.chain_id,
                        residue_index=local_index,
                        residue_name=residue_name,
                        atom_name=atom_name,
                        element=_element_for_atom(atom_name),
                        is_ligand=False,
                    )
                )

    for ligand in fold_input.ligands:
        for local_index, element in enumerate(ligand.elements, start=1):
            token = TokenRecord(
                token_index=len(tokens),
                chain_id=ligand.chain_id,
                residue_index=local_index,
                residue_name="LIG",
                residue_code=element,
                is_ligand=True,
            )
            tokens.append(token)
            atoms.append(
                AtomRecord(
                    atom_index=len(atoms),
                    token_index=token.token_index,
                    chain_id=ligand.chain_id,
                    residue_index=local_index,
                    residue_name="LIG",
                    atom_name=element,
                    element=element,
                    is_ligand=True,
                )
            )

    return FeatureBatch(
        name=fold_input.name,
        tokens=tuple(tokens),
        atoms=tuple(atoms),
        token_mask=tuple(1 for _ in tokens),
        atom_mask=tuple(1 for _ in atoms),
    )


def _element_for_atom(atom_name: str) -> str:
    if atom_name.startswith("C"):
        return "C"
    if atom_name.startswith("N"):
        return "N"
    if atom_name.startswith("O"):
        return "O"
    return atom_name[0]
