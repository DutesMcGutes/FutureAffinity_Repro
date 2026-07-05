"""AF3-style JSON input parsing, extended with an optional ligand entry (co-folding, not just folding)."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from futureaffinity_reproduction.chem.residues import parse_smiles_elements, validate_protein_sequence


@dataclass(frozen=True)
class ProteinEntity:
    chain_id: str
    sequence: str


@dataclass(frozen=True)
class LigandEntity:
    chain_id: str
    smiles: str
    elements: tuple[str, ...]


@dataclass(frozen=True)
class FoldInput:
    name: str
    proteins: tuple[ProteinEntity, ...]
    ligands: tuple[LigandEntity, ...]
    model_seeds: tuple[int, ...]
    dialect: str = "futureaffinity"
    version: int = 1

    @property
    def residue_count(self) -> int:
        return sum(len(protein.sequence) for protein in self.proteins)

    @property
    def ligand_atom_count(self) -> int:
        return sum(len(ligand.elements) for ligand in self.ligands)


def load_fold_input(path: str | Path) -> FoldInput:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return parse_fold_input(payload)


def parse_fold_input(payload: dict[str, Any]) -> FoldInput:
    name = str(payload.get("name") or "futureaffinity_reproduction_prediction")
    dialect = str(payload.get("dialect") or "futureaffinity")
    version = int(payload.get("version") or 1)
    model_seeds = tuple(int(seed) for seed in payload.get("modelSeeds", [1]))

    raw_sequences = payload.get("sequences")
    if not isinstance(raw_sequences, list) or not raw_sequences:
        raise ValueError("Input must include a non-empty 'sequences' list.")

    proteins: list[ProteinEntity] = []
    ligands: list[LigandEntity] = []
    for entry in raw_sequences:
        if not isinstance(entry, dict):
            raise ValueError("Each sequence entry must be an object with a 'protein' or 'ligand' key.")

        if "protein" in entry:
            proteins.extend(_parse_protein_entry(entry["protein"]))
        elif "ligand" in entry:
            ligands.append(_parse_ligand_entry(entry["ligand"]))
        else:
            raise ValueError("Only 'protein' and 'ligand' sequence entries are currently supported.")

    return FoldInput(
        name=name,
        proteins=tuple(proteins),
        ligands=tuple(ligands),
        model_seeds=model_seeds,
        dialect=dialect,
        version=version,
    )


def _normalized_chain_ids(raw_chain_ids: Any, entry_kind: str) -> list[str]:
    if isinstance(raw_chain_ids, str):
        return [raw_chain_ids]
    if isinstance(raw_chain_ids, list) and raw_chain_ids:
        return [str(chain_id) for chain_id in raw_chain_ids]
    raise ValueError(f"{entry_kind} entry must include 'id' as a string or non-empty list.")


def _parse_protein_entry(protein: dict[str, Any]) -> list[ProteinEntity]:
    sequence = validate_protein_sequence(str(protein.get("sequence") or ""))
    chain_ids = _normalized_chain_ids(protein.get("id"), "Protein")
    return [ProteinEntity(chain_id=chain_id, sequence=sequence) for chain_id in chain_ids]


def _parse_ligand_entry(ligand: dict[str, Any]) -> LigandEntity:
    smiles = str(ligand.get("smiles") or "")
    if not smiles:
        raise ValueError("Ligand entry must include a non-empty 'smiles' string.")
    chain_ids = _normalized_chain_ids(ligand.get("id"), "Ligand")
    return LigandEntity(chain_id=chain_ids[0], smiles=smiles, elements=parse_smiles_elements(smiles))
