from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
BACKBONE_ATOMS = ("N", "CA", "C", "O")
SMILES_ATOM_PATTERN = re.compile(r"\[[^\]]+\]|Br|Cl|[BCNOFPSIbcnops]")


@dataclass(frozen=True)
class TutorialToken:
    index: int
    chain_id: str
    residue_index: int
    residue_code: str
    is_ligand: bool = False


@dataclass(frozen=True)
class TutorialAtom:
    index: int
    token_index: int
    atom_name: str
    is_ligand: bool = False


def load_payload(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_sequence(sequence: str) -> str:
    """TODO: Normalize a sequence and reject unsupported amino-acid codes."""
    cleaned = "".join(sequence.split()).upper()
    invalid = sorted(set(cleaned) - AMINO_ACIDS)
    if invalid:
        raise ValueError(f"Unsupported amino acid code(s): {', '.join(invalid)}")
    if not cleaned:
        raise ValueError("Protein sequence must not be empty.")
    return cleaned


def parse_smiles_elements(smiles: str) -> list[str]:
    """TODO: Read a SMILES string as a bag of atom elements (no bonds, no stereochemistry).

    This is deliberately not a real SMILES parser -- just enough of one to
    turn a ligand string into a list of atom types for a co-folding token
    stream. See chem/residues.py in the src package for the maintained version.
    """
    elements = []
    for token in SMILES_ATOM_PATTERN.findall(smiles):
        if token.startswith("["):
            match = re.match(r"[A-Za-z]{1,2}", token[1:-1])
            element = match.group(0) if match else "C"
        else:
            element = token
        elements.append(element.upper())
    return elements


def build_tokens(payload: dict) -> list[TutorialToken]:
    """TODO: Create one token per protein residue, and one token per ligand heavy atom."""
    tokens: list[TutorialToken] = []
    for entry in payload["sequences"]:
        if "protein" in entry:
            protein = entry["protein"]
            chain_ids = protein["id"]
            if isinstance(chain_ids, str):
                chain_ids = [chain_ids]
            sequence = validate_sequence(protein["sequence"])
            for chain_id in chain_ids:
                for residue_index, residue_code in enumerate(sequence, start=1):
                    tokens.append(
                        TutorialToken(
                            index=len(tokens),
                            chain_id=chain_id,
                            residue_index=residue_index,
                            residue_code=residue_code,
                            is_ligand=False,
                        )
                    )
        elif "ligand" in entry:
            ligand = entry["ligand"]
            chain_id = ligand["id"] if isinstance(ligand["id"], str) else ligand["id"][0]
            for residue_index, element in enumerate(parse_smiles_elements(ligand["smiles"]), start=1):
                tokens.append(
                    TutorialToken(
                        index=len(tokens),
                        chain_id=chain_id,
                        residue_index=residue_index,
                        residue_code=element,
                        is_ligand=True,
                    )
                )
    return tokens


def build_atoms(tokens: list[TutorialToken]) -> list[TutorialAtom]:
    """TODO: Create demo backbone atoms for protein tokens, one atom for each ligand token."""
    atoms: list[TutorialAtom] = []
    for token in tokens:
        if token.is_ligand:
            atoms.append(TutorialAtom(index=len(atoms), token_index=token.index, atom_name=token.residue_code, is_ligand=True))
        else:
            for atom_name in BACKBONE_ATOMS:
                atoms.append(TutorialAtom(index=len(atoms), token_index=token.index, atom_name=atom_name, is_ligand=False))
    return atoms


def build_pair_hints(tokens: list[TutorialToken]) -> list[list[float]]:
    """TODO: Encode same-chain residue proximity as a pair matrix (cross-chain pairs get 0)."""
    pair_hints: list[list[float]] = []
    for left in tokens:
        row: list[float] = []
        for right in tokens:
            if left.chain_id != right.chain_id:
                row.append(0.0)
            else:
                row.append(1.0 / (1.0 + abs(left.residue_index - right.residue_index)))
        pair_hints.append(row)
    return pair_hints


def main() -> None:
    payload = load_payload(ROOT / "examples" / "aspirin_lysozyme.json")
    tokens = build_tokens(payload)
    atoms = build_atoms(tokens)
    pair_hints = build_pair_hints(tokens)
    num_ligand_tokens = sum(1 for t in tokens if t.is_ligand)
    print(
        f"tokens={len(tokens)} (protein={len(tokens) - num_ligand_tokens}, ligand={num_ligand_tokens}) "
        f"atoms={len(atoms)} pair_shape={len(pair_hints)}x{len(pair_hints[0])}"
    )


if __name__ == "__main__":
    main()
