"""Small residue, atom, and ligand-element vocabulary for the demo pipeline."""

from __future__ import annotations

import re

STANDARD_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

BACKBONE_ATOMS = ("N", "CA", "C", "O")

THREE_LETTER_CODES = {
    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE", "G": "GLY", "H": "HIS",
    "I": "ILE", "K": "LYS", "L": "LEU", "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN",
    "R": "ARG", "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR",
}

# Ligand heavy-atom elements this demo pipeline recognizes (co-folding's other "chain type").
LIGAND_ELEMENTS = ("C", "N", "O", "S", "P", "F", "CL", "BR", "I")

_SMILES_ATOM_PATTERN = re.compile(r"\[[^\]]+\]|Br|Cl|[BCNOFPSIbcnops]")


def validate_protein_sequence(sequence: str) -> str:
    cleaned = "".join(sequence.split()).upper()
    invalid = sorted(set(cleaned) - STANDARD_AMINO_ACIDS)
    if invalid:
        raise ValueError(f"Unsupported amino acid code(s): {', '.join(invalid)}")
    if not cleaned:
        raise ValueError("Protein sequence must not be empty.")
    return cleaned


def parse_smiles_elements(smiles: str) -> tuple[str, ...]:
    """A "bag of elements" SMILES reader: no bonds, no stereochemistry, no aromaticity beyond
    upper-casing -- just enough to turn a ligand SMILES string into a list of atom types, the
    same way `chem.py` in the FutureAffinity_PyTorch implementation does it (see that repo's
    docstring for why this is a real-but-simplified reading of SMILES, not a full parser).
    """
    elements = []
    for token in _SMILES_ATOM_PATTERN.findall(smiles):
        if token.startswith("["):
            match = re.match(r"[A-Za-z]{1,2}", token[1:-1])
            element = match.group(0) if match else "C"
        else:
            element = token
        elements.append(element.upper())
    if not elements:
        raise ValueError(f"Could not read any atoms from SMILES {smiles!r}.")
    return tuple(elements)
