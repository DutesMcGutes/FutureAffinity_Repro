from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import math

Vector = Tuple[float, ...]
_DEFAULT_LDDT_THRESHOLDS = (0.5, 1.0, 2.0, 4.0)


@dataclass(frozen=True)
class Atom:
    index: int
    x: float
    y: float
    z: float
    is_ligand: bool = False


def _distance(a: Atom, b: Atom) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def compute_lddt(
    predicted: Tuple[Atom, ...], true: Tuple[Atom, ...], radius: float = 15.0, thresholds: Tuple[float, ...] = _DEFAULT_LDDT_THRESHOLDS
) -> Tuple[float, ...]:
    """TODO: Per-atom lDDT (0-100) -- the fraction of nearby-pair distances preserved within tolerance.

    This is a *real* metric (the same one AlphaFold's pLDDT is trained to
    predict), computed directly from two sets of coordinates -- not a
    learned stand-in. If `predicted == true`, every score should be 100.
    """
    n = len(true)
    true_distances = [[_distance(true[i], true[j]) for j in range(n)] for i in range(n)]
    predicted_distances = [[_distance(predicted[i], predicted[j]) for j in range(n)] for i in range(n)]

    scores = []
    for i in range(n):
        preserved_total, neighbor_count = 0.0, 0
        for j in range(n):
            if i == j or true_distances[i][j] >= radius:
                continue
            neighbor_count += 1
            diff = abs(predicted_distances[i][j] - true_distances[i][j])
            preserved_total += sum(1.0 for threshold in thresholds if diff < threshold) / len(thresholds)
        scores.append((preserved_total / neighbor_count * 100.0) if neighbor_count else 100.0)
    return tuple(scores)


def contact_map(atoms: Tuple[Atom, ...], threshold: float = 8.0) -> Tuple[Tuple[int, ...], ...]:
    """TODO: A real (distance-thresholded) contact map."""
    n = len(atoms)
    return tuple(tuple(1 if _distance(atoms[i], atoms[j]) < threshold else 0 for j in range(n)) for i in range(n))


def predict_affinity(atoms: Tuple[Atom, ...]) -> float:
    """TODO: A toy E(protein, ligand, conformation): higher = tighter predicted binder.

    Deliberately depends on the *coordinates* passed in, not just which
    atoms are present -- see FutureAffinity_PyTorch's `heads/affinity.py` for
    why that matters (idea: learn an energy over conformations, not a
    lookup keyed on identity alone).
    """
    protein_atoms = [atom for atom in atoms if not atom.is_ligand]
    ligand_atoms = [atom for atom in atoms if atom.is_ligand]
    if not protein_atoms or not ligand_atoms:
        return 0.0
    nearest_distances = [min(_distance(ligand_atom, p) for p in protein_atoms) for ligand_atom in ligand_atoms]
    return 10.0 - (sum(nearest_distances) / len(nearest_distances)) / 2.0


def main() -> None:
    # a tiny toy "complex": 4 protein atoms in a line, 2 ligand atoms nearby
    protein = tuple(Atom(index=i, x=float(i) * 3.8, y=0.0, z=0.0, is_ligand=False) for i in range(4))
    ligand = (
        Atom(index=4, x=11.0, y=1.0, z=0.0, is_ligand=True),
        Atom(index=5, x=11.5, y=-1.0, z=0.0, is_ligand=True),
    )
    complex_atoms = protein + ligand

    lddt = compute_lddt(complex_atoms, complex_atoms)
    print(f"mean lDDT (identical structures, should be 100): {sum(lddt) / len(lddt):.2f}")

    contacts = contact_map(complex_atoms)
    print(f"contact map: {contacts[0]}")

    affinity = predict_affinity(complex_atoms)
    print(f"predicted affinity: {affinity:.4f}")


if __name__ == "__main__":
    main()
