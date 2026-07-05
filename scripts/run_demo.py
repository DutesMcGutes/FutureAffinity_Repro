from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from futureaffinity_reproduction.export.mmcif import to_mmcif
from futureaffinity_reproduction.features.builder import build_features
from futureaffinity_reproduction.generator.diffusion import per_atom_uncertainty, run_diffusion_ensemble
from futureaffinity_reproduction.heads.affinity import predict_affinity
from futureaffinity_reproduction.heads.confidence import compute_lddt
from futureaffinity_reproduction.heads.contacts import contact_map
from futureaffinity_reproduction.io.schema import load_fold_input


def main() -> None:
    input_path = ROOT / "examples" / "aspirin_lysozyme.json"
    output_dir = ROOT / "examples" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "aspirin_lysozyme_demo.cif"

    fold_input = load_fold_input(input_path)
    features = build_features(fold_input)

    ensemble = run_diffusion_ensemble(features, seeds=(1, 2, 3, 4, 5), num_steps=8)
    representative = ensemble[0]
    uncertainty = per_atom_uncertainty(ensemble)
    lddt = compute_lddt(representative, representative)
    contacts = contact_map(representative)
    affinity = predict_affinity(features, representative)

    output_path.write_text(to_mmcif(features, representative), encoding="utf-8")

    print(f"Wrote {output_path}")
    print(f"Protein chains: {len(fold_input.proteins)}, ligands: {len(fold_input.ligands)}")
    print(f"Tokens: {len(features.tokens)}, atoms: {len(features.atoms)}")
    print(f"Ensemble size: {len(ensemble)}, mean structural uncertainty: {sum(uncertainty) / len(uncertainty):.4f}")
    print(f"Mean lDDT (self-consistency check): {sum(lddt) / len(lddt):.2f}")
    print(f"Predicted contacts: {sum(sum(row) for row in contacts)}")
    print(f"Predicted affinity: {affinity:.4f}")
    print(
        "\nReminder: nothing here is trained -- these numbers demonstrate the pipeline runs, "
        "not that they're accurate. See docs/limitations.md."
    )


if __name__ == "__main__":
    main()
