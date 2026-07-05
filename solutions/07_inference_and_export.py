from __future__ import annotations

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from futureaffinity_reproduction.features.builder import build_features
from futureaffinity_reproduction.generator.diffusion import per_atom_uncertainty, run_diffusion_ensemble
from futureaffinity_reproduction.heads.affinity import predict_affinity
from futureaffinity_reproduction.heads.confidence import compute_lddt
from futureaffinity_reproduction.heads.contacts import contact_map
from futureaffinity_reproduction.io.schema import parse_fold_input
from futureaffinity_reproduction.export.mmcif import to_mmcif


def summarize_inference_and_export(payload: dict) -> dict:
    """The full pipeline: parse -> features -> diffusion ensemble -> every head -> mmCIF.

    Uses the first ensemble member as "the" exported structure and reports the
    ensemble's per-atom uncertainty alongside it -- the same "one structure to
    look at, plus a real uncertainty estimate" shape a real prediction would have.
    """
    fold_input = parse_fold_input(payload)
    features = build_features(fold_input)

    ensemble = run_diffusion_ensemble(features, seeds=(1, 2, 3), num_steps=6)
    representative = ensemble[0]
    uncertainty = per_atom_uncertainty(ensemble)

    lddt = compute_lddt(representative, representative)
    contacts = contact_map(representative)
    affinity = predict_affinity(features, representative)
    mmcif_text = to_mmcif(features, representative)

    return {
        "num_tokens": len(features.tokens),
        "num_atoms": len(representative),
        "num_ensemble_samples": len(ensemble),
        "mean_uncertainty_rounded": round(sum(uncertainty) / len(uncertainty), 4),
        "mean_lddt_rounded": round(sum(lddt) / len(lddt), 4),
        "num_contacts": sum(sum(row) for row in contacts),
        "affinity_rounded": round(affinity, 4),
        "mmcif_line_count": len(mmcif_text.splitlines()),
        "mmcif_contains_hetatm": any(line.startswith("HETATM") for line in mmcif_text.splitlines()),
    }


def main() -> None:
    payload = json.loads((ROOT / "examples" / "aspirin_lysozyme.json").read_text(encoding="utf-8"))
    summary = summarize_inference_and_export(payload)
    print(json.dumps(summary, indent=2, sort_keys=True))

    fold_input = parse_fold_input(payload)
    features = build_features(fold_input)
    ensemble = run_diffusion_ensemble(features, seeds=(1, 2, 3), num_steps=6)
    mmcif_text = to_mmcif(features, ensemble[0])

    output_dir = ROOT / "examples" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{fold_input.name}.cif"
    output_path.write_text(mmcif_text, encoding="utf-8")
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
