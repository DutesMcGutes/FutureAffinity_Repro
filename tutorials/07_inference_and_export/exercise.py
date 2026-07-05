from __future__ import annotations

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# By this chapter, every piece (features, embeddings, trunk, diffusion, heads, export) has its
# own chapter and its own standalone exercise. This one is about *integration*: chaining the
# finished library pieces into one inference call, the same shape `predict()` has in
# FutureAffinity_PyTorch's `inference/predict.py` -- sequence(s)(+ligand) in, structure ensemble +
# every head's prediction + a real uncertainty estimate + an exported file out.
from futureaffinity_reproduction.export.mmcif import to_mmcif
from futureaffinity_reproduction.features.builder import build_features
from futureaffinity_reproduction.generator.diffusion import per_atom_uncertainty, run_diffusion_ensemble
from futureaffinity_reproduction.heads.affinity import predict_affinity
from futureaffinity_reproduction.heads.confidence import compute_lddt
from futureaffinity_reproduction.heads.contacts import contact_map
from futureaffinity_reproduction.io.schema import parse_fold_input


def run_futureaffinity(payload: dict, num_ensemble_samples: int = 5, num_steps: int = 8) -> dict:
    """TODO: Chain feature extraction, a diffusion ensemble, and every head into one result dict."""
    fold_input = parse_fold_input(payload)
    features = build_features(fold_input)

    seeds = tuple(range(1, num_ensemble_samples + 1))
    ensemble = run_diffusion_ensemble(features, seeds=seeds, num_steps=num_steps)
    representative = ensemble[0]
    uncertainty = per_atom_uncertainty(ensemble)

    lddt = compute_lddt(representative, representative)
    contacts = contact_map(representative)
    affinity = predict_affinity(features, representative)

    return {
        "features": features,
        "representative_coordinates": representative,
        "ensemble_size": len(ensemble),
        "mean_uncertainty": sum(uncertainty) / len(uncertainty),
        "mean_lddt": sum(lddt) / len(lddt),
        "num_contacts": sum(sum(row) for row in contacts),
        "affinity": affinity,
    }


def main() -> None:
    payload = json.loads((ROOT / "examples" / "aspirin_lysozyme.json").read_text(encoding="utf-8"))
    result = run_futureaffinity(payload)

    print(f"tokens: {len(result['features'].tokens)}")
    print(f"ensemble size: {result['ensemble_size']}")
    print(f"mean structural uncertainty: {result['mean_uncertainty']:.4f}")
    print(f"mean lDDT (self-consistency check): {result['mean_lddt']:.2f}")
    print(f"num predicted contacts: {result['num_contacts']}")
    print(f"predicted affinity: {result['affinity']:.4f}")

    mmcif_text = to_mmcif(result["features"], result["representative_coordinates"])
    output_dir = ROOT / "examples" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "exercise_output.cif"
    output_path.write_text(mmcif_text, encoding="utf-8")
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
