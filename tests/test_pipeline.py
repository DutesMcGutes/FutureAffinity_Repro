from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from futureaffinity_reproduction.datasources.mock_docking import dock, toy_pairwise_energy
from futureaffinity_reproduction.export.mmcif import to_mmcif
from futureaffinity_reproduction.features.builder import build_features
from futureaffinity_reproduction.features.embedding import EMBED_DIM, build_embeddings
from futureaffinity_reproduction.generator.diffusion import per_atom_uncertainty, run_diffusion, run_diffusion_ensemble
from futureaffinity_reproduction.generator.tiny import generate_demo_coordinates
from futureaffinity_reproduction.heads.affinity import predict_affinity
from futureaffinity_reproduction.heads.confidence import compute_lddt
from futureaffinity_reproduction.heads.contacts import contact_map
from futureaffinity_reproduction.heads.ddg import predict_ddg
from futureaffinity_reproduction.io.schema import parse_fold_input
from futureaffinity_reproduction.trunk.pairformer import run_pairformer, run_pairformer_with_attention

COFOLD_PAYLOAD = {
    "sequences": [
        {"protein": {"id": "A", "sequence": "ACDE"}},
        {"ligand": {"id": "L", "smiles": "CCO"}},
    ]
}


class PipelineTests(unittest.TestCase):
    def test_parse_protein_only_input(self) -> None:
        fold_input = parse_fold_input(
            {
                "name": "small",
                "sequences": [{"protein": {"id": "A", "sequence": "ACDE"}}],
                "modelSeeds": [7],
                "dialect": "futureaffinity",
                "version": 1,
            }
        )
        self.assertEqual(fold_input.name, "small")
        self.assertEqual(fold_input.residue_count, 4)
        self.assertEqual(fold_input.model_seeds, (7,))
        self.assertEqual(fold_input.ligands, ())

    def test_parse_cofold_input_with_ligand(self) -> None:
        fold_input = parse_fold_input(COFOLD_PAYLOAD)
        self.assertEqual(len(fold_input.proteins), 1)
        self.assertEqual(len(fold_input.ligands), 1)
        self.assertEqual(fold_input.ligands[0].elements, ("C", "C", "O"))
        self.assertEqual(fold_input.ligand_atom_count, 3)

    def test_invalid_sequence_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_fold_input({"sequences": [{"protein": {"id": "A", "sequence": "ABZ"}}]})

    def test_ligand_without_smiles_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_fold_input({"sequences": [{"ligand": {"id": "L", "smiles": ""}}]})

    def test_feature_shapes_mark_ligand_tokens(self) -> None:
        fold_input = parse_fold_input(COFOLD_PAYLOAD)
        features = build_features(fold_input)

        # 4 protein tokens * 4 backbone atoms + 3 ligand tokens * 1 atom each
        self.assertEqual(len(features.tokens), 4 + 3)
        self.assertEqual(len(features.atoms), 16 + 3)
        self.assertEqual(sum(1 for t in features.tokens if t.is_ligand), 3)
        self.assertEqual(sum(1 for t in features.tokens if not t.is_ligand), 4)

    def test_embedding_shapes(self) -> None:
        fold_input = parse_fold_input(COFOLD_PAYLOAD)
        features = build_features(fold_input)
        embeddings = build_embeddings(features)

        num_tokens = len(features.tokens)
        self.assertEqual(len(embeddings.token_embeddings), num_tokens)
        self.assertEqual(len(embeddings.token_embeddings[0]), EMBED_DIM)
        self.assertEqual(len(embeddings.pair_embeddings), num_tokens)
        self.assertEqual(len(embeddings.pair_embeddings[0]), num_tokens)

    def test_pairformer_preserves_shapes_and_changes_values(self) -> None:
        fold_input = parse_fold_input(COFOLD_PAYLOAD)
        features = build_features(fold_input)
        embeddings = build_embeddings(features)
        tokens, pairs = run_pairformer(embeddings.token_embeddings, embeddings.pair_embeddings, num_blocks=2)

        self.assertEqual(len(tokens), len(features.tokens))
        self.assertEqual(len(tokens[0]), EMBED_DIM)
        self.assertNotEqual(tokens[0], embeddings.token_embeddings[0])

    def test_attention_variant_differs_from_averaging_variant(self) -> None:
        fold_input = parse_fold_input(COFOLD_PAYLOAD)
        features = build_features(fold_input)
        embeddings = build_embeddings(features)

        averaged_tokens, _ = run_pairformer(embeddings.token_embeddings, embeddings.pair_embeddings, num_blocks=2)
        attended_tokens, _ = run_pairformer_with_attention(
            embeddings.token_embeddings, embeddings.pair_embeddings, num_blocks=2
        )
        self.assertNotEqual(averaged_tokens[0], attended_tokens[0])

    def test_diffusion_converges_to_target(self) -> None:
        fold_input = parse_fold_input(COFOLD_PAYLOAD)
        features = build_features(fold_input)
        target = generate_demo_coordinates(features, seed=3)
        result = run_diffusion(features, seed=3, num_steps=6)

        for actual, expected in zip(result, target):
            self.assertAlmostEqual(actual.x, expected.x, places=6)
            self.assertAlmostEqual(actual.y, expected.y, places=6)
            self.assertAlmostEqual(actual.z, expected.z, places=6)

    def test_diffusion_ensemble_and_uncertainty(self) -> None:
        fold_input = parse_fold_input(COFOLD_PAYLOAD)
        features = build_features(fold_input)
        ensemble = run_diffusion_ensemble(features, seeds=(1, 2, 3), num_steps=4)
        self.assertEqual(len(ensemble), 3)

        uncertainty = per_atom_uncertainty(ensemble)
        self.assertEqual(len(uncertainty), len(ensemble[0]))
        self.assertTrue(all(value >= 0.0 for value in uncertainty))

    def test_lddt_is_100_for_identical_structures(self) -> None:
        fold_input = parse_fold_input(COFOLD_PAYLOAD)
        features = build_features(fold_input)
        coords = run_diffusion(features, seed=1, num_steps=4)
        lddt = compute_lddt(coords, coords)
        self.assertTrue(all(score == 100.0 for score in lddt))

    def test_contact_map_is_symmetric_and_self_contacting(self) -> None:
        fold_input = parse_fold_input(COFOLD_PAYLOAD)
        features = build_features(fold_input)
        coords = run_diffusion(features, seed=1, num_steps=4)
        contacts = contact_map(coords)

        n = len(contacts)
        for i in range(n):
            self.assertEqual(contacts[i][i], 1)  # zero self-distance is always "in contact"
            for j in range(n):
                self.assertEqual(contacts[i][j], contacts[j][i])

    def test_affinity_is_zero_without_a_ligand(self) -> None:
        fold_input = parse_fold_input({"sequences": [{"protein": {"id": "A", "sequence": "ACDE"}}]})
        features = build_features(fold_input)
        coords = run_diffusion(features, seed=1, num_steps=4)
        self.assertEqual(predict_affinity(features, coords), 0.0)

    def test_ddg_is_zero_for_the_wildtype_sequence(self) -> None:
        fold_input = parse_fold_input(COFOLD_PAYLOAD)
        features = build_features(fold_input)
        coords = run_diffusion(features, seed=1, num_steps=4)
        wildtype_sequence = fold_input.proteins[0].sequence
        self.assertEqual(predict_ddg(features, coords, wildtype_sequence), 0.0)

    def test_mock_docking_prefers_moderate_separation(self) -> None:
        fold_input = parse_fold_input(COFOLD_PAYLOAD)
        features = build_features(fold_input)
        coords = run_diffusion(features, seed=1, num_steps=4)
        protein_coords = tuple(c for a, c in zip(features.atoms, coords) if not a.is_ligand)
        ligand_coords = tuple(c for a, c in zip(features.atoms, coords) if a.is_ligand)

        poses = dock(protein_coords, ligand_coords, num_poses=3, num_candidates=6, seed=0)
        self.assertEqual(len(poses), 3)
        energies = [energy for _, energy in poses]
        self.assertEqual(energies, sorted(energies))

    def test_toy_pairwise_energy_is_a_real_number(self) -> None:
        fold_input = parse_fold_input(COFOLD_PAYLOAD)
        features = build_features(fold_input)
        coords = run_diffusion(features, seed=1, num_steps=4)
        protein_coords = tuple(c for a, c in zip(features.atoms, coords) if not a.is_ligand)
        ligand_coords = tuple(c for a, c in zip(features.atoms, coords) if a.is_ligand)
        energy = toy_pairwise_energy(protein_coords, ligand_coords)
        self.assertIsInstance(energy, float)

    def test_demo_export_contains_hetatm_for_ligand(self) -> None:
        fold_input = parse_fold_input(COFOLD_PAYLOAD)
        features = build_features(fold_input)
        coordinates = run_diffusion(features, seed=1, num_steps=4)
        mmcif = to_mmcif(features, coordinates)

        self.assertIn("_atom_site.Cartn_x", mmcif)
        atom_lines = [line for line in mmcif.splitlines() if line.startswith("ATOM ")]
        hetatm_lines = [line for line in mmcif.splitlines() if line.startswith("HETATM")]
        self.assertEqual(len(atom_lines) + len(hetatm_lines), len(features.atoms))
        self.assertEqual(len(hetatm_lines), 3)


if __name__ == "__main__":
    unittest.main()
