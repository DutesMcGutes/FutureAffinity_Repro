from __future__ import annotations

from pathlib import Path
import importlib.util
import json
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

CHAPTER_FUNCTIONS = {
    "01_feature_extraction": "summarize_feature_extraction",
    "02_input_embedding": "summarize_input_embedding",
    "03_pairformer": "summarize_pairformer",
    "04_diffusion": "summarize_diffusion",
    "05_multitask_heads": "summarize_multitask_heads",
    "06_synthetic_supervision": "summarize_synthetic_supervision",
    "07_inference_and_export": "summarize_inference_and_export",
    "08_metrics_and_equivariance": "summarize_metrics_and_equivariance",
}


def _load_solution_module(chapter: str):
    solution_path = ROOT / "solutions" / f"{chapter}.py"
    spec = importlib.util.spec_from_file_location(f"{chapter}_solution", solution_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TutorialTests(unittest.TestCase):
    def _assert_matches_expected(self, chapter: str) -> None:
        module = _load_solution_module(chapter)
        summarize = getattr(module, CHAPTER_FUNCTIONS[chapter])

        payload = json.loads((ROOT / "examples" / "aspirin_lysozyme.json").read_text(encoding="utf-8"))
        summary = summarize(payload)

        expected_path = ROOT / "tests" / "expected" / f"{chapter}.json"
        expected = json.loads(expected_path.read_text(encoding="utf-8"))

        self.assertEqual(summary, expected)

    def test_01_feature_extraction_matches_expected(self) -> None:
        self._assert_matches_expected("01_feature_extraction")

    def test_02_input_embedding_matches_expected(self) -> None:
        self._assert_matches_expected("02_input_embedding")

    def test_03_pairformer_matches_expected(self) -> None:
        self._assert_matches_expected("03_pairformer")

    def test_04_diffusion_matches_expected(self) -> None:
        self._assert_matches_expected("04_diffusion")

    def test_05_multitask_heads_matches_expected(self) -> None:
        self._assert_matches_expected("05_multitask_heads")

    def test_06_synthetic_supervision_matches_expected(self) -> None:
        self._assert_matches_expected("06_synthetic_supervision")

    def test_07_inference_and_export_matches_expected(self) -> None:
        self._assert_matches_expected("07_inference_and_export")

    def test_08_metrics_and_equivariance_matches_expected(self) -> None:
        self._assert_matches_expected("08_metrics_and_equivariance")


if __name__ == "__main__":
    unittest.main()
