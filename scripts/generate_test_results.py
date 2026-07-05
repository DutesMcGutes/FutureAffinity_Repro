from __future__ import annotations

from pathlib import Path
import importlib.util
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CHAPTERS = {
    "01_feature_extraction": "summarize_feature_extraction",
    "02_input_embedding": "summarize_input_embedding",
    "03_pairformer": "summarize_pairformer",
    "04_diffusion": "summarize_diffusion",
    "05_multitask_heads": "summarize_multitask_heads",
    "06_synthetic_supervision": "summarize_synthetic_supervision",
    "07_inference_and_export": "summarize_inference_and_export",
}


def _load_solution_module(chapter: str):
    solution_path = ROOT / "solutions" / f"{chapter}.py"
    spec = importlib.util.spec_from_file_location(f"{chapter}_solution", solution_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load solution module at {solution_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generate_chapter(chapter: str) -> Path:
    function_name = CHAPTERS[chapter]
    module = _load_solution_module(chapter)
    summarize = getattr(module, function_name)

    payload = json.loads((ROOT / "examples" / "aspirin_lysozyme.json").read_text(encoding="utf-8"))
    summary = summarize(payload)

    output_dir = ROOT / "tests" / "expected"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{chapter}.json"
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    chapter = sys.argv[1] if len(sys.argv) > 1 else "01_feature_extraction"
    if chapter == "all":
        for name in CHAPTERS:
            print(f"Wrote {generate_chapter(name)}")
        return
    if chapter not in CHAPTERS:
        raise SystemExit(f"Unknown chapter: {chapter}")
    print(f"Wrote {generate_chapter(chapter)}")


if __name__ == "__main__":
    main()
