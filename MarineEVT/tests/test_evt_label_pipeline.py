import argparse
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location("generate_evt_label_dataset", ROOT / "scripts" / "generate_evt_label_dataset.py")
PIPELINE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = PIPELINE
SPEC.loader.exec_module(PIPELINE)


class PipelineIntegrationTests(unittest.TestCase):
    def test_image_to_quality_gated_coco(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow is not installed")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            Image.new("RGB", (100, 80), "navy").save(input_dir / "frame.jpg")
            categories = root / "categories.json"
            categories.write_text(json.dumps({"categories": [{
                "id": 1,
                "name": "fish",
                "prompts": ["fish", "a fish underwater"],
            }]}), encoding="utf-8")

            argv = [
                "generate_evt_label_dataset.py",
                "--input", str(input_dir),
                "--output", str(output_dir),
                "--categories-file", str(categories),
                "--planner", "heuristic",
                "--sam-urls", "http://fake/v1/detect",
                "--skip-health-check",
                "--splits", "1,0,0",
                "--review-overlays", "0",
            ]

            def fake_http(url, payload, timeout):
                self.assertEqual(len(payload["prompts"]), 2)
                return {"result": {"detections": [
                    {"category_id": 1, "prompt": "fish", "score": 0.9, "bbox_xyxy": [10, 10, 50, 50]},
                    {"category_id": 1, "prompt": "a fish underwater", "score": 0.8, "bbox_xyxy": [11, 11, 51, 51]},
                ]}}

            with mock.patch.object(sys, "argv", argv), mock.patch.object(PIPELINE, "http_json", side_effect=fake_http):
                self.assertEqual(PIPELINE.main(), 0)

            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                PIPELINE, "http_json", side_effect=AssertionError("SAM should have been resumed from cache")
            ):
                self.assertEqual(PIPELINE.main(), 0)

            coco = json.loads((output_dir / "annotations" / "instances_train.json").read_text(encoding="utf-8"))
            manifest = json.loads((output_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(len(coco["images"]), 1)
            self.assertEqual(len(coco["annotations"]), 1)
            self.assertEqual(coco["annotations"][0]["category_id"], 1)
            self.assertEqual(manifest["quality"]["ACCEPT"], 1)
            self.assertTrue((output_dir / "work" / "annotation_records.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
