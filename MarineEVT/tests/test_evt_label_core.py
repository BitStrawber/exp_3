import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evt_label.core import Category, assign_tracks, fuse_detections, load_categories, quality_gate


class CategoryTests(unittest.TestCase):
    def test_loads_multiple_prompts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "categories.json"
            path.write_text(json.dumps({"categories": [{
                "id": 1, "name": "fish", "prompts": ["fish", "a fish underwater"]
            }]}), encoding="utf-8")
            categories = load_categories(path)
        self.assertEqual(categories[0].prompts, ("fish", "a fish underwater"))


class FusionTests(unittest.TestCase):
    def test_prompt_agreement_and_nms(self):
        categories = [Category(1, "fish", ("fish", "a fish underwater"))]
        detections = [
            {"category_id": 1, "prompt": "fish", "score": 0.8, "bbox_xyxy": [10, 10, 50, 50]},
            {"category_id": 1, "prompt": "a fish underwater", "score": 0.7, "bbox_xyxy": [11, 11, 51, 51]},
            {"category_id": 1, "prompt": "fish", "score": 0.3, "bbox_xyxy": [70, 70, 90, 90]},
        ]
        fused = fuse_detections(detections, categories, 100, 100, global_min_score=0.35)
        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0]["prompt_support"], 2)
        self.assertEqual(fused[0]["prompt_agreement"], 1.0)


class TrackingAndQualityTests(unittest.TestCase):
    def test_tracks_are_one_to_one_and_quality_is_gated(self):
        records = [
            {"source_key": "video", "media_type": "video", "annotations": [
                {"category_id": 1, "xyxy": [10, 10, 30, 30], "score": 0.7, "prompt_agreement": 1.0},
                {"category_id": 1, "xyxy": [60, 60, 80, 80], "score": 0.7, "prompt_agreement": 1.0},
            ]},
            {"source_key": "video", "media_type": "video", "annotations": [
                {"category_id": 1, "xyxy": [11, 11, 31, 31], "score": 0.7, "prompt_agreement": 1.0},
                {"category_id": 1, "xyxy": [61, 61, 81, 81], "score": 0.7, "prompt_agreement": 1.0},
            ]},
        ]
        lengths = assign_tracks(records)
        self.assertEqual(sorted(lengths.values()), [2, 2])
        self.assertNotEqual(records[1]["annotations"][0]["track_id"], records[1]["annotations"][1]["track_id"])
        counts = quality_gate(records, min_track_length=2)
        self.assertEqual(counts["ACCEPT"], 4)

    def test_vlm_rejection_overrides_high_sam_score(self):
        records = [{"source_key": "image", "media_type": "image", "annotations": [{
            "category_id": 1, "xyxy": [0, 0, 10, 10], "score": 0.99,
            "prompt_agreement": 1.0, "vlm_is_target": False,
        }]}]
        counts = quality_gate(records)
        self.assertEqual(counts["REJECT"], 1)


if __name__ == "__main__":
    unittest.main()
