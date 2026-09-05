import importlib.util
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_sfishtrack_one", ROOT / "scripts" / "prepare_sfishtrack_one.py"
)
PREPARE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = PREPARE
SPEC.loader.exec_module(PREPARE)


class SfishtrackPrepareTests(unittest.TestCase):
    def test_extracts_only_selected_video_and_matching_companions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "sfishtrack.zip"
            output = root / "subset"
            annotation = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "fish"}]}
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("dataset/videos/video_001.mp4", b"one")
                archive.writestr("dataset/videos/video_002.mp4", b"two")
                archive.writestr("dataset/annotations/video_001.json", json.dumps(annotation))
                archive.writestr("dataset/annotations/video_002.json", json.dumps(annotation))
                archive.writestr("dataset/metadata/video_001.json", "{}")
                archive.writestr("dataset/frames/video_001/frame_000001.jpg", b"unused")

            PREPARE.prepare_destination(output, force=False)
            result = PREPARE.copy_from_zip(archive_path, output, "video_001")

            self.assertEqual((output / "input" / "video_001.mp4").read_bytes(), b"one")
            self.assertTrue((output / "ground_truth" / "video_001.json").is_file())
            self.assertTrue((output / "metadata" / "video_001.json").is_file())
            self.assertFalse((output / "input" / "video_002.mp4").exists())
            self.assertEqual(Path(result["video"]).name, "video_001.mp4")


if __name__ == "__main__":
    unittest.main()
