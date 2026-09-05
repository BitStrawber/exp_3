#!/usr/bin/env python3
"""Extract or select one SFISHTRACK video and its matching COCO annotation."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path, PurePosixPath


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="SFISHTRACK ZIP archive or extracted directory")
    parser.add_argument("--output", type=Path, required=True, help="Destination for the one-video smoke-test subset")
    parser.add_argument("--video", help="Optional video filename/stem; otherwise select the first video")
    parser.add_argument("--force", action="store_true", help="Replace an already prepared subset")
    return parser.parse_args()


def choose_video(paths: list[str], requested: str | None) -> str:
    videos = sorted(path for path in paths if PurePosixPath(path).suffix.lower() in VIDEO_SUFFIXES)
    if not videos:
        raise ValueError("No video file was found in the SFISHTRACK source")
    if not requested:
        return videos[0]
    wanted = requested.casefold()
    matches = [
        path for path in videos
        if PurePosixPath(path).name.casefold() == wanted
        or PurePosixPath(path).stem.casefold() == Path(requested).stem.casefold()
    ]
    if len(matches) != 1:
        sample = ", ".join(PurePosixPath(path).name for path in videos[:10])
        raise ValueError(f"Expected one video matching {requested!r}, found {len(matches)}. Examples: {sample}")
    return matches[0]


def choose_companion(paths: list[str], video_path: str, directory_hint: str) -> str | None:
    stem = PurePosixPath(video_path).stem.casefold()
    candidates = [
        path for path in paths
        if PurePosixPath(path).suffix.lower() == ".json"
        and PurePosixPath(path).stem.casefold() == stem
        and directory_hint in path.casefold()
    ]
    if not candidates:
        candidates = [
            path for path in paths
            if PurePosixPath(path).suffix.lower() == ".json"
            and PurePosixPath(path).stem.casefold() == stem
        ]
    return sorted(candidates, key=lambda item: (len(PurePosixPath(item).parts), item))[0] if candidates else None


def prepare_destination(output: Path, force: bool) -> None:
    if output.exists() and any(output.iterdir()):
        if not force:
            raise FileExistsError(f"Destination is not empty: {output}. Add --force to replace it.")
        shutil.rmtree(output)
    (output / "input").mkdir(parents=True, exist_ok=True)
    (output / "ground_truth").mkdir(parents=True, exist_ok=True)
    (output / "metadata").mkdir(parents=True, exist_ok=True)


def copy_from_zip(source: Path, output: Path, requested: str | None) -> dict[str, str | None]:
    with zipfile.ZipFile(source) as archive:
        paths = [item.filename for item in archive.infolist() if not item.is_dir()]
        video = choose_video(paths, requested)
        annotation = choose_companion(paths, video, "annotation")
        metadata = choose_companion(paths, video, "metadata")
        selected = {"video": video, "annotation": annotation, "metadata": metadata}
        destinations = {
            "video": output / "input" / PurePosixPath(video).name,
            "annotation": output / "ground_truth" / PurePosixPath(annotation).name if annotation else None,
            "metadata": output / "metadata" / PurePosixPath(metadata).name if metadata else None,
        }
        for kind, member in selected.items():
            target = destinations[kind]
            if member and target:
                with archive.open(member) as source_handle, target.open("wb") as target_handle:
                    shutil.copyfileobj(source_handle, target_handle, length=8 * 1024 * 1024)
        return {kind: str(path.resolve()) if path else None for kind, path in destinations.items()}


def copy_from_directory(source: Path, output: Path, requested: str | None) -> dict[str, str | None]:
    files = [path for path in source.rglob("*") if path.is_file()]
    relative = [path.relative_to(source).as_posix() for path in files]
    video_rel = choose_video(relative, requested)
    annotation_rel = choose_companion(relative, video_rel, "annotation")
    metadata_rel = choose_companion(relative, video_rel, "metadata")
    selected = {
        "video": (source / video_rel, output / "input" / PurePosixPath(video_rel).name),
        "annotation": (
            source / annotation_rel,
            output / "ground_truth" / PurePosixPath(annotation_rel).name,
        ) if annotation_rel else None,
        "metadata": (
            source / metadata_rel,
            output / "metadata" / PurePosixPath(metadata_rel).name,
        ) if metadata_rel else None,
    }
    result: dict[str, str | None] = {}
    for kind, pair in selected.items():
        if pair is None:
            result[kind] = None
            continue
        source_path, target_path = pair
        shutil.copy2(source_path, target_path)
        result[kind] = str(target_path.resolve())
    return result


def main() -> int:
    args = parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.exists():
        print(f"Source does not exist: {source}", file=sys.stderr)
        return 1
    try:
        prepare_destination(output, args.force)
        if source.is_file() and zipfile.is_zipfile(source):
            manifest = copy_from_zip(source, output, args.video)
        elif source.is_dir():
            manifest = copy_from_directory(source, output, args.video)
        else:
            raise ValueError("Source must be an extracted directory or a ZIP archive")
        manifest["source"] = str(source)
        manifest_path = output / "subset_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps(manifest, indent=2))
        if not manifest.get("annotation"):
            print("WARNING: matching official annotation was not found; generation can run, evaluation cannot.", file=sys.stderr)
        return 0
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
