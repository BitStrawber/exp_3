#!/usr/bin/env python3
"""Generate a COCO object-detection dataset with MarineEVT's SAM3 service.

The script accepts videos and/or image directories, samples/copies images, sends
one text prompt per category to ``tool_server.py``, filters the returned boxes,
and writes standard COCO ``instances_*.json`` files.

The SAM3 service and this script must see the generated image paths through the
same filesystem. This is naturally true when both run on the same machine or
inside containers with a shared mounted output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import random
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

LOGGER = logging.getLogger("marineevt.coco")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


@dataclass(frozen=True)
class Category:
    id: int
    name: str
    prompt: str
    supercategory: str = "object"


@dataclass(frozen=True)
class FrameItem:
    source_key: str
    source_path: str
    frame_index: int | None
    output_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate COCO detection annotations using MarineEVT SAM3.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Video/image file or directory searched recursively.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Dataset output directory.")
    category_group = parser.add_mutually_exclusive_group(required=True)
    category_group.add_argument(
        "--categories",
        help="Comma-separated names, for example: fish,shark,turtle,diver.",
    )
    category_group.add_argument(
        "--categories-file",
        type=Path,
        help="JSON category configuration. See scripts/coco_categories.example.json.",
    )
    parser.add_argument("--sam-url", default="http://127.0.0.1:8111/sam")
    parser.add_argument(
        "--ground-type",
        choices=("all", "highest"),
        default="all",
        help="Keep every SAM3 proposal or only its highest scoring proposal.",
    )
    parser.add_argument("--sample-every-seconds", type=float, default=2.0)
    parser.add_argument("--max-frames-per-video", type=int, default=0, help="0 means unlimited.")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--min-box-width", type=float, default=4.0)
    parser.add_argument("--min-box-height", type=float, default=4.0)
    parser.add_argument("--min-area-ratio", type=float, default=0.0001)
    parser.add_argument("--max-area-ratio", type=float, default=0.95)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--request-retries", type=int, default=2)
    parser.add_argument("--retry-backoff", type=float, default=2.0)
    parser.add_argument(
        "--splits",
        default="0.8,0.1,0.1",
        help="Train,val,test ratios; split assignment is by source video/image group.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-empty",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep images for which no object was detected as negative examples.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse completed records from output/work/progress.jsonl.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Process at most N frames; useful for smoke tests.")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser.parse_args()


def load_categories(args: argparse.Namespace) -> list[Category]:
    if args.categories:
        names = [part.strip() for part in args.categories.split(",") if part.strip()]
        raw: list[Any] = names
    else:
        with args.categories_file.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
        raw = document["categories"] if isinstance(document, dict) else document

    categories: list[Category] = []
    seen_names: set[str] = set()
    for index, item in enumerate(raw, start=1):
        if isinstance(item, str):
            item = {"name": item}
        if not isinstance(item, dict) or not str(item.get("name", "")).strip():
            raise ValueError(f"Invalid category at position {index}: {item!r}")
        name = str(item["name"]).strip()
        normalized = name.casefold()
        if normalized in seen_names:
            raise ValueError(f"Duplicate category name: {name}")
        seen_names.add(normalized)
        categories.append(
            Category(
                id=int(item.get("id", index)),
                name=name,
                prompt=str(item.get("prompt", name)).strip(),
                supercategory=str(item.get("supercategory", "object")).strip(),
            )
        )
    if not categories:
        raise ValueError("At least one category is required")
    ids = [category.id for category in categories]
    if len(ids) != len(set(ids)) or any(category_id <= 0 for category_id in ids):
        raise ValueError("Category ids must be unique positive integers")
    return categories


def discover_sources(input_path: Path) -> tuple[list[Path], list[Path]]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input does not exist: {input_path}")
    candidates = [input_path] if input_path.is_file() else sorted(p for p in input_path.rglob("*") if p.is_file())
    images = [p for p in candidates if p.suffix.lower() in IMAGE_SUFFIXES]
    videos = [p for p in candidates if p.suffix.lower() in VIDEO_SUFFIXES]
    return images, videos


def safe_stem(path: Path) -> str:
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:10]
    cleaned = "".join(char if char.isalnum() or char in "-_" else "_" for char in path.stem)
    return f"{cleaned[:80]}_{digest}"


def prepare_image(source: Path, images_dir: Path, jpeg_quality: int) -> FrameItem:
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("Image processing requires Pillow. Install the project dependencies.") from error
    stem = safe_stem(source)
    destination = images_dir / f"{stem}.jpg"
    if not destination.exists():
        with Image.open(source) as image:
            image.convert("RGB").save(destination, format="JPEG", quality=jpeg_quality)
    return FrameItem(str(source.resolve()), str(source.resolve()), None, destination.resolve())


def extract_video_frames(
    video: Path,
    images_dir: Path,
    sample_every_seconds: float,
    max_frames: int,
    jpeg_quality: int,
) -> list[FrameItem]:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "Video input requires OpenCV. Install the project dependencies or run "
            "`pip install opencv-python-headless`."
        ) from error
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        LOGGER.warning("Could not open video: %s", video)
        return []
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        fps = 25.0
        LOGGER.warning("Invalid FPS for %s; falling back to %.1f", video, fps)
    step = max(1, round(fps * sample_every_seconds))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    indices: Iterable[int]
    if total > 0:
        indices = range(0, total, step)
    else:
        indices = iter(lambda: -1, 0)  # handled by sequential fallback below

    items: list[FrameItem] = []
    prefix = safe_stem(video)
    if total > 0:
        for frame_index in indices:
            if max_frames and len(items) >= max_frames:
                break
            destination = images_dir / f"{prefix}_f{frame_index:09d}.jpg"
            if not destination.exists():
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if not ok:
                    LOGGER.warning("Failed to decode %s at frame %d", video, frame_index)
                    continue
                if not cv2.imwrite(str(destination), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]):
                    LOGGER.warning("Failed to write frame: %s", destination)
                    continue
            items.append(FrameItem(str(video.resolve()), str(video.resolve()), frame_index, destination.resolve()))
    else:
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % step == 0:
                destination = images_dir / f"{prefix}_f{frame_index:09d}.jpg"
                if not destination.exists():
                    cv2.imwrite(str(destination), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                items.append(FrameItem(str(video.resolve()), str(video.resolve()), frame_index, destination.resolve()))
                if max_frames and len(items) >= max_frames:
                    break
            frame_index += 1
    capture.release()
    return items


def request_boxes(
    sam_url: str,
    image_path: Path,
    prompt: str,
    ground_type: str,
    timeout: float,
    retries: int,
    backoff: float,
) -> list[list[float]]:
    payload = {"prompt": prompt, "image_paths": [str(image_path)], "ground_type": ground_type}
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(
                sam_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                response_body = json.loads(response.read().decode("utf-8"))
            return parse_sam_boxes(response_body)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError, TypeError, KeyError) as error:
            last_error = error
            if attempt < retries:
                time.sleep(backoff * (2**attempt))
    raise RuntimeError(f"SAM3 request failed after {retries + 1} attempts: {last_error}")


def parse_sam_boxes(response: Any) -> list[list[float]]:
    """Normalize current and common SAM API response shapes to XYXY boxes."""
    node = response.get("result", response) if isinstance(response, dict) else response
    node = node.get("boxes", node) if isinstance(node, dict) else node
    boxes: list[list[float]] = []

    def visit(value: Any) -> None:
        if isinstance(value, (list, tuple)):
            if len(value) == 4:
                try:
                    box = [float(coordinate) for coordinate in value]
                except (TypeError, ValueError):
                    pass
                else:
                    if all(math.isfinite(coordinate) for coordinate in box):
                        boxes.append(box)
                    return
            for child in value:
                visit(child)

    visit(node)
    return boxes


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    intersection_w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    intersection_h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = intersection_w * intersection_h
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def filter_boxes(boxes: Iterable[Sequence[float]], width: int, height: int, args: argparse.Namespace) -> list[list[float]]:
    valid: list[list[float]] = []
    image_area = float(width * height)
    for raw in boxes:
        x1, y1, x2, y2 = map(float, raw)
        x1, x2 = sorted((max(0.0, min(x1, width)), max(0.0, min(x2, width))))
        y1, y2 = sorted((max(0.0, min(y1, height)), max(0.0, min(y2, height))))
        box_width, box_height = x2 - x1, y2 - y1
        area_ratio = box_width * box_height / image_area if image_area else 0.0
        if box_width < args.min_box_width or box_height < args.min_box_height:
            continue
        if area_ratio < args.min_area_ratio or area_ratio > args.max_area_ratio:
            continue
        valid.append([x1, y1, x2, y2])

    # No scores are exposed by the current service, so prefer smaller boxes when
    # suppressing duplicates; this avoids a broad enclosing box swallowing instances.
    valid.sort(key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
    kept: list[list[float]] = []
    for candidate in valid:
        if all(box_iou(candidate, existing) < args.nms_iou for existing in kept):
            kept.append(candidate)
    return kept


def load_progress(path: Path, signature: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
                # Failed records must be retried. A signature prevents silently
                # reusing annotations made with different categories or filters.
                if not record.get("errors") and record.get("signature") == signature:
                    records[record["image_path"]] = record
            except (json.JSONDecodeError, KeyError):
                LOGGER.warning("Ignoring invalid progress line %d", line_number)
    return records


def annotate_frame(
    item: FrameItem,
    categories: list[Category],
    args: argparse.Namespace,
    signature: str,
) -> dict[str, Any]:
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("Image processing requires Pillow. Install the project dependencies.") from error
    with Image.open(item.output_path) as image:
        width, height = image.size
    annotations: list[dict[str, Any]] = []
    errors: list[str] = []
    for category in categories:
        try:
            raw_boxes = request_boxes(
                args.sam_url, item.output_path, category.prompt, args.ground_type,
                args.request_timeout, args.request_retries, args.retry_backoff,
            )
            boxes = filter_boxes(raw_boxes, width, height, args)
            annotations.extend({"category_id": category.id, "xyxy": box} for box in boxes)
        except RuntimeError as error:
            errors.append(f"{category.name}: {error}")
            LOGGER.error("%s | %s", item.output_path.name, errors[-1])
    return {
        "image_path": str(item.output_path),
        "source_key": item.source_key,
        "source_path": item.source_path,
        "frame_index": item.frame_index,
        "width": width,
        "height": height,
        "annotations": annotations,
        "errors": errors,
        "signature": signature,
    }


def annotation_signature(categories: list[Category], args: argparse.Namespace) -> str:
    relevant_config = {
        "categories": [category.__dict__ for category in categories],
        "sam_url": args.sam_url,
        "ground_type": args.ground_type,
        "min_box_width": args.min_box_width,
        "min_box_height": args.min_box_height,
        "min_area_ratio": args.min_area_ratio,
        "max_area_ratio": args.max_area_ratio,
        "nms_iou": args.nms_iou,
    }
    encoded = json.dumps(relevant_config, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_splits(value: str) -> tuple[float, float, float]:
    try:
        ratios = tuple(float(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise ValueError("--splits must contain three numeric ratios") from error
    if len(ratios) != 3 or any(ratio < 0 for ratio in ratios) or not math.isclose(sum(ratios), 1.0, abs_tol=1e-6):
        raise ValueError("--splits must contain three non-negative ratios summing to 1")
    return ratios  # type: ignore[return-value]


def assign_splits(source_keys: Iterable[str], ratios: tuple[float, float, float], seed: int) -> dict[str, str]:
    keys = sorted(set(source_keys))
    random.Random(seed).shuffle(keys)
    total = len(keys)
    train_end = round(total * ratios[0])
    val_end = train_end + round(total * ratios[1])
    val_end = min(val_end, total)
    return {
        key: "train" if index < train_end else "val" if index < val_end else "test"
        for index, key in enumerate(keys)
    }


def write_coco_files(
    records: list[dict[str, Any]],
    categories: list[Category],
    output: Path,
    split_map: dict[str, str],
    include_empty: bool,
) -> None:
    annotation_dir = output / "annotations"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    coco_categories = [
        {"id": item.id, "name": item.name, "supercategory": item.supercategory}
        for item in categories
    ]
    for split in ("train", "val", "test"):
        split_records = [record for record in records if split_map[record["source_key"]] == split]
        if not include_empty:
            split_records = [record for record in split_records if record["annotations"]]
        images: list[dict[str, Any]] = []
        annotations: list[dict[str, Any]] = []
        for image_id, record in enumerate(split_records, start=1):
            image_path = Path(record["image_path"])
            relative_path = image_path.relative_to(output).as_posix()
            images.append({
                "id": image_id,
                "file_name": relative_path,
                "width": record["width"],
                "height": record["height"],
                "source": record["source_path"],
                "frame_index": record["frame_index"],
            })
            for item in record["annotations"]:
                x1, y1, x2, y2 = item["xyxy"]
                box_width, box_height = x2 - x1, y2 - y1
                annotations.append({
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": item["category_id"],
                    "bbox": [round(x1, 3), round(y1, 3), round(box_width, 3), round(box_height, 3)],
                    "area": round(box_width * box_height, 3),
                    "iscrowd": 0,
                })
        document = {
            "info": {"description": "MarineEVT SAM3 auto-generated pseudo-labels", "version": "1.0"},
            "licenses": [],
            "images": images,
            "annotations": annotations,
            "categories": coco_categories,
        }
        destination = annotation_dir / f"instances_{split}.json"
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(document, handle, ensure_ascii=False, indent=2)
        LOGGER.info("%s: %d images, %d annotations -> %s", split, len(images), len(annotations), destination)


def validate_records(records: list[dict[str, Any]], category_ids: set[int]) -> None:
    for record in records:
        width, height = record["width"], record["height"]
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image dimensions: {record['image_path']}")
        for annotation in record["annotations"]:
            if annotation["category_id"] not in category_ids:
                raise ValueError(f"Unknown category id in {record['image_path']}")
            x1, y1, x2, y2 = annotation["xyxy"]
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError(f"Out-of-bounds box in {record['image_path']}: {annotation['xyxy']}")


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(message)s")
    try:
        categories = load_categories(args)
        ratios = parse_splits(args.splits)
        if args.sample_every_seconds <= 0:
            raise ValueError("--sample-every-seconds must be positive")
        if not 0 <= args.nms_iou <= 1:
            raise ValueError("--nms-iou must be between 0 and 1")
        if not 1 <= args.jpeg_quality <= 100:
            raise ValueError("--jpeg-quality must be between 1 and 100")

        output = args.output.resolve()
        images_dir = output / "images" / "all"
        work_dir = output / "work"
        images_dir.mkdir(parents=True, exist_ok=True)
        work_dir.mkdir(parents=True, exist_ok=True)
        image_sources, video_sources = discover_sources(args.input.resolve())
        if not image_sources and not video_sources:
            raise ValueError(f"No supported images or videos found under {args.input}")
        LOGGER.info("Found %d images and %d videos", len(image_sources), len(video_sources))

        frames = [prepare_image(path, images_dir, args.jpeg_quality) for path in image_sources]
        for index, video in enumerate(video_sources, start=1):
            LOGGER.info("Extracting video %d/%d: %s", index, len(video_sources), video)
            frames.extend(extract_video_frames(
                video, images_dir, args.sample_every_seconds, args.max_frames_per_video, args.jpeg_quality,
            ))
        if args.limit:
            frames = frames[: args.limit]
        LOGGER.info("Prepared %d frames", len(frames))

        progress_path = work_dir / "progress.jsonl"
        signature = annotation_signature(categories, args)
        completed = load_progress(progress_path, signature) if args.resume else {}
        mode = "a" if args.resume else "w"
        records: list[dict[str, Any]] = []
        with progress_path.open(mode, encoding="utf-8") as progress_file:
            for index, item in enumerate(frames, start=1):
                key = str(item.output_path)
                if key in completed:
                    record = completed[key]
                    LOGGER.info("[%d/%d] Reusing %s", index, len(frames), item.output_path.name)
                else:
                    LOGGER.info("[%d/%d] Annotating %s", index, len(frames), item.output_path.name)
                    record = annotate_frame(item, categories, args, signature)
                    progress_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    progress_file.flush()
                records.append(record)

        validate_records(records, {category.id for category in categories})
        split_map = assign_splits((record["source_key"] for record in records), ratios, args.seed)
        write_coco_files(records, categories, output, split_map, args.include_empty)
        manifest = {
            "input": str(args.input.resolve()),
            "sam_url": args.sam_url,
            "categories": [category.__dict__ for category in categories],
            "frames": len(records),
            "annotations": sum(len(record["annotations"]) for record in records),
            "source_splits": split_map,
            "warning": "Annotations are model-generated pseudo-labels and require quality review.",
        }
        with (output / "dataset_manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
        LOGGER.info("Dataset generation completed: %s", output)
        return 0
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        LOGGER.error("%s", error)
        return 1


if __name__ == "__main__":
    sys.exit(main())
