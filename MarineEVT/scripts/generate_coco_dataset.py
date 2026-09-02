#!/usr/bin/env python3
"""Generate COCO detection/instance pseudo-labels with multi-GPU SAM3 services."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import logging
import math
import random
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
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
    min_score: float | None = None


@dataclass(frozen=True)
class FrameItem:
    source_key: str
    source_path: str
    frame_index: int | None
    output_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate COCO annotations using one or more MarineEVT SAM3 services.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True, help="Video/image file or recursive directory.")
    parser.add_argument("--output", type=Path, required=True, help="Dataset output directory.")
    category_group = parser.add_mutually_exclusive_group(required=True)
    category_group.add_argument("--categories", help="Comma-separated names, e.g. fish,shark,turtle,diver.")
    category_group.add_argument(
        "--categories-file", type=Path,
        help="JSON category configuration; supports per-category min_score.",
    )
    parser.add_argument(
        "--sam-url", action="append", dest="sam_url_items",
        help="Full /v1/detect endpoint; repeat this option for multiple workers.",
    )
    parser.add_argument(
        "--sam-urls",
        help="Comma-separated /v1/detect endpoints; may be combined with repeated --sam-url.",
    )
    parser.add_argument(
        "--workers", type=int, default=0,
        help="Concurrent frame requests; 0 uses the number of SAM endpoints.",
    )
    parser.add_argument("--ground-type", choices=("all", "highest"), default="all")
    parser.add_argument("--min-score", type=float, default=0.50, help="Global score threshold.")
    parser.add_argument(
        "--include-masks", action=argparse.BooleanOptionalAction, default=False,
        help="Store uncompressed COCO RLE masks; increases network and JSON size.",
    )
    parser.add_argument("--sample-every-seconds", type=float, default=2.0)
    parser.add_argument("--max-frames-per-video", type=int, default=0, help="0 means unlimited.")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--min-box-width", type=float, default=4.0)
    parser.add_argument("--min-box-height", type=float, default=4.0)
    parser.add_argument("--min-area-ratio", type=float, default=0.0001)
    parser.add_argument("--max-area-ratio", type=float, default=0.95)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--request-retries", type=int, default=2)
    parser.add_argument("--retry-backoff", type=float, default=2.0)
    parser.add_argument(
        "--skip-health-check", action="store_true",
        help="Do not verify every SAM endpoint before processing.",
    )
    parser.add_argument(
        "--splits", default="0.8,0.1,0.1",
        help="Train,val,test ratios; assignment is grouped by source video/image.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-empty", action=argparse.BooleanOptionalAction, default=True,
        help="Keep no-detection frames as negative examples.",
    )
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True,
        help="Reuse successful records from output/work/progress.jsonl.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Process at most N frames for smoke tests.")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser.parse_args()


def resolve_sam_urls(args: argparse.Namespace) -> list[str]:
    values = list(args.sam_url_items or [])
    if args.sam_urls:
        values.extend(part.strip() for part in args.sam_urls.split(",") if part.strip())
    if not values:
        values = ["http://127.0.0.1:8111/v1/detect"]
    urls: list[str] = []
    for value in values:
        url = value.rstrip("/")
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"Invalid SAM URL: {value}")
        if url not in urls:
            urls.append(url)
    return urls


def load_categories(args: argparse.Namespace) -> list[Category]:
    if args.categories:
        raw: list[Any] = [part.strip() for part in args.categories.split(",") if part.strip()]
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
        if name.casefold() in seen_names:
            raise ValueError(f"Duplicate category name: {name}")
        seen_names.add(name.casefold())
        min_score = item.get("min_score")
        min_score = float(min_score) if min_score is not None else None
        if min_score is not None and not 0.0 <= min_score <= 1.0:
            raise ValueError(f"min_score for {name} must be between 0 and 1")
        categories.append(Category(
            id=int(item.get("id", index)),
            name=name,
            prompt=str(item.get("prompt", name)).strip(),
            supercategory=str(item.get("supercategory", "object")).strip(),
            min_score=min_score,
        ))
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
    return (
        [path for path in candidates if path.suffix.lower() in IMAGE_SUFFIXES],
        [path for path in candidates if path.suffix.lower() in VIDEO_SUFFIXES],
    )


def safe_stem(path: Path) -> str:
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:10]
    cleaned = "".join(character if character.isalnum() or character in "-_" else "_" for character in path.stem)
    return f"{cleaned[:80]}_{digest}"


def prepare_image(source: Path, images_dir: Path, jpeg_quality: int) -> FrameItem:
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("Image processing requires Pillow") from error
    destination = images_dir / f"{safe_stem(source)}.jpg"
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
        raise RuntimeError("Video input requires opencv-python-headless") from error
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        LOGGER.warning("Could not open video: %s", video)
        return []
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        fps = 25.0
        LOGGER.warning("Invalid FPS for %s; using %.1f", video, fps)
    step = max(1, round(fps * sample_every_seconds))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    prefix = safe_stem(video)
    items: list[FrameItem] = []

    if total > 0:
        for frame_index in range(0, total, step):
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
                    if not cv2.imwrite(str(destination), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]):
                        frame_index += 1
                        continue
                items.append(FrameItem(str(video.resolve()), str(video.resolve()), frame_index, destination.resolve()))
                if max_frames and len(items) >= max_frames:
                    break
            frame_index += 1
    capture.release()
    return items


def http_json(url: str, payload: dict[str, Any] | None, timeout: float) -> Any:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8") if payload is not None else None,
        headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def health_url(detect_url: str) -> str:
    if detect_url.endswith("/v1/detect"):
        return detect_url[: -len("/v1/detect")] + "/health"
    return detect_url.rsplit("/", maxsplit=1)[0] + "/health"


def check_endpoints(urls: list[str], timeout: float) -> None:
    for url in urls:
        try:
            response = http_json(health_url(url), None, min(timeout, 30.0))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            raise RuntimeError(f"SAM endpoint health check failed for {url}: {error}") from error
        if response.get("status") != "ok" or not response.get("model_loaded"):
            raise RuntimeError(f"SAM endpoint is not ready: {url}: {response}")
        LOGGER.info("SAM ready: %s | GPU %s | %s", url, response.get("physical_gpu"), response.get("cuda_device_name"))


def request_detections(
    sam_url: str,
    image_path: Path,
    categories: list[Category],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    server_floor = min(
        category.min_score if category.min_score is not None else args.min_score
        for category in categories
    )
    payload = {
        "image_path": str(image_path),
        "prompts": [{"category_id": category.id, "text": category.prompt} for category in categories],
        "ground_type": args.ground_type,
        "min_score": server_floor,
        "include_masks": args.include_masks,
    }
    last_error: Exception | None = None
    for attempt in range(args.request_retries + 1):
        try:
            response = http_json(sam_url, payload, args.request_timeout)
            result = response.get("result", response)
            detections = result["detections"]
            if not isinstance(detections, list):
                raise ValueError("detections is not a list")
            return detections
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError, TypeError, KeyError) as error:
            last_error = error
            if attempt < args.request_retries:
                time.sleep(args.retry_backoff * (2**attempt))
    raise RuntimeError(f"SAM3 request to {sam_url} failed after {args.request_retries + 1} attempts: {last_error}")


def parse_sam_boxes(response: Any) -> list[list[float]]:
    """Retained for compatibility tests and old saved responses."""
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
    intersection_width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    intersection_height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = intersection_width * intersection_height
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def filter_detections(
    detections: Iterable[dict[str, Any]],
    categories: list[Category],
    width: int,
    height: int,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    category_map = {category.id: category for category in categories}
    image_area = float(width * height)
    grouped: dict[int, list[dict[str, Any]]] = {category.id: [] for category in categories}
    for raw in detections:
        category_id = int(raw["category_id"])
        if category_id not in category_map:
            LOGGER.warning("Ignoring unknown category id %s", category_id)
            continue
        score = float(raw.get("score", 0.0))
        threshold = category_map[category_id].min_score
        threshold = args.min_score if threshold is None else threshold
        if not math.isfinite(score) or score < threshold:
            continue
        coordinates = raw.get("bbox_xyxy", raw.get("bbox"))
        if not isinstance(coordinates, (list, tuple)) or len(coordinates) != 4:
            continue
        x1, y1, x2, y2 = map(float, coordinates)
        if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
            continue
        x1, x2 = sorted((max(0.0, min(x1, width)), max(0.0, min(x2, width))))
        y1, y2 = sorted((max(0.0, min(y1, height)), max(0.0, min(y2, height))))
        box_width, box_height = x2 - x1, y2 - y1
        area_ratio = box_width * box_height / image_area if image_area else 0.0
        if box_width < args.min_box_width or box_height < args.min_box_height:
            continue
        if area_ratio < args.min_area_ratio or area_ratio > args.max_area_ratio:
            continue
        item = {
            "category_id": category_id,
            "xyxy": [x1, y1, x2, y2],
            "score": score,
        }
        if isinstance(raw.get("segmentation"), dict):
            item["segmentation"] = raw["segmentation"]
            item["mask_area"] = int(raw.get("mask_area", 0))
        grouped[category_id].append(item)

    kept: list[dict[str, Any]] = []
    for candidates in grouped.values():
        candidates.sort(key=lambda item: item["score"], reverse=True)
        category_kept: list[dict[str, Any]] = []
        for candidate in candidates:
            if all(box_iou(candidate["xyxy"], existing["xyxy"]) < args.nms_iou for existing in category_kept):
                category_kept.append(candidate)
        kept.extend(category_kept)
    return kept


def failed_record(item: FrameItem, signature: str, message: str) -> dict[str, Any]:
    return {
        "image_path": str(item.output_path),
        "source_key": item.source_key,
        "source_path": item.source_path,
        "frame_index": item.frame_index,
        "width": 0,
        "height": 0,
        "annotations": [],
        "errors": [message],
        "signature": signature,
    }


def annotate_frame(
    item: FrameItem,
    categories: list[Category],
    args: argparse.Namespace,
    signature: str,
    sam_url: str,
) -> dict[str, Any]:
    try:
        from PIL import Image
        with Image.open(item.output_path) as image:
            width, height = image.size
        raw = request_detections(sam_url, item.output_path, categories, args)
        annotations = filter_detections(raw, categories, width, height, args)
        return {
            "image_path": str(item.output_path),
            "source_key": item.source_key,
            "source_path": item.source_path,
            "frame_index": item.frame_index,
            "width": width,
            "height": height,
            "annotations": annotations,
            "errors": [],
            "signature": signature,
            "sam_url": sam_url,
        }
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
        return failed_record(item, signature, str(error))


def annotation_signature(categories: list[Category], args: argparse.Namespace, sam_urls: list[str]) -> str:
    config = {
        "categories": [asdict(category) for category in categories],
        "sam_urls": sorted(sam_urls),
        "ground_type": args.ground_type,
        "min_score": args.min_score,
        "include_masks": args.include_masks,
        "min_box_width": args.min_box_width,
        "min_box_height": args.min_box_height,
        "min_area_ratio": args.min_area_ratio,
        "max_area_ratio": args.max_area_ratio,
        "nms_iou": args.nms_iou,
    }
    return hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def load_progress(path: Path, signature: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
                if not record.get("errors") and record.get("signature") == signature:
                    records[record["image_path"]] = record
            except (json.JSONDecodeError, KeyError):
                LOGGER.warning("Ignoring invalid progress line %d", line_number)
    return records


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
    train_end = round(len(keys) * ratios[0])
    val_end = min(train_end + round(len(keys) * ratios[1]), len(keys))
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
            images.append({
                "id": image_id,
                "file_name": image_path.relative_to(output).as_posix(),
                "width": record["width"],
                "height": record["height"],
                "source": record["source_path"],
                "frame_index": record["frame_index"],
            })
            for item in record["annotations"]:
                x1, y1, x2, y2 = item["xyxy"]
                box_width, box_height = x2 - x1, y2 - y1
                annotation: dict[str, Any] = {
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": item["category_id"],
                    "bbox": [round(x1, 3), round(y1, 3), round(box_width, 3), round(box_height, 3)],
                    "area": round(item.get("mask_area") or box_width * box_height, 3),
                    "iscrowd": 0,
                    "score": round(item["score"], 6),
                }
                if "segmentation" in item:
                    annotation["segmentation"] = item["segmentation"]
                annotations.append(annotation)
        destination = annotation_dir / f"instances_{split}.json"
        document = {
            "info": {"description": "MarineEVT SAM3 auto-generated pseudo-labels", "version": "2.0"},
            "licenses": [],
            "images": images,
            "annotations": annotations,
            "categories": coco_categories,
        }
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


def validate_arguments(args: argparse.Namespace) -> None:
    if args.sample_every_seconds <= 0:
        raise ValueError("--sample-every-seconds must be positive")
    if args.workers < 0:
        raise ValueError("--workers cannot be negative")
    if not 0 <= args.min_score <= 1:
        raise ValueError("--min-score must be between 0 and 1")
    if not 0 <= args.nms_iou <= 1:
        raise ValueError("--nms-iou must be between 0 and 1")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 1 and 100")


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(message)s")
    try:
        validate_arguments(args)
        categories = load_categories(args)
        sam_urls = resolve_sam_urls(args)
        ratios = parse_splits(args.splits)
        worker_count = args.workers or len(sam_urls)
        if not args.skip_health_check:
            check_endpoints(sam_urls, args.request_timeout)

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
        LOGGER.info("Prepared %d frames; %d endpoints; %d concurrent requests", len(frames), len(sam_urls), worker_count)

        progress_path = work_dir / "progress.jsonl"
        signature = annotation_signature(categories, args, sam_urls)
        completed = load_progress(progress_path, signature) if args.resume else {}
        records_by_path = dict(completed)
        pending = [item for item in frames if str(item.output_path) not in completed]
        mode = "a" if args.resume else "w"

        with progress_path.open(mode, encoding="utf-8") as progress_file:
            with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
                future_map = {
                    executor.submit(
                        annotate_frame,
                        item,
                        categories,
                        args,
                        signature,
                        sam_urls[index % len(sam_urls)],
                    ): item
                    for index, item in enumerate(pending)
                }
                for completed_count, future in enumerate(concurrent.futures.as_completed(future_map), start=1):
                    item = future_map[future]
                    try:
                        record = future.result()
                    except Exception as error:  # isolate one frame from the rest of a large job
                        record = failed_record(item, signature, f"Unexpected worker error: {error}")
                    records_by_path[str(item.output_path)] = record
                    progress_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    progress_file.flush()
                    if record["errors"]:
                        LOGGER.error("[%d/%d] %s | %s", completed_count, len(pending), item.output_path.name, record["errors"][0])
                    else:
                        LOGGER.info("[%d/%d] %s | %d annotations", completed_count, len(pending), item.output_path.name, len(record["annotations"]))

        failed = [records_by_path[str(item.output_path)] for item in frames if records_by_path[str(item.output_path)]["errors"]]
        if failed:
            failure_path = work_dir / "failed_records.json"
            with failure_path.open("w", encoding="utf-8") as handle:
                json.dump(failed, handle, ensure_ascii=False, indent=2)
            raise RuntimeError(f"{len(failed)} frames failed; rerun with --resume after fixing services. See {failure_path}")

        records = [records_by_path[str(item.output_path)] for item in frames]
        validate_records(records, {category.id for category in categories})
        split_map = assign_splits((record["source_key"] for record in records), ratios, args.seed)
        write_coco_files(records, categories, output, split_map, args.include_empty)
        manifest = {
            "input": str(args.input.resolve()),
            "sam_urls": sam_urls,
            "workers": worker_count,
            "categories": [asdict(category) for category in categories],
            "frames": len(records),
            "annotations": sum(len(record["annotations"]) for record in records),
            "include_masks": args.include_masks,
            "source_splits": split_map,
            "annotation_signature": signature,
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
