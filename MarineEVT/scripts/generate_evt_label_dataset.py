#!/usr/bin/env python3
"""Event-centric Qwen3-VL + SAM3 pipeline for producing reviewable COCO labels."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import logging
import math
import random
import re
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evt_label.core import Category, assign_tracks, fuse_detections, load_categories, quality_gate


LOGGER = logging.getLogger("evt-label")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


@dataclass(frozen=True)
class MediaInfo:
    source_key: str
    path: Path
    media_type: str
    duration: float | None
    fps: float | None
    frame_count: int | None


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    sample_fps: float
    categories: tuple[str, ...]
    priority: str = "normal"


@dataclass(frozen=True)
class FrameItem:
    source_key: str
    source_path: str
    media_type: str
    frame_index: int | None
    timestamp: float | None
    path: Path
    categories: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use Qwen3-VL planning/review and SAM3 grounding to build a quality-gated COCO dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--categories-file", type=Path, required=True)
    parser.add_argument("--planner", choices=("heuristic", "qwen"), default="heuristic")
    parser.add_argument("--plan-file", type=Path, help="Reuse a previously generated segment_plans.jsonl.")
    parser.add_argument("--qwen-review", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--qwen-model", help="Local path or Hub id, e.g. Qwen/Qwen3-VL-8B-Instruct.")
    parser.add_argument("--qwen-url", help="Persistent Qwen service endpoint, e.g. http://127.0.0.1:8100/v1/generate-json.")
    parser.add_argument("--qwen-timeout", type=float, default=600.0)
    parser.add_argument("--qwen-device", default="cuda:0")
    parser.add_argument("--qwen-dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--qwen-max-new-tokens", type=int, default=1024)
    parser.add_argument("--qwen-min-pixels", type=int, default=200704)
    parser.add_argument("--qwen-max-pixels", type=int, default=401408)
    parser.add_argument("--planner-coarse-frames", type=int, default=12)
    parser.add_argument("--planner-max-segments", type=int, default=8)
    parser.add_argument("--allow-planner-fallback", action="store_true")
    parser.add_argument("--sample-every-seconds", type=float, default=1.0)
    parser.add_argument("--max-frames-per-video", type=int, default=0)
    parser.add_argument("--sam-urls", default=",".join(f"http://127.0.0.1:{port}/v1/detect" for port in range(8111, 8119)))
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--include-masks", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--min-score", type=float, default=0.35)
    parser.add_argument("--cluster-iou", type=float, default=0.55)
    parser.add_argument("--nms-iou", type=float, default=0.75)
    parser.add_argument("--min-box-size", type=float, default=6.0)
    parser.add_argument("--min-area-ratio", type=float, default=0.0001)
    parser.add_argument("--max-area-ratio", type=float, default=0.90)
    parser.add_argument("--track-iou", type=float, default=0.30)
    parser.add_argument("--track-max-gap", type=int, default=2)
    parser.add_argument("--accept-score", type=float, default=0.65)
    parser.add_argument("--review-score", type=float, default=0.40)
    parser.add_argument("--accept-prompt-agreement", type=float, default=0.34)
    parser.add_argument("--min-track-length", type=int, default=2)
    parser.add_argument("--accept-single-frame-score", type=float, default=0.85)
    parser.add_argument("--require-vlm-accept", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-empty", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--exclude-review-frames", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--splits", default="0.8,0.1,0.1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--review-overlays", type=int, default=200)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-health-check", action="store_true")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in ("min_score", "cluster_iou", "nms_iou", "track_iou", "accept_score", "review_score"):
        value = getattr(args, name)
        if not 0 <= value <= 1:
            raise ValueError(f"--{name.replace('_', '-')} must be between 0 and 1")
    if args.sample_every_seconds <= 0 or args.planner_coarse_frames <= 0:
        raise ValueError("Sampling values must be positive")
    if args.qwen_min_pixels <= 0 or args.qwen_max_pixels < args.qwen_min_pixels:
        raise ValueError("Qwen pixel limits must be positive and max must be greater than or equal to min")
    if args.planner == "qwen" or args.qwen_review:
        if bool(args.qwen_model) == bool(args.qwen_url):
            raise ValueError("Qwen planning/review requires exactly one of --qwen-model or --qwen-url")
    if args.require_vlm_accept and not args.qwen_review:
        raise ValueError("--require-vlm-accept requires --qwen-review")


def safe_name(path: Path) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", path.stem)[:64]
    digest = hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:10]
    return f"{cleaned}_{digest}"


def discover_media(root: Path) -> list[MediaInfo]:
    if not root.exists():
        raise FileNotFoundError(root)
    files = [root] if root.is_file() else sorted(path for path in root.rglob("*") if path.is_file())
    result: list[MediaInfo] = []
    for path in files:
        suffix = path.suffix.lower()
        try:
            source_key = path.resolve().relative_to(root.resolve()).as_posix() if root.is_dir() else path.name
        except ValueError:
            source_key = str(path.resolve())
        if suffix in IMAGE_SUFFIXES:
            result.append(MediaInfo(source_key, path.resolve(), "image", None, None, None))
        elif suffix in VIDEO_SUFFIXES:
            try:
                import cv2
            except ImportError as error:
                raise RuntimeError("Video input requires opencv-python-headless") from error
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                LOGGER.warning("Skipping unreadable video: %s", path)
                continue
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            capture.release()
            fps = fps if math.isfinite(fps) and fps > 0 else 25.0
            duration = count / fps if count > 0 else 0.0
            result.append(MediaInfo(source_key, path.resolve(), "video", duration, fps, count))
    if not result:
        raise ValueError(f"No supported media found under {root}")
    return result


class QwenAgent:
    """Lazy local Qwen3-VL adapter with strict JSON output parsing."""

    def __init__(
        self, model_path: str, device: str, dtype_name: str, max_new_tokens: int,
        min_pixels: int, max_pixels: int,
    ):
        try:
            import torch
            from transformers import AutoProcessor
            try:
                from transformers import AutoModelForImageTextToText as AutoVisionModel
            except ImportError:
                from transformers import AutoModelForVision2Seq as AutoVisionModel
        except ImportError as error:
            raise RuntimeError("Qwen mode requires recent torch and transformers") from error
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype_name]
        LOGGER.info("Loading Qwen3-VL from %s on %s", model_path, device)
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True, min_pixels=min_pixels, max_pixels=max_pixels,
        )
        try:
            self.model = AutoVisionModel.from_pretrained(
                model_path, dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True,
            ).to(device).eval()
        except TypeError:
            self.model = AutoVisionModel.from_pretrained(
                model_path, torch_dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True,
            ).to(device).eval()
        self.device = device
        self.max_new_tokens = max_new_tokens

    @staticmethod
    def parse_json(text: str) -> dict[str, Any]:
        text = text.strip()
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fenced:
            text = fenced.group(1)
        else:
            start, end = text.find("{"), text.rfind("}")
            if start >= 0 and end > start:
                text = text[start:end + 1]
        value = json.loads(text)
        if not isinstance(value, dict):
            raise ValueError("Qwen response is not a JSON object")
        return value

    def generate_json(self, image_paths: list[Path], instruction: str) -> dict[str, Any]:
        import torch
        from PIL import Image
        images = []
        try:
            images = [Image.open(path).convert("RGB") for path in image_paths]
            content = [{"type": "image", "image": image} for image in images]
            content.append({"type": "text", "text": instruction})
            messages = [{"role": "user", "content": content}]
            prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[prompt], images=images, padding=True, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                generated = self.model.generate(**inputs, do_sample=False, max_new_tokens=self.max_new_tokens)
            generated = generated[:, inputs.input_ids.shape[1]:]
            text = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
            return self.parse_json(text)
        finally:
            for image in images:
                image.close()


class QwenHTTPAgent:
    """Client for a persistent, same-host Qwen3-VL service."""

    def __init__(self, endpoint: str, timeout: float):
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout

    def generate_json(self, image_paths: list[Path], instruction: str) -> dict[str, Any]:
        response = http_json(
            self.endpoint,
            {"image_paths": [str(path.resolve()) for path in image_paths], "instruction": instruction},
            self.timeout,
        )
        result = response.get("result", response)
        if not isinstance(result, dict):
            raise ValueError("Qwen service response is not a JSON object")
        return result


def write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def coarse_video_frames(media: MediaInfo, destination: Path, count: int) -> list[tuple[float, Path]]:
    import cv2
    destination.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(media.path))
    total = media.frame_count or 0
    indices = sorted({round(i * max(0, total - 1) / max(1, count - 1)) for i in range(count)})
    result: list[tuple[float, Path]] = []
    for frame_index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            continue
        target = destination / f"f{frame_index:09d}.jpg"
        cv2.imwrite(str(target), frame)
        result.append((frame_index / (media.fps or 25.0), target))
    capture.release()
    return result


def heuristic_segments(media: MediaInfo, categories: list[Category], args: argparse.Namespace) -> list[Segment]:
    if media.media_type == "image":
        return []
    duration = max(0.0, float(media.duration or 0.0))
    return [Segment(0.0, duration, 1.0 / args.sample_every_seconds, tuple(category.name for category in categories))]


def qwen_segments(
    media: MediaInfo, categories: list[Category], args: argparse.Namespace, agent: Any, work_dir: Path,
) -> list[Segment]:
    samples = coarse_video_frames(media, work_dir / "planner_frames" / safe_name(media.path), args.planner_coarse_frames)
    if not samples:
        raise RuntimeError(f"Could not sample planner frames from {media.path}")
    allowed = [category.name for category in categories]
    timestamps = [round(timestamp, 3) for timestamp, _ in samples]
    instruction = f"""You are planning automatic object annotation for an underwater video.
The attached frames are uniformly ordered and have timestamps {timestamps} seconds.
Allowed COCO categories: {allowed}.
Select at most {args.planner_max_segments} time intervals likely to contain clear instances of allowed categories.
Prefer short high-value intervals, but preserve recall. Use higher sample_fps for fast motion or brief events.
Only use allowed category names. Return JSON only with this schema:
{{"segments":[{{"start":0.0,"end":5.0,"sample_fps":2.0,"categories":["fish"],"priority":"high"}}]}}
Times must be seconds in [0,{media.duration or 0.0}]. Do not add commentary."""
    response = agent.generate_json([path for _, path in samples], instruction)
    raw_segments = response.get("segments")
    if not isinstance(raw_segments, list):
        raise ValueError("Qwen planner did not return a segments list")
    valid: list[Segment] = []
    allowed_set = set(allowed)
    duration = float(media.duration or 0.0)
    for raw in raw_segments[: args.planner_max_segments]:
        try:
            start = max(0.0, min(float(raw["start"]), duration))
            end = max(0.0, min(float(raw["end"]), duration))
            sample_fps = max(0.05, min(float(raw.get("sample_fps", 1.0)), media.fps or 25.0))
            selected = tuple(name for name in raw.get("categories", []) if name in allowed_set)
        except (KeyError, TypeError, ValueError):
            continue
        if end > start and selected:
            valid.append(Segment(start, end, sample_fps, selected, str(raw.get("priority", "normal"))))
    if not valid:
        raise ValueError("Qwen planner returned no valid annotation segments")
    return valid


def build_plans(
    media_items: list[MediaInfo], categories: list[Category], args: argparse.Namespace,
    agent: Any | None, work_dir: Path,
) -> dict[str, list[Segment]]:
    if args.plan_file:
        rows = read_jsonl(args.plan_file)
        return {
            row["source_key"]: [Segment(
                float(item["start"]), float(item["end"]), float(item["sample_fps"]),
                tuple(item["categories"]), str(item.get("priority", "normal")),
            ) for item in row.get("segments", [])]
            for row in rows
        }
    plans: dict[str, list[Segment]] = {}
    for index, media in enumerate(media_items, start=1):
        if media.media_type == "image":
            plans[media.source_key] = []
            continue
        LOGGER.info("Planning video %d/%d: %s", index, len(media_items), media.path)
        if args.planner == "qwen":
            try:
                assert agent is not None
                plans[media.source_key] = qwen_segments(media, categories, args, agent, work_dir)
            except Exception as error:
                if not args.allow_planner_fallback:
                    raise RuntimeError(f"Qwen planning failed for {media.path}: {error}") from error
                LOGGER.warning("Qwen planning failed; using full-video fallback: %s", error)
                plans[media.source_key] = heuristic_segments(media, categories, args)
        else:
            plans[media.source_key] = heuristic_segments(media, categories, args)
    rows = [
        {"source_key": media.source_key, "source_path": str(media.path), "segments": [asdict(item) for item in plans[media.source_key]]}
        for media in media_items
    ]
    write_jsonl(work_dir / "segment_plans.jsonl", rows)
    return plans


def prepare_frames(
    media_items: list[MediaInfo], plans: dict[str, list[Segment]], categories: list[Category],
    output: Path, args: argparse.Namespace,
) -> list[FrameItem]:
    from PIL import Image
    images_dir = output / "images" / "all"
    images_dir.mkdir(parents=True, exist_ok=True)
    result: list[FrameItem] = []
    all_category_names = tuple(category.name for category in categories)
    for media in media_items:
        prefix = safe_name(media.path)
        if media.media_type == "image":
            target = images_dir / f"{prefix}.jpg"
            if not target.exists():
                with Image.open(media.path) as image:
                    image.convert("RGB").save(target, quality=95)
            result.append(FrameItem(
                media.source_key, str(media.path), "image", None, None,
                target.resolve(), all_category_names,
            ))
            continue
        import cv2
        capture = cv2.VideoCapture(str(media.path))
        fps = media.fps or 25.0
        frame_categories: dict[int, set[str]] = {}
        for segment in plans.get(media.source_key, []):
            step = max(1, round(fps / segment.sample_fps))
            for frame_index in range(round(segment.start * fps), round(segment.end * fps) + 1, step):
                frame_categories.setdefault(frame_index, set()).update(segment.categories)
        ordered = sorted(frame_categories)
        if args.max_frames_per_video:
            ordered = ordered[: args.max_frames_per_video]
        for frame_index in ordered:
            target = images_dir / f"{prefix}_f{frame_index:09d}.jpg"
            if not target.exists():
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if not ok or not cv2.imwrite(str(target), frame):
                    LOGGER.warning("Could not decode %s at frame %d", media.path, frame_index)
                    continue
            result.append(FrameItem(
                media.source_key, str(media.path), "video", frame_index, frame_index / fps, target.resolve(),
                tuple(sorted(frame_categories[frame_index])),
            ))
        capture.release()
    result.sort(key=lambda item: (item.source_key, item.frame_index if item.frame_index is not None else -1))
    return result[: args.limit] if args.limit else result


def http_json(url: str, payload: dict[str, Any] | None, timeout: float) -> Any:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"}, method="POST" if payload is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def endpoint_health(url: str) -> str:
    return url[:-len("/v1/detect")] + "/health" if url.endswith("/v1/detect") else url.rsplit("/", 1)[0] + "/health"


def annotate_one(
    frame: FrameItem, categories: list[Category], url: str, args: argparse.Namespace,
) -> dict[str, Any]:
    from PIL import Image
    with Image.open(frame.path) as image:
        width, height = image.size
    selected_names = set(frame.categories)
    selected_categories = [category for category in categories if category.name in selected_names]
    if not selected_categories:
        raise ValueError(f"Frame has no allowed planned categories: {frame.path}")
    payload = {
        "image_path": str(frame.path),
        "prompts": [
            {"category_id": category.id, "text": prompt}
            for category in selected_categories for prompt in category.prompts
        ],
        "ground_type": "all",
        "min_score": args.min_score,
        "include_masks": args.include_masks,
    }
    response = http_json(url, payload, args.request_timeout)
    raw = response.get("result", response).get("detections", [])
    annotations = fuse_detections(
        raw, categories, width, height, global_min_score=args.min_score,
        cluster_iou=args.cluster_iou, nms_iou=args.nms_iou, min_box_size=args.min_box_size,
        min_area_ratio=args.min_area_ratio, max_area_ratio=args.max_area_ratio,
    )
    return {
        "source_key": frame.source_key,
        "source_path": frame.source_path,
        "media_type": frame.media_type,
        "frame_index": frame.frame_index,
        "timestamp": frame.timestamp,
        "planned_categories": list(frame.categories),
        "image_path": str(frame.path),
        "width": width,
        "height": height,
        "annotations": annotations,
        "errors": [],
    }


def annotation_signature(categories: list[Category], urls: list[str], args: argparse.Namespace) -> str:
    value = {
        "categories": [asdict(category) for category in categories],
        "urls": sorted(urls), "include_masks": args.include_masks,
        "min_score": args.min_score, "cluster_iou": args.cluster_iou, "nms_iou": args.nms_iou,
        "min_box_size": args.min_box_size, "min_area_ratio": args.min_area_ratio,
        "max_area_ratio": args.max_area_ratio,
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load_sam_progress(path: Path, signature: str) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    result: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        if row.get("signature") == signature and not row.get("errors") and row.get("image_path"):
            result[row["image_path"]] = row
    return result


def draw_numbered_candidates(record: dict[str, Any], target: Path) -> None:
    from PIL import Image, ImageDraw
    with Image.open(record["image_path"]) as opened:
        image = opened.convert("RGB")
    draw = ImageDraw.Draw(image)
    for candidate_id, annotation in enumerate(record["annotations"]):
        x1, y1, x2, y2 = annotation["xyxy"]
        draw.rectangle((x1, y1, x2, y2), outline="yellow", width=max(2, round(min(image.size) / 300)))
        draw.text((x1 + 3, y1 + 3), str(candidate_id), fill="yellow", stroke_width=2, stroke_fill="black")
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target, quality=92)


def qwen_review_records(
    records: list[dict[str, Any]], categories: list[Category], agent: Any, work_dir: Path,
    *, model_name: str, resume: bool,
) -> None:
    category_map = {category.id: category.name for category in categories}
    progress_path = work_dir / "qwen_review_progress.jsonl"
    review_signature = hashlib.sha256(model_name.encode()).hexdigest()
    cache: dict[str, dict[str, Any]] = {}
    if resume and progress_path.exists():
        for row in read_jsonl(progress_path):
            if row.get("signature") == review_signature:
                cache[row.get("fingerprint", "")] = row
    progress_file = progress_path.open("a" if resume else "w", encoding="utf-8")
    for index, record in enumerate(records, start=1):
        if not record["annotations"]:
            continue
        fingerprint = hashlib.sha256(json.dumps({
            "image_path": record["image_path"],
            "candidates": [
                {"category_id": item["category_id"], "xyxy": item["xyxy"], "score": item["score"]}
                for item in record["annotations"]
            ],
        }, sort_keys=True).encode()).hexdigest()
        cached = cache.get(fingerprint)
        if cached is not None:
            for annotation, decision in zip(record["annotations"], cached["decisions"]):
                annotation["vlm_is_target"] = decision["vlm_is_target"]
                annotation["vlm_confidence"] = decision["vlm_confidence"]
            continue
        overlay = work_dir / "qwen_review" / f"{index:09d}.jpg"
        draw_numbered_candidates(record, overlay)
        candidates = [
            {"candidate_id": idx, "category": category_map[item["category_id"]], "sam_score": round(item["score"], 4)}
            for idx, item in enumerate(record["annotations"])
        ]
        instruction = f"""Review automatic underwater object annotations. The first image is original; the second has numbered yellow boxes.
Candidates: {json.dumps(candidates, ensure_ascii=False)}
For every candidate decide whether the box actually contains an instance of its stated category and whether the box covers that instance reasonably.
Be conservative about rocks, coral, shadows, reflections and water artifacts. Return JSON only:
{{"decisions":[{{"candidate_id":0,"is_target":true,"confidence":0.9}}]}}
Include every candidate exactly once. Do not add commentary."""
        try:
            response = agent.generate_json([Path(record["image_path"]), overlay], instruction)
        except Exception as error:
            LOGGER.warning("Qwen review failed for %s; sending candidates to REVIEW: %s", record["image_path"], error)
            for annotation in record["annotations"]:
                annotation["vlm_is_target"] = None
                annotation["vlm_confidence"] = None
            continue
        by_id = {
            int(item["candidate_id"]): item for item in response.get("decisions", [])
            if isinstance(item, dict) and "candidate_id" in item
        }
        for candidate_id, annotation in enumerate(record["annotations"]):
            decision = by_id.get(candidate_id)
            if decision is None:
                annotation["vlm_is_target"] = None
                annotation["vlm_confidence"] = None
            else:
                raw_target = decision.get("is_target")
                annotation["vlm_is_target"] = raw_target if isinstance(raw_target, bool) else None
                try:
                    annotation["vlm_confidence"] = max(0.0, min(float(decision.get("confidence", 0.0)), 1.0))
                except (TypeError, ValueError):
                    annotation["vlm_confidence"] = None
        decisions = [
            {"vlm_is_target": item.get("vlm_is_target"), "vlm_confidence": item.get("vlm_confidence")}
            for item in record["annotations"]
        ]
        progress_file.write(json.dumps({
            "signature": review_signature, "fingerprint": fingerprint,
            "image_path": record["image_path"], "decisions": decisions,
        }, ensure_ascii=False) + "\n")
        progress_file.flush()
        if index % 25 == 0:
            LOGGER.info("Qwen reviewed %d/%d frames", index, len(records))
    progress_file.close()


def parse_splits(value: str) -> tuple[float, float, float]:
    ratios = tuple(float(item) for item in value.split(","))
    if len(ratios) != 3 or any(item < 0 for item in ratios) or not math.isclose(sum(ratios), 1.0, abs_tol=1e-6):
        raise ValueError("--splits must contain three non-negative values summing to 1")
    return ratios  # type: ignore[return-value]


def split_sources(records: list[dict[str, Any]], ratios: tuple[float, float, float], seed: int) -> dict[str, str]:
    sources = sorted({record["source_key"] for record in records})
    random.Random(seed).shuffle(sources)
    train_end = round(len(sources) * ratios[0])
    val_end = min(len(sources), train_end + round(len(sources) * ratios[1]))
    return {
        source: "train" if index < train_end else "val" if index < val_end else "test"
        for index, source in enumerate(sources)
    }


def write_coco(
    records: list[dict[str, Any]], categories: list[Category], output: Path,
    source_splits: dict[str, str], args: argparse.Namespace,
) -> dict[str, dict[str, int]]:
    annotation_dir = output / "annotations"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        images: list[dict[str, Any]] = []
        annotations: list[dict[str, Any]] = []
        for record in (item for item in records if source_splits[item["source_key"]] == split):
            accepted = [item for item in record["annotations"] if item["quality_decision"] == "ACCEPT"]
            has_review = any(item["quality_decision"] == "REVIEW" for item in record["annotations"])
            if args.exclude_review_frames and has_review:
                continue
            if not accepted and not (args.include_empty and not record["annotations"]):
                continue
            image_id = len(images) + 1
            images.append({
                "id": image_id,
                "file_name": Path(record["image_path"]).relative_to(output).as_posix(),
                "width": record["width"], "height": record["height"],
                "source": record["source_path"], "frame_index": record["frame_index"],
                "timestamp": record["timestamp"],
            })
            for item in accepted:
                x1, y1, x2, y2 = item["xyxy"]
                width, height = x2 - x1, y2 - y1
                annotation: dict[str, Any] = {
                    "id": len(annotations) + 1, "image_id": image_id,
                    "category_id": item["category_id"],
                    "bbox": [round(x1, 3), round(y1, 3), round(width, 3), round(height, 3)],
                    "area": round(item.get("mask_area") or width * height, 3), "iscrowd": 0,
                    "score": round(item["score"], 6), "track_id": item.get("track_id"),
                    "prompt_agreement": round(item.get("prompt_agreement", 1.0), 6),
                    "vlm_confidence": item.get("vlm_confidence"), "label_source": "Qwen3-VL+SAM3",
                }
                if "segmentation" in item:
                    annotation["segmentation"] = item["segmentation"]
                annotations.append(annotation)
        document = {
            "info": {"description": "EVT-Label Qwen3-VL + SAM3 pseudo-labels", "version": "1.0"},
            "licenses": [], "images": images, "annotations": annotations,
            "categories": [{"id": item.id, "name": item.name, "supercategory": item.supercategory} for item in categories],
        }
        with (annotation_dir / f"instances_{split}.json").open("w", encoding="utf-8") as handle:
            json.dump(document, handle, ensure_ascii=False, indent=2)
        stats[split] = {"images": len(images), "annotations": len(annotations)}
    return stats


def write_review_artifacts(records: list[dict[str, Any]], output: Path, max_overlays: int) -> int:
    review_records = []
    overlay_count = 0
    for record in records:
        review = [item for item in record["annotations"] if item["quality_decision"] == "REVIEW"]
        if not review:
            continue
        review_records.append({**record, "annotations": review})
        if overlay_count < max_overlays:
            draw_numbered_candidates({**record, "annotations": review}, output / "review" / "overlays" / f"{overlay_count:06d}.jpg")
            overlay_count += 1
    write_jsonl(output / "review" / "review_queue.jsonl", review_records)
    return len(review_records)


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(message)s")
    try:
        validate_args(args)
        output = args.output.resolve()
        work_dir = output / "work"
        work_dir.mkdir(parents=True, exist_ok=True)
        categories = load_categories(args.categories_file)
        media_items = discover_media(args.input.resolve())
        write_jsonl(work_dir / "media_index.jsonl", [
            {**asdict(item), "path": str(item.path)} for item in media_items
        ])

        agent = None
        qwen_identity = str(args.qwen_model or args.qwen_url)
        qwen_service: dict[str, Any] | None = None
        if args.planner == "qwen" or args.qwen_review:
            if args.qwen_url:
                normalized_qwen_url = args.qwen_url.rstrip("/")
                qwen_health_url = (
                    normalized_qwen_url[:-len("/v1/generate-json")] + "/health"
                    if normalized_qwen_url.endswith("/v1/generate-json")
                    else normalized_qwen_url + "/health"
                )
                health = http_json(qwen_health_url, None, min(30.0, args.qwen_timeout))
                if health.get("status") != "ok" or not health.get("model_loaded"):
                    raise RuntimeError(f"Qwen endpoint is not ready: {args.qwen_url}: {health}")
                LOGGER.info("Using persistent Qwen3-VL service at %s", args.qwen_url)
                agent = QwenHTTPAgent(args.qwen_url, args.qwen_timeout)
                qwen_service = {
                    key: health.get(key) for key in (
                        "model", "physical_gpu", "dtype", "min_pixels",
                        "max_pixels", "max_new_tokens",
                    )
                }
                qwen_identity = json.dumps(qwen_service, sort_keys=True)
            else:
                agent = QwenAgent(
                    args.qwen_model, args.qwen_device, args.qwen_dtype, args.qwen_max_new_tokens,
                    args.qwen_min_pixels, args.qwen_max_pixels,
                )
        plans = build_plans(media_items, categories, args, agent, work_dir)
        frames = prepare_frames(media_items, plans, categories, output, args)
        if not frames:
            raise RuntimeError("Planning produced no frames")
        LOGGER.info("Prepared %d frames", len(frames))

        urls = [item.strip().rstrip("/") for item in args.sam_urls.split(",") if item.strip()]
        if not urls:
            raise ValueError("At least one SAM URL is required")
        if not args.skip_health_check:
            for url in urls:
                health = http_json(endpoint_health(url), None, min(30.0, args.request_timeout))
                if health.get("status") != "ok" or not health.get("model_loaded"):
                    raise RuntimeError(f"SAM endpoint is not ready: {url}: {health}")
        workers = args.workers or len(urls)
        signature = annotation_signature(categories, urls, args)
        sam_progress_path = work_dir / "sam_progress.jsonl"
        records_by_path = load_sam_progress(sam_progress_path, signature) if args.resume else {}
        pending = [
            frame for frame in frames
            if str(frame.path) not in records_by_path
            or tuple(records_by_path[str(frame.path)].get("planned_categories", ())) != frame.categories
        ]
        progress_file = sam_progress_path.open("a" if args.resume else "w", encoding="utf-8")
        failed_records: list[dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(annotate_one, frame, categories, urls[index % len(urls)], args): frame
                for index, frame in enumerate(pending)
            }
            for index, future in enumerate(concurrent.futures.as_completed(future_map), start=1):
                frame = future_map[future]
                try:
                    record = future.result()
                except Exception as error:
                    record = {
                        "source_key": frame.source_key, "source_path": frame.source_path,
                        "media_type": frame.media_type, "frame_index": frame.frame_index,
                        "timestamp": frame.timestamp, "image_path": str(frame.path),
                        "planned_categories": list(frame.categories),
                        "width": 0, "height": 0, "annotations": [], "errors": [str(error)],
                    }
                    failed_records.append(record)
                record["signature"] = signature
                records_by_path[str(frame.path)] = record
                progress_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                progress_file.flush()
                if index % 25 == 0 or index == len(pending):
                    LOGGER.info("SAM annotated %d/%d pending frames (%d cached)", index, len(pending), len(frames) - len(pending))
        progress_file.close()
        if failed_records:
            with (work_dir / "failed_records.json").open("w", encoding="utf-8") as handle:
                json.dump(failed_records, handle, ensure_ascii=False, indent=2)
            raise RuntimeError(f"{len(failed_records)} SAM frame(s) failed; fix the service and rerun with --resume")
        records = [records_by_path[str(frame.path)] for frame in frames]
        records.sort(key=lambda item: (item["source_key"], item["frame_index"] if item["frame_index"] is not None else -1))
        assign_tracks(records, match_iou=args.track_iou, max_frame_gap=args.track_max_gap)
        if args.qwen_review:
            assert agent is not None
            qwen_review_records(
                records, categories, agent, work_dir,
                model_name=qwen_identity, resume=args.resume,
            )
        quality_counts = quality_gate(
            records, accept_score=args.accept_score, review_score=args.review_score,
            accept_prompt_agreement=args.accept_prompt_agreement, min_track_length=args.min_track_length,
            accept_single_frame_score=args.accept_single_frame_score, require_vlm_accept=args.require_vlm_accept,
        )
        write_jsonl(work_dir / "annotation_records.jsonl", records)
        review_frames = write_review_artifacts(records, output, args.review_overlays)
        source_splits = split_sources(records, parse_splits(args.splits), args.seed)
        split_stats = write_coco(records, categories, output, source_splits, args)
        manifest = {
            "pipeline": "EVT-Label", "version": "1.0", "input": str(args.input.resolve()),
            "planner": args.planner, "qwen_model": args.qwen_model,
            "qwen_url": args.qwen_url, "qwen_service": qwen_service,
            "qwen_review": args.qwen_review, "sam_urls": urls,
            "categories": [asdict(item) for item in categories], "media": len(media_items),
            "frames_processed": len(records), "quality": quality_counts,
            "review_frames": review_frames, "splits": split_stats, "source_splits": source_splits,
            "warning": "Model-generated pseudo-labels require Gold Set evaluation and human review.",
        }
        with (output / "dataset_manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
        LOGGER.info("Completed: %s", output)
        LOGGER.info("Quality: %s | COCO: %s", quality_counts, split_stats)
        return 0
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError, urllib.error.URLError) as error:
        LOGGER.error("%s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
