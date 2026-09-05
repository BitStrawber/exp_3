"""Pure, testable components for the EVT-Label data generation pipeline."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class Category:
    id: int
    name: str
    prompts: tuple[str, ...]
    supercategory: str = "marine animal"
    min_score: float | None = None


def load_categories(path: Path) -> list[Category]:
    with path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    raw = document.get("categories", document) if isinstance(document, dict) else document
    if not isinstance(raw, list) or not raw:
        raise ValueError("Category file must contain a non-empty categories list")

    categories: list[Category] = []
    seen_ids: set[int] = set()
    seen_names: set[str] = set()
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Category {index} must be an object")
        category_id = int(item.get("id", index))
        name = str(item.get("name", "")).strip()
        raw_prompts = item.get("prompts", item.get("positive_prompts", item.get("prompt", name)))
        prompts = [raw_prompts] if isinstance(raw_prompts, str) else list(raw_prompts or [])
        prompts = [str(prompt).strip() for prompt in prompts if str(prompt).strip()]
        min_score = item.get("min_score")
        min_score = float(min_score) if min_score is not None else None
        if category_id <= 0 or category_id in seen_ids:
            raise ValueError(f"Category id must be a unique positive integer: {category_id}")
        if not name or name.casefold() in seen_names:
            raise ValueError(f"Category name must be non-empty and unique: {name!r}")
        if not prompts:
            raise ValueError(f"Category {name!r} has no prompts")
        if min_score is not None and not 0 <= min_score <= 1:
            raise ValueError(f"min_score for {name!r} must be between 0 and 1")
        seen_ids.add(category_id)
        seen_names.add(name.casefold())
        categories.append(Category(
            id=category_id,
            name=name,
            prompts=tuple(dict.fromkeys(prompts)),
            supercategory=str(item.get("supercategory", "marine animal")).strip(),
            min_score=min_score,
        ))
    return categories


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _sanitize_detection(
    raw: dict[str, Any],
    category_map: dict[int, Category],
    width: int,
    height: int,
    global_min_score: float,
    min_box_size: float,
    min_area_ratio: float,
    max_area_ratio: float,
) -> dict[str, Any] | None:
    try:
        category_id = int(raw["category_id"])
        category = category_map[category_id]
        score = float(raw.get("score", 0.0))
        coords = raw.get("bbox_xyxy", raw.get("bbox"))
        if not isinstance(coords, (list, tuple)) or len(coords) != 4:
            return None
        x1, y1, x2, y2 = map(float, coords)
    except (KeyError, TypeError, ValueError):
        return None
    threshold = category.min_score if category.min_score is not None else global_min_score
    if not math.isfinite(score) or score < threshold:
        return None
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        return None
    x1, x2 = sorted((max(0.0, min(x1, width)), max(0.0, min(x2, width))))
    y1, y2 = sorted((max(0.0, min(y1, height)), max(0.0, min(y2, height))))
    bw, bh = x2 - x1, y2 - y1
    area_ratio = bw * bh / float(width * height) if width > 0 and height > 0 else 0.0
    if bw < min_box_size or bh < min_box_size:
        return None
    if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
        return None
    result: dict[str, Any] = {
        "category_id": category_id,
        "xyxy": [x1, y1, x2, y2],
        "score": score,
        "prompt": str(raw.get("prompt", "")).strip(),
    }
    if isinstance(raw.get("segmentation"), dict):
        result["segmentation"] = raw["segmentation"]
        result["mask_area"] = int(raw.get("mask_area", 0))
    return result


def fuse_detections(
    raw_detections: Iterable[dict[str, Any]],
    categories: list[Category],
    width: int,
    height: int,
    *,
    global_min_score: float = 0.35,
    cluster_iou: float = 0.55,
    nms_iou: float = 0.75,
    min_box_size: float = 4.0,
    min_area_ratio: float = 0.0001,
    max_area_ratio: float = 0.95,
) -> list[dict[str, Any]]:
    """Fuse repeated text-prompt detections and preserve agreement evidence."""
    category_map = {category.id: category for category in categories}
    grouped: dict[int, list[dict[str, Any]]] = {category.id: [] for category in categories}
    for raw in raw_detections:
        item = _sanitize_detection(
            raw, category_map, width, height, global_min_score, min_box_size,
            min_area_ratio, max_area_ratio,
        )
        if item is not None:
            grouped[item["category_id"]].append(item)

    fused: list[dict[str, Any]] = []
    for category_id, candidates in grouped.items():
        candidates.sort(key=lambda item: item["score"], reverse=True)
        clusters: list[list[dict[str, Any]]] = []
        for candidate in candidates:
            target = next(
                (cluster for cluster in clusters if box_iou(candidate["xyxy"], cluster[0]["xyxy"]) >= cluster_iou),
                None,
            )
            if target is None:
                clusters.append([candidate])
            else:
                target.append(candidate)

        category = category_map[category_id]
        category_results: list[dict[str, Any]] = []
        for cluster in clusters:
            best = max(cluster, key=lambda item: item["score"])
            prompts = sorted({item["prompt"] for item in cluster if item["prompt"]})
            total_prompts = max(1, len(category.prompts))
            item = dict(best)
            item["supporting_prompts"] = prompts
            item["prompt_support"] = len(prompts)
            item["prompt_agreement"] = min(1.0, len(prompts) / total_prompts)
            item["mean_score"] = sum(value["score"] for value in cluster) / len(cluster)
            item["candidate_count"] = len(cluster)
            category_results.append(item)

        category_results.sort(key=lambda item: (item["score"], item["prompt_agreement"]), reverse=True)
        kept: list[dict[str, Any]] = []
        for candidate in category_results:
            if all(box_iou(candidate["xyxy"], existing["xyxy"]) < nms_iou for existing in kept):
                kept.append(candidate)
        fused.extend(kept)
    return fused


def assign_tracks(
    records: list[dict[str, Any]],
    *,
    match_iou: float = 0.30,
    max_frame_gap: int = 2,
) -> dict[int, int]:
    """Assign deterministic short-term IoU tracks independently for each video/class."""
    next_track_id = 1
    active: dict[tuple[str, int], list[dict[str, Any]]] = {}
    lengths: dict[int, int] = {}

    for record_index, record in enumerate(records):
        if record.get("media_type") != "video":
            for annotation in record["annotations"]:
                annotation["track_id"] = None
                annotation["track_length"] = 1
            continue
        source = str(record["source_key"])
        current_keys: set[tuple[str, int]] = set()
        used_track_ids: set[int] = set()
        for annotation in sorted(record["annotations"], key=lambda item: (item["category_id"], -item["score"])):
            key = (source, int(annotation["category_id"]))
            current_keys.add(key)
            candidates = active.get(key, [])
            best = None
            best_iou = match_iou
            for track in candidates:
                if int(track["track_id"]) in used_track_ids:
                    continue
                gap = record_index - track["last_record_index"]
                if gap <= 0 or gap > max_frame_gap:
                    continue
                overlap = box_iou(annotation["xyxy"], track["xyxy"])
                if overlap >= best_iou:
                    best, best_iou = track, overlap
            if best is None:
                track_id = next_track_id
                next_track_id += 1
                best = {"track_id": track_id, "last_record_index": record_index, "xyxy": annotation["xyxy"]}
                candidates.append(best)
                active[key] = candidates
                lengths[track_id] = 0
            track_id = int(best["track_id"])
            used_track_ids.add(track_id)
            annotation["track_id"] = track_id
            annotation["track_match_iou"] = best_iou if lengths[track_id] else None
            best["last_record_index"] = record_index
            best["xyxy"] = annotation["xyxy"]
            lengths[track_id] += 1

        for key in current_keys:
            active[key] = [
                track for track in active[key]
                if record_index - track["last_record_index"] <= max_frame_gap
            ]

    for record in records:
        for annotation in record["annotations"]:
            track_id = annotation.get("track_id")
            annotation["track_length"] = lengths.get(track_id, 1) if track_id is not None else 1
    return lengths


def quality_gate(
    records: list[dict[str, Any]],
    *,
    accept_score: float = 0.65,
    review_score: float = 0.40,
    accept_prompt_agreement: float = 0.34,
    min_track_length: int = 2,
    accept_single_frame_score: float = 0.85,
    require_vlm_accept: bool = False,
) -> dict[str, int]:
    """Classify annotations as ACCEPT, REVIEW, or REJECT with audit reasons."""
    counts = {"ACCEPT": 0, "REVIEW": 0, "REJECT": 0}
    for record in records:
        is_video = record.get("media_type") == "video"
        for annotation in record["annotations"]:
            score = float(annotation["score"])
            agreement = float(annotation.get("prompt_agreement", 1.0))
            track_length = int(annotation.get("track_length", 1))
            vlm_is_target = annotation.get("vlm_is_target")
            reasons: list[str] = []
            spatial_ok = score >= accept_score and agreement >= accept_prompt_agreement
            temporal_ok = not is_video or track_length >= min_track_length or score >= accept_single_frame_score
            vlm_ok = vlm_is_target is True or (vlm_is_target is None and not require_vlm_accept)
            if vlm_is_target is False:
                decision = "REJECT"
                reasons.append("vlm_rejected")
            elif spatial_ok and temporal_ok and vlm_ok:
                decision = "ACCEPT"
                reasons.append("spatial_confidence")
                reasons.append("temporal_support" if is_video and track_length >= min_track_length else "high_single_frame_confidence")
            elif score >= review_score:
                decision = "REVIEW"
                if score < accept_score:
                    reasons.append("borderline_score")
                if agreement < accept_prompt_agreement:
                    reasons.append("weak_prompt_agreement")
                if is_video and track_length < min_track_length:
                    reasons.append("weak_temporal_support")
                if require_vlm_accept and vlm_is_target is None:
                    reasons.append("missing_vlm_review")
            else:
                decision = "REJECT"
                reasons.append("low_score")
            annotation["quality_decision"] = decision
            annotation["quality_reasons"] = reasons
            counts[decision] += 1
    return counts
