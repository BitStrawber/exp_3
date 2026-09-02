"""Configurable SAM3 inference used by MarineEVT services and agents."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

import torch
from PIL import Image

from evt_r1.tools.sam3.sam3 import build_sam3_image_model
from evt_r1.tools.sam3.sam3.model.sam3_image_processor import Sam3Processor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BPE_PATH = PROJECT_ROOT / "evt_r1" / "tools" / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
RUNTIME_DEVICE = os.getenv("SAM3_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
BPE_PATH = Path(os.getenv("SAM3_BPE_PATH", str(DEFAULT_BPE_PATH))).expanduser().resolve()
CHECKPOINT_VALUE = os.getenv("SAM3_CHECKPOINT_PATH")
CHECKPOINT_PATH = Path(CHECKPOINT_VALUE).expanduser().resolve() if CHECKPOINT_VALUE else None
BASE_CONFIDENCE = float(os.getenv("SAM3_BASE_CONFIDENCE", "0.05"))
ALLOWED_DATA_VALUE = os.getenv("SAM3_ALLOWED_DATA_ROOT")
ALLOWED_DATA_ROOT = Path(ALLOWED_DATA_VALUE).expanduser().resolve() if ALLOWED_DATA_VALUE else None


def _validate_configuration() -> None:
    if not BPE_PATH.is_file():
        raise FileNotFoundError(f"SAM3 BPE vocabulary not found: {BPE_PATH}")
    if CHECKPOINT_PATH is not None and not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"SAM3 checkpoint not found: {CHECKPOINT_PATH}")
    if not 0.0 <= BASE_CONFIDENCE <= 1.0:
        raise ValueError("SAM3_BASE_CONFIDENCE must be between 0 and 1")


_validate_configuration()

# Pre-download and configure a local checkpoint for multi-worker deployments.
# Without SAM3_CHECKPOINT_PATH every worker may try to download from Hugging Face.
sam_model = build_sam3_image_model(
    bpe_path=str(BPE_PATH),
    checkpoint_path=str(CHECKPOINT_PATH) if CHECKPOINT_PATH else None,
    load_from_HF=CHECKPOINT_PATH is None,
    device=RUNTIME_DEVICE,
)
sam_processor = Sam3Processor(sam_model, confidence_threshold=BASE_CONFIDENCE, device=RUNTIME_DEVICE)


def _resolve_image_path(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    if ALLOWED_DATA_ROOT is not None:
        try:
            path.relative_to(ALLOWED_DATA_ROOT)
        except ValueError as error:
            raise PermissionError(f"Image path is outside SAM3_ALLOWED_DATA_ROOT: {path}") from error
    return path


def _as_float_list(tensor: Any) -> list[float]:
    if hasattr(tensor, "detach"):
        tensor = tensor.detach().cpu().flatten().tolist()
    return [float(value) for value in tensor]


def _uncompressed_coco_rle(mask: Any) -> dict[str, Any]:
    """Encode a binary mask as JSON-safe uncompressed COCO RLE."""
    if hasattr(mask, "detach"):
        mask = mask.detach().to(device="cpu", dtype=torch.uint8).squeeze().numpy()
    height, width = mask.shape
    flat = mask.flatten(order="F")
    counts: list[int] = []
    previous = 0
    run_length = 0
    for value in flat:
        current = 1 if int(value) else 0
        if current == previous:
            run_length += 1
        else:
            counts.append(run_length)
            run_length = 1
            previous = current
    counts.append(run_length)
    return {"size": [int(height), int(width)], "counts": counts}


def _normalize_prompts(prompts: Iterable[str | dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, prompt in enumerate(prompts, start=1):
        if isinstance(prompt, str):
            normalized.append({"category_id": index, "text": prompt})
            continue
        text = str(prompt.get("text", prompt.get("prompt", ""))).strip()
        if not text:
            raise ValueError(f"Prompt {index} has no text")
        normalized.append({"category_id": int(prompt.get("category_id", index)), "text": text})
    if not normalized:
        raise ValueError("At least one prompt is required")
    return normalized


def detect_image(
    image_path: str | Path,
    prompts: Iterable[str | dict[str, Any]],
    *,
    ground_type: str = "all",
    min_score: float = 0.0,
    include_masks: bool = False,
) -> dict[str, Any]:
    """Detect all requested text categories while encoding the image once."""
    if ground_type not in {"all", "highest"}:
        raise ValueError("ground_type must be 'all' or 'highest'")
    if not 0.0 <= min_score <= 1.0:
        raise ValueError("min_score must be between 0 and 1")

    resolved_path = _resolve_image_path(image_path)
    normalized_prompts = _normalize_prompts(prompts)
    detections: list[dict[str, Any]] = []
    autocast_device = RUNTIME_DEVICE.split(":", maxsplit=1)[0]

    with Image.open(resolved_path) as opened_image:
        image = opened_image.convert("RGB")
        width, height = image.size
        with torch.amp.autocast(device_type=autocast_device, enabled=autocast_device == "cuda"):
            state = sam_processor.set_image(image)
            for prompt in normalized_prompts:
                sam_processor.reset_all_prompts(state)
                results = sam_processor.set_text_prompt(prompt=prompt["text"], state=state)
                scores = _as_float_list(results["scores"])
                candidate_indices = [index for index, score in enumerate(scores) if score >= min_score]
                if ground_type == "highest" and candidate_indices:
                    candidate_indices = [max(candidate_indices, key=scores.__getitem__)]

                for index in candidate_indices:
                    detection: dict[str, Any] = {
                        "category_id": prompt["category_id"],
                        "prompt": prompt["text"],
                        "bbox_xyxy": _as_float_list(results["boxes"][index]),
                        "score": scores[index],
                    }
                    if include_masks:
                        mask = results["masks"][index]
                        detection["segmentation"] = _uncompressed_coco_rle(mask)
                        detection["mask_area"] = int(mask.sum().item())
                    detections.append(detection)

    return {
        "image_path": str(resolved_path),
        "width": width,
        "height": height,
        "detections": detections,
    }


def call_sam(json_content: dict[str, Any], image_paths: Iterable[str]) -> dict[str, Any]:
    """Backward-compatible adapter for the original MarineEVT tool protocol."""
    content = json_content.get("function", {}).get("arguments", json_content)
    prompt = str(content["prompt"])
    ground_type = str(content.get("ground_type", "highest"))
    all_boxes: list[list[list[float]]] = []
    for image_path in image_paths:
        result = detect_image(
            image_path,
            [{"category_id": 1, "text": prompt}],
            ground_type=ground_type,
            include_masks=False,
        )
        all_boxes.append([detection["bbox_xyxy"] for detection in result["detections"]])
    return {"boxes": all_boxes}


if __name__ == "__main__":
    pass
