"""Persistent, single-GPU Qwen3-VL service for MarineEVT dataset generation."""

from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


MODEL_VALUE = os.getenv("QWEN_MODEL_PATH")
if not MODEL_VALUE:
    raise RuntimeError("QWEN_MODEL_PATH is required")

MODEL_PATH = Path(MODEL_VALUE).expanduser().resolve()
DEVICE = os.getenv("QWEN_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE_NAME = os.getenv("QWEN_DTYPE", "float16")
MIN_PIXELS = int(os.getenv("QWEN_MIN_PIXELS", "100352"))
MAX_PIXELS = int(os.getenv("QWEN_MAX_PIXELS", "200704"))
MAX_NEW_TOKENS = int(os.getenv("QWEN_MAX_NEW_TOKENS", "512"))
ALLOWED_ROOT_VALUE = os.getenv("QWEN_ALLOWED_DATA_ROOT")
ALLOWED_DATA_ROOT = (
    Path(ALLOWED_ROOT_VALUE).expanduser().resolve() if ALLOWED_ROOT_VALUE else None
)

if not MODEL_PATH.is_dir():
    raise FileNotFoundError(f"Qwen model directory not found: {MODEL_PATH}")
if not (MODEL_PATH / "config.json").is_file():
    raise FileNotFoundError(f"Qwen config.json not found: {MODEL_PATH / 'config.json'}")
if DTYPE_NAME not in {"float16", "bfloat16", "float32"}:
    raise ValueError("QWEN_DTYPE must be float16, bfloat16 or float32")
if MIN_PIXELS <= 0 or MAX_PIXELS < MIN_PIXELS:
    raise ValueError("Qwen pixel limits are invalid")

from transformers import AutoProcessor

try:
    from transformers import AutoModelForImageTextToText as AutoVisionModel
except ImportError:
    from transformers import AutoModelForVision2Seq as AutoVisionModel


DTYPE = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}[DTYPE_NAME]

processor = AutoProcessor.from_pretrained(
    str(MODEL_PATH),
    trust_remote_code=True,
    local_files_only=True,
    min_pixels=MIN_PIXELS,
    max_pixels=MAX_PIXELS,
)
try:
    model = AutoVisionModel.from_pretrained(
        str(MODEL_PATH),
        dtype=DTYPE,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to(DEVICE).eval()
except TypeError:
    model = AutoVisionModel.from_pretrained(
        str(MODEL_PATH),
        torch_dtype=DTYPE,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to(DEVICE).eval()


class GenerateRequest(BaseModel):
    image_paths: list[str] = Field(min_length=1, max_length=64)
    instruction: str = Field(min_length=1, max_length=50000)


def resolve_image(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    if ALLOWED_DATA_ROOT is not None:
        try:
            path.relative_to(ALLOWED_DATA_ROOT)
        except ValueError as error:
            raise PermissionError(
                f"Image path is outside QWEN_ALLOWED_DATA_ROOT: {path}"
            ) from error
    return path


def parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Qwen response is not a JSON object")
    return value


def generate_json(image_paths: list[str], instruction: str) -> dict[str, Any]:
    from PIL import Image

    paths = [resolve_image(value) for value in image_paths]
    images = []
    try:
        images = [Image.open(path).convert("RGB") for path in paths]
        content = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": instruction})
        messages = [{"role": "user", "content": content}]
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[prompt], images=images, padding=True, return_tensors="pt"
        ).to(DEVICE)
        with torch.inference_mode():
            generated = model.generate(
                **inputs, do_sample=False, max_new_tokens=MAX_NEW_TOKENS
            )
        generated = generated[:, inputs.input_ids.shape[1] :]
        raw_text = processor.batch_decode(generated, skip_special_tokens=True)[0]
        return parse_json(raw_text)
    finally:
        for image in images:
            image.close()


app = FastAPI(title="MarineEVT Qwen3-VL Service", version="1.0.0")
inference_lock = threading.Lock()
started_at = time.time()


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": True,
        "model": str(MODEL_PATH),
        "device": DEVICE,
        "physical_gpu": os.getenv("QWEN_PHYSICAL_GPU", os.getenv("CUDA_VISIBLE_DEVICES", "unknown")),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "dtype": DTYPE_NAME,
        "min_pixels": MIN_PIXELS,
        "max_pixels": MAX_PIXELS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "allowed_data_root": str(ALLOWED_DATA_ROOT) if ALLOWED_DATA_ROOT else None,
        "uptime_seconds": round(time.time() - started_at, 3),
    }


@app.post("/v1/generate-json")
def generate(request: GenerateRequest) -> dict[str, Any]:
    try:
        with inference_lock:
            result = generate_json(request.image_paths, request.instruction)
        return {"result": result}
    except (FileNotFoundError, PermissionError, ValueError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
