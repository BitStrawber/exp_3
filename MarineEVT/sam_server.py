"""Dedicated, one-GPU-per-process SAM3 HTTP service for dataset generation."""

from __future__ import annotations

import os
import threading
import time
from typing import Literal

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from evt_r1.tools.call_sam3 import (
    ALLOWED_DATA_ROOT,
    BASE_CONFIDENCE,
    BPE_PATH,
    CHECKPOINT_PATH,
    RUNTIME_DEVICE,
    call_sam,
    detect_image,
)


class PromptSpec(BaseModel):
    category_id: int = Field(gt=0)
    text: str = Field(min_length=1, max_length=256)


class DetectRequest(BaseModel):
    image_path: str
    prompts: list[PromptSpec] = Field(min_length=1, max_length=512)
    ground_type: Literal["all", "highest"] = "all"
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    include_masks: bool = False


class LegacyRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=256)
    image_paths: list[str] = Field(min_length=1, max_length=128)
    ground_type: Literal["all", "highest"] | None = "highest"


app = FastAPI(title="MarineEVT SAM3 Service", version="1.0.0")
inference_lock = threading.Lock()
started_at = time.time()


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": True,
        "device": RUNTIME_DEVICE,
        "physical_gpu": os.getenv("SAM3_PHYSICAL_GPU", os.getenv("CUDA_VISIBLE_DEVICES", "unknown")),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "checkpoint": str(CHECKPOINT_PATH) if CHECKPOINT_PATH else "huggingface:facebook/sam3",
        "bpe_path": str(BPE_PATH),
        "base_confidence": BASE_CONFIDENCE,
        "allowed_data_root": str(ALLOWED_DATA_ROOT) if ALLOWED_DATA_ROOT else None,
        "uptime_seconds": round(time.time() - started_at, 3),
    }


@app.post("/v1/detect")
def detect(request: DetectRequest) -> dict:
    try:
        with inference_lock:
            result = detect_image(
                request.image_path,
                [prompt.model_dump() for prompt in request.prompts],
                ground_type=request.ground_type,
                min_score=request.min_score,
                include_masks=request.include_masks,
            )
        return {"result": result}
    except (FileNotFoundError, PermissionError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.post("/sam")
def legacy_sam(request: LegacyRequest) -> dict:
    try:
        with inference_lock:
            result = call_sam(
                {"prompt": request.prompt, "ground_type": request.ground_type},
                request.image_paths,
            )
        return {"result": result}
    except (FileNotFoundError, PermissionError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
