# MarineEVT persistent 8-GPU inference deployment

This profile reserves physical GPU 0 for one persistent Qwen3-VL-8B service and
uses physical GPUs 1-7 for seven independent SAM3 workers.

## GPU layout

| Physical GPU | Process | Endpoint |
| --- | --- | --- |
| 0 | Qwen3-VL-8B | `127.0.0.1:8100` |
| 1 | SAM3 worker 0 | `127.0.0.1:8111` |
| 2 | SAM3 worker 1 | `127.0.0.1:8112` |
| 3 | SAM3 worker 2 | `127.0.0.1:8113` |
| 4 | SAM3 worker 3 | `127.0.0.1:8114` |
| 5 | SAM3 worker 4 | `127.0.0.1:8115` |
| 6 | SAM3 worker 5 | `127.0.0.1:8116` |
| 7 | SAM3 worker 6 | `127.0.0.1:8117` |

## Configuration

Copy `deploy/marineevt.env.example` to `~/xcx/configs/marineevt.env` and verify
the local model and data paths. The important service settings are:

```bash
QWEN_MODEL_PATH="${HOME}/xcx/models/Qwen3-VL-8B-Instruct"
QWEN_ALLOWED_DATA_ROOT="${HOME}/xcx/data"
QWEN_GPU_ID="0"
QWEN_PORT="8100"
SAM3_GPU_IDS="1,2,3,4,5,6,7"
SAM3_BASE_PORT="8111"
```

## Start and check services

Stop an older three-worker deployment before changing the GPU map:

```bash
bash scripts/stop_marineevt_services.sh || true
bash scripts/start_marineevt_8gpu.sh
watch -n 5 'bash scripts/check_qwen_service.sh; bash scripts/check_sam3_workers.sh'
```

The first Qwen service startup loads all four local safetensor shards. The
health endpoint becomes available only after loading finishes.

## Run a full input-directory build

```bash
bash scripts/run_evt_label_sfishtrack_8gpu_full.sh \
  "${HOME}/xcx/data/sfishtrack_smoke/input" \
  "${HOME}/xcx/data/generated/sfishtrack_evt_8gpu_full"
```

`full` means every input media file and every frame selected by the event
planner, with no global or per-video frame cap. It does not turn a one-video
download into the complete upstream SFISHTRACK release. The command resumes
from frame-level SAM and Qwen caches after a failure.

## Category routing

The categories file is the stable COCO taxonomy. Qwen may select a subset of
those categories for each planned time segment. Each extracted frame stores the
union of categories selected by all overlapping segments, and only prompts for
that subset are sent to SAM3. Qwen cannot silently invent a new COCO category.

The supplied SFISHTRACK profile contains only `fish`, so category routing is
observable only after more allowed classes are added to the categories file.

## Stop services

```bash
bash scripts/stop_marineevt_services.sh
```
