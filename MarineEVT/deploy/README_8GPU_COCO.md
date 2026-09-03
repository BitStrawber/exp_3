# MarineEVT 8×RTX 3090 COCO 自动标注部署

本方案使用一张 GPU 一个 SAM3 进程。默认启动 8 个服务，端口为
`127.0.0.1:8111` 至 `127.0.0.1:8118`，数据生成器以 8 个并发请求分发帧。

## 1. 克隆并进入项目

```bash
mkdir -p ~/xcx
cd ~/xcx
git clone https://github.com/BitStrawber/exp_3.git
cd ~/xcx/exp_3/MarineEVT
```

## 2. 创建环境

先确认 `nvidia-smi` 能识别 8 张显卡，再运行：

```bash
bash scripts/setup_label_env.sh
conda activate marineevt-label
```

该脚本创建 Python 3.12 环境，安装 PyTorch 2.7.0/CUDA 12.6、SAM3、
FastAPI、OpenCV、Pillow 和 pycocotools。不要为纯标注任务安装仓库根目录
的完整 `requirements.txt`；其中包含 VERL/Ray/vLLM 训练依赖。

## 3. 下载 SAM3 权重

默认从 GitCode 的 `hf_mirrors/facebook/sam3` 镜像下载。权重由 Git LFS
管理，先安装 Git LFS：

```bash
sudo apt-get update
sudo apt-get install -y git-lfs
bash scripts/download_sam3_gitcode.sh ~/xcx/models/sam3
```

脚本使用镜像地址
`https://gitcode.com/hf_mirrors/facebook/sam3.git`，跳过仓库中的
`model.safetensors`，只拉取本部署需要的 `sam3.pt`，并校验文件大小与
SHA-256。下载完成后的默认路径为 `~/xcx/models/sam3/sam3.pt`。

如需将模型放到大容量磁盘，直接把目标目录作为第一个参数：

```bash
bash scripts/download_sam3_gitcode.sh /data/fuping/marineevt/models/sam3
```

然后相应修改配置文件中的 `MODEL_ROOT` 或 `SAM3_CHECKPOINT_PATH`。首次使用
前仍应阅读镜像仓库携带的 `LICENSE`，并遵守模型许可。8 个 Worker 共享磁盘
上的同一份权重，不要分别下载 8 份。

## 4. 配置路径

```bash
mkdir -p ~/xcx/configs ~/xcx/data ~/xcx/logs ~/xcx/run
cp deploy/marineevt.env.example ~/xcx/configs/marineevt.env
nano ~/xcx/configs/marineevt.env
chmod 600 ~/xcx/configs/marineevt.env
```

所有输入视频、抽取帧和输出数据集应放在 `SAM3_ALLOWED_DATA_ROOT` 下。
SAM3 服务只允许读取这个目录中的图片路径。

## 5. 启动和检查 8 个服务

```bash
conda activate marineevt-label
cd ~/xcx/exp_3/MarineEVT
chmod +x scripts/*.sh
bash scripts/start_sam3_workers.sh
```

模型加载可能需要数分钟。重复运行健康检查，直到 8 个 Worker 全部 READY：

```bash
bash scripts/check_sam3_workers.sh
nvidia-smi
```

查看单卡日志：

```bash
tail -f ~/xcx/logs/sam3/gpu_0.log
```

停止服务：

```bash
bash scripts/stop_sam3_workers.sh
```

不要用 `uvicorn --workers 8` 启动单个端口；那会让多个进程争抢同一张卡。

## 6. 准备类别配置

```bash
cp scripts/coco_categories.example.json ~/xcx/configs/coco_categories.json
nano ~/xcx/configs/coco_categories.json
```

每个类别支持独立阈值：

```json
{
  "id": 1,
  "name": "fish",
  "prompt": "fish",
  "supercategory": "marine animal",
  "min_score": 0.5
}
```

示例阈值仅用于冒烟测试。正式阈值应使用人工 Gold Set 校准。

## 7. 小规模测试

把视频放到 `~/xcx/data/incoming/job_001/`，然后运行：

```bash
mkdir -p ~/xcx/data/incoming/job_001

bash scripts/run_coco_8gpu.sh \
  ~/xcx/data/incoming/job_001 \
  ~/xcx/data/jobs/job_001_smoke \
  ~/xcx/configs/coco_categories.json \
  --sample-every-seconds 5 \
  --max-frames-per-video 5 \
  --limit 40 \
  --min-score 0.5 \
  --no-include-masks
```

## 8. 正式生成

```bash
bash scripts/run_coco_8gpu.sh \
  ~/xcx/data/incoming/job_001 \
  ~/xcx/data/datasets/job_001_v1 \
  ~/xcx/configs/coco_categories.json \
  --sample-every-seconds 1 \
  --max-frames-per-video 1000 \
  --min-score 0.5 \
  --min-box-width 8 \
  --min-box-height 8 \
  --min-area-ratio 0.0001 \
  --max-area-ratio 0.9 \
  --nms-iou 0.7 \
  --splits 0.8,0.1,0.1 \
  --include-empty \
  --resume
```

如需实例分割标注，增加 `--include-masks`。不启用时只生成目标检测框，
网络传输和 JSON 文件都会更小。

## 9. 输出

```text
job_001_v1/
├── images/all/
├── annotations/
│   ├── instances_train.json
│   ├── instances_val.json
│   └── instances_test.json
├── work/
│   ├── progress.jsonl
│   └── failed_records.json     # 仅失败时产生
└── dataset_manifest.json
```

Pipeline 按源视频划分 train/val/test，避免相邻帧跨集合泄漏。运行中断后使用
相同参数和 `--resume` 重跑；成功记录会复用，失败帧会再次提交。

## 10. COCO 验证

```bash
python - <<'PY'
from pathlib import Path
from pycocotools.coco import COCO

root = Path.home() / "xcx/data/datasets/job_001_v1"
for split in ("train", "val", "test"):
    coco = COCO(root / "annotations" / f"instances_{split}.json")
    print(split, "images=", len(coco.imgs), "annotations=", len(coco.anns), "categories=", len(coco.cats))
PY
```

自动结果属于伪标签。正式使用前应可视化抽检，制作人工 Gold Set，按类别校准
阈值，并对验证集和测试集进行人工复核。
