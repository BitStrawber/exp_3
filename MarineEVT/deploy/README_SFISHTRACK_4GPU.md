# SFISHTRACK 单视频四卡冒烟测试

本流程只使用物理 GPU `4,5,6,7`：GPU 4 加载 Qwen3-VL，GPU 5–7 各运行一个
SAM3 Worker。默认从一段 SFISHTRACK 视频中最多处理 40 帧，输出 COCO 检测框和
实例分割伪标签，适合先验证数据生成能力，不代表完整数据集性能。

SFISHTRACK 官方以约 20 GB 的单个 Google Drive 归档发布。下载脚本支持断点续传，
并只从归档中解压一段视频、对应 COCO 标注和元数据，以减少解压后的空间占用。

## 1. 更新代码与环境

```bash
cd ~/xcx/exp_3
git pull origin main
cd MarineEVT

conda activate marineevt-label
python -m pip install -r deploy/marineevt-label.requirements.txt
```

如果环境还没有创建，并且大盘路径是 `~/xcx`：

```bash
cd ~/xcx/exp_3/MarineEVT
export CONDA_PKGS_DIRS="$HOME/xcx/.conda/pkgs"
export PIP_CACHE_DIR="$HOME/xcx/.cache/pip"
bash scripts/setup_label_env.sh
conda activate marineevt-label
```

整个环境不需要 `sudo`。

## 2. 配置 GPU 5、6、7 上的 SAM3 Worker

```bash
mkdir -p ~/xcx/configs ~/xcx/logs ~/xcx/run ~/xcx/data
cp deploy/marineevt.env.example ~/xcx/configs/marineevt.env
```

打开 `~/xcx/configs/marineevt.env`，确认至少包含：

```bash
PROJECT_ROOT="${HOME}/xcx/exp_3/MarineEVT"
MODEL_ROOT="${HOME}/xcx/models"
DATA_ROOT="${HOME}/xcx/data"
LOG_ROOT="${HOME}/xcx/logs"
RUN_ROOT="${HOME}/xcx/run"

SAM3_CHECKPOINT_PATH="${MODEL_ROOT}/sam3/sam3.pt"
SAM3_BPE_PATH="${PROJECT_ROOT}/evt_r1/tools/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
SAM3_ALLOWED_DATA_ROOT="${DATA_ROOT}"
SAM3_DEVICE="cuda"
SAM3_BASE_CONFIDENCE="0.05"
SAM3_BASE_PORT="8111"
SAM3_GPU_IDS="5,6,7"
SAM3_BIND_HOST="127.0.0.1"
```

检查四张卡当前是否空闲：

```bash
nvidia-smi -i 4,5,6,7
```

启动并检查服务：

```bash
cd ~/xcx/exp_3/MarineEVT
bash scripts/stop_sam3_workers.sh || true
bash scripts/start_sam3_workers.sh
watch -n 5 bash scripts/check_sam3_workers.sh
```

出现三个 `READY` 后按 `Ctrl+C` 退出监视。

## 3. 下载并只准备一个视频

先确认下载盘至少有约 25 GB 可用空间：

```bash
df -h ~/xcx
```

下载官方归档并选择排序后的第一段视频：

```bash
cd ~/xcx/exp_3/MarineEVT
conda activate marineevt-label
bash scripts/download_sfishtrack_one.sh "$HOME/xcx/data/sfishtrack_smoke"
```

下载中断后重复同一条命令即可续传。要指定视频名，可传第二个参数：

```bash
bash scripts/download_sfishtrack_one.sh \
  "$HOME/xcx/data/sfishtrack_smoke" \
  "video_001.mp4"
```

准备成功后确认文件：

```bash
find ~/xcx/data/sfishtrack_smoke -maxdepth 2 -type f -printf '%p %k KB\n'
```

## 4. 下载或指定 Qwen3-VL

把 Hugging Face 缓存放在大容量磁盘：

```bash
export HF_HOME="$HOME/xcx/models/huggingface"
mkdir -p "$HF_HOME"
```

首次运行可以直接使用 `Qwen/Qwen3-VL-8B-Instruct`，Transformers 会自动下载。
如果服务器不能访问 Hugging Face，则先从可访问的镜像下载到
`$HOME/xcx/models/Qwen3-VL-8B-Instruct`，运行时把模型参数换成本地路径。

## 5. 运行单视频测试

```bash
cd ~/xcx/exp_3/MarineEVT
conda activate marineevt-label
export HF_HOME="$HOME/xcx/models/huggingface"

bash scripts/run_evt_label_sfishtrack_4gpu.sh \
  "$HOME/xcx/data/sfishtrack_smoke/input" \
  "$HOME/xcx/data/generated/sfishtrack_evt_smoke" \
  "Qwen/Qwen3-VL-8B-Instruct"
```

这个命令固定使用：

- 物理 GPU 4：Qwen3-VL；
- 物理 GPU 5、6、7：SAM3服务；
- 最多40个待标注帧；
- 单类别 `fish`；
- Qwen时间段规划和候选框复核；
- Qwen时间规划输出格式异常时自动回退到规则抽帧，但候选框复核仍保持严格；
- SAM3检测框和实例掩码；
- 严格质量门控，只把 `ACCEPT` 标注写入训练集。

若第一次只想极小规模测试，可在命令末尾覆盖帧数：

```bash
bash scripts/run_evt_label_sfishtrack_4gpu.sh \
  "$HOME/xcx/data/sfishtrack_smoke/input" \
  "$HOME/xcx/data/generated/sfishtrack_evt_smoke_8frames" \
  "Qwen/Qwen3-VL-8B-Instruct" \
  --limit 8 \
  --review-overlays 8
```

重复运行同一输出目录会从 `sam_progress.jsonl` 和
`qwen_review_progress.jsonl` 断点续跑。

## 6. 检查结果

```bash
python -m json.tool \
  ~/xcx/data/generated/sfishtrack_evt_smoke/dataset_manifest.json

python -m json.tool \
  ~/xcx/data/generated/sfishtrack_evt_smoke/annotations/instances_train.json \
  >/dev/null && echo "COCO JSON 语法正常"

find ~/xcx/data/generated/sfishtrack_evt_smoke/review/overlays \
  -maxdepth 1 -type f | head
```

核心结果是：

```text
annotations/instances_train.json   自动接受的COCO标注
review/review_queue.jsonl          需要人工复核的候选
review/overlays/                   可视化候选框
work/annotation_records.jsonl      包含接受/复核/拒绝原因的完整审计记录
dataset_manifest.json              本次运行的配置和数量统计
```

如果生成结果确认正常，而且磁盘紧张，可删除已下载的完整归档；单视频子集和输出不会受影响：

```bash
rm -- "$HOME/xcx/downloads/sfishtrack/SFISHTRACK.zip"
```

## 7. 常见问题

- 三个SAM服务必须先全部显示 `READY`，再运行pipeline。
- 不要在启动pipeline的父shell中全局设置 `CUDA_VISIBLE_DEVICES=4`；运行脚本会只对Qwen进程设置。
- `No space left on device` 时先检查 `df -h ~/xcx`、Conda包缓存和pip缓存。
- 单个视频只有一个数据源，所以本次输出固定全部写入 `instances_train.json`，验证集和测试集为空是正常的。
- 自动标注是伪标签；本次先验证链路是否成功，之后再增加官方COCO真值的定量评价。
