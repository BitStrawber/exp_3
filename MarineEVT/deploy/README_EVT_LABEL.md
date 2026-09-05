# EVT-Label：Qwen3-VL + SAM3 视频/DeepFish 自动 COCO 标注

`EVT-Label` 将 MarineEVT 的“先理解、再调用视觉工具、最后验证”思路用于数据集生成：

1. Qwen3-VL 对视频粗采样帧生成高价值时间段计划；
2. SAM3 使用每类别多个文本提示生成框和可选掩码；
3. 多提示结果按 IoU 融合，视频帧按 IoU 建立短期轨迹；
4. Qwen3-VL 对原图和编号框图进行二次审核；
5. 质量门控输出 `ACCEPT / REVIEW / REJECT`；
6. 仅将满足策略的标注写入标准 COCO 文件，并保存全部审计记录。

这是推理 Pipeline，不需要 EVT-R1、GRPO、Ray 或 VERL 权重。第一阶段直接使用
`Qwen/Qwen3-VL-8B-Instruct` 验证思路；未来拿到 EVT-R1 权重后可将相同参数替换为
本地 EVT-R1 Hugging Face 模型目录。

## 1. 显卡分配

Qwen3-VL-8B FP16 单卡推理在 24 GB 3090 上较紧，因此限制粗采样帧数和输出长度。
推荐 GPU 0 运行 Qwen，GPU 1–7 各运行一个 SAM3 Worker。在
`~/xcx/configs/marineevt.env` 中设置：

```bash
SAM3_GPU_IDS="1,2,3,4,5,6,7"
SAM3_BASE_PORT="8111"
```

重新启动并检查 Worker：

```bash
cd ~/xcx/exp_3/MarineEVT
bash scripts/stop_sam3_workers.sh
bash scripts/start_sam3_workers.sh
bash scripts/check_sam3_workers.sh
```

如果单卡 FP16 仍然显存不足，可以先将 `--qwen-dtype` 改成 `bfloat16`（显存基本不变），
或后续接入量化加载。不要同时在 GPU 0 启动 SAM3 和完整 Qwen3-VL-8B。

## 2. 准备 Qwen3-VL

联网服务器可以直接指定 Hub ID，让 Transformers 首次运行时下载：

```bash
export HF_HOME=/path/to/large_disk/huggingface
```

也可以提前下载到大容量磁盘，并把下方第三个参数改成本地目录：

```text
/data/models/Qwen3-VL-8B-Instruct
```

## 3. DeepFish 冒烟测试

DeepFish underwater 数据约有 39,766 张来自 20 个 habitat 的帧；原始任务包括分类、
点定位和少量分割标注，因此这里先统一测试单类别 `fish` COCO 检测/实例分割。官方
标注并非所有图片都有 bbox，后续评估时必须分别处理 classification、localization
和 segmentation 子集，不能把点标注直接当 bbox 真值。

先选择 20–100 张图建立小目录，例如：

```bash
mkdir -p ~/xcx/data/deepfish_smoke
# 将少量 DeepFish 水下图片复制或软链接到该目录
```

运行：

```bash
bash scripts/run_evt_label_deepfish.sh \
  ~/xcx/data/deepfish_smoke \
  ~/xcx/data/datasets/deepfish_evt_smoke \
  Qwen/Qwen3-VL-8B-Instruct \
  --limit 40 \
    --review-overlays 40
```

SAM3 结果保存在 `work/sam_progress.jsonl`，Qwen 审核结果保存在
`work/qwen_review_progress.jsonl`。相同输入和配置重新运行时默认断点续跑；更改类别、
Prompt 或关键阈值会产生新签名并重新计算。使用 `--no-resume` 可显式忽略缓存。

如果输入是图片目录，Qwen 的时间规划阶段自然跳过，但 Qwen 候选框审核仍会执行。
如果输入包含视频，则 Qwen 同时选择高价值时间段和采样频率。

只测试 Qwen 视频规划但暂不强制所有候选通过审核，可运行：

```bash
python scripts/generate_evt_label_dataset.py \
  --input ~/xcx/data/incoming/test_videos \
  --output ~/xcx/data/datasets/evt_video_smoke \
  --categories-file scripts/evt_label_categories.deepfish.json \
  --planner qwen \
  --qwen-model Qwen/Qwen3-VL-8B-Instruct \
  --qwen-device cuda:0 \
  --sam-urls http://127.0.0.1:8111/v1/detect,http://127.0.0.1:8112/v1/detect,http://127.0.0.1:8113/v1/detect,http://127.0.0.1:8114/v1/detect,http://127.0.0.1:8115/v1/detect,http://127.0.0.1:8116/v1/detect,http://127.0.0.1:8117/v1/detect \
  --workers 7 \
  --limit 40
```

## 4. 输出结构

```text
output/
├── images/all/                         # 纳入处理的图像/视频帧
├── annotations/
│   ├── instances_train.json
│   ├── instances_val.json
│   └── instances_test.json
├── review/
│   ├── review_queue.jsonl              # 人工复核队列
│   └── overlays/                       # 编号候选框可视化
├── work/
│   ├── media_index.jsonl               # 输入索引
│   ├── segment_plans.jsonl             # Qwen/规则时间计划
│   ├── annotation_records.jsonl        # 所有候选、轨迹、VLM 决策和质量决策
│   ├── planner_frames/                 # 视频粗采样帧
│   └── qwen_review/                    # Qwen 看到的编号框图
└── dataset_manifest.json               # 参数和统计摘要
```

默认策略不会把含 `REVIEW` 候选的帧写进自动训练集，避免将未确认目标误当作背景；
审核完成后应更新记录并重新导出。`score`、`track_id`、`prompt_agreement` 和
`vlm_confidence` 是额外审计字段，常见 COCO 读取器会忽略它们。

## 5. 比较实验

建议在同一批 DeepFish Gold Set 上运行两次：

- 纯 SAM3：`--planner heuristic --no-qwen-review`；
- Qwen + SAM3：`--planner heuristic --qwen-review --require-vlm-accept`。

DeepFish 图片输入没有时间规划差异，因此这组实验专门测量 Qwen 审核对误检率、召回率和
人工修正率的影响。视频输入再额外比较 Qwen 时间规划是否在减少处理帧数的同时保持目标召回。

正式结论至少报告每类别 Precision/Recall、bbox 或 mask AP、空帧误检率、审核率、
人工修正率、处理时间和 GPU 成本。所有自动结果都是伪标签，不能用模型自己的置信度代替
人工 Gold Set 评估。
