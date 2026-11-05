# InternVL2.5-1B ONNX + TensorRT 部署说明

本目录包含 InternVL2.5-1B 的完整 ONNX / TensorRT 部署链路脚本。

> **设计原则**：把 InternVL 拆成 **Vision Encoder (ViT + MLP1)** 与 **LLM (Qwen2.5-0.5B)** 两个模块分别导出/转 TRT，最后在 Python 生成循环里把图像特征注入到 text embedding 中。每完成一步都有独立验证脚本与量化指标。

---

## 环境要求

- CUDA >= 12.0
- Python 3.10
- 已安装项目根目录 `requirements.txt` 中的依赖，并额外安装：

```bash
/home/jovyan/work/envs/myenv/bin/python3 -m pip install \
  onnx>=1.16 \
  onnxruntime-gpu>=1.23.0 \
  tensorrt==10.7.0 \
  --extra-index-url https://pypi.nvidia.com
```

---

## 文件说明

### 导出脚本

| 脚本 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `export_vision_onnx.py` | pixel_values `(1,3,448,448)` | `internvl2_vision.onnx` | Vision Encoder (ViT + pixel_shuffle + mlp1) |
| `export_vision_trt.py` | ONNX | `internvl2_vision.trt` | Vision TensorRT engine |
| `export_llm_onnx.py` | input_ids `(1,seq_len)` | `internvl2_llm.onnx` | LLM (input_ids 模式) |
| `export_llm_trt.py` | ONNX | `internvl2_llm.trt` | LLM TensorRT engine (input_ids) |
| `export_llm_embed_onnx.py` | inputs_embeds `(1,seq_len,896)` | `internvl2_llm_embed.onnx` | LLM (inputs_embeds 模式，端到端用) |
| `export_llm_embed_trt.py` | ONNX | `internvl2_llm_embed.trt` | LLM TensorRT engine (inputs_embeds) |

### 验证脚本

| 脚本 | 对比双方 | 关键指标 |
|------|----------|----------|
| `verify_vision_onnx.py` | CPU ONNX vs CPU PyTorch | `max_abs_error < 1e-2`, `rel_error < 1e-3`, `cosine_sim > 0.9999` |
| `verify_vision_trt.py` | TRT vs PyTorch / ONNX | `rel_error < 5%`, `cosine_sim > 0.999` |
| `verify_llm_onnx.py` | CPU ONNX vs CPU PyTorch | `token_match = 100%`, greedy 序列一致 |
| `verify_llm_trt.py` | TRT vs PyTorch / ONNX | `token_match = 100%`, greedy 序列一致 |

### 端到端脚本

| 脚本 | 用途 |
|------|------|
| `run_trt_internvl.py` | 图片 + prompt -> TRT Vision -> 注入 image embedding -> TRT LLM 贪心生成 |
| `compare_torch_trt.py` | 同一张图/同一个 prompt 分别跑 PyTorch `model.chat()` 与 TRT，输出 token 匹配率与耗时 |
| `benchmark_vision_latency.py` | 单独测试 Vision Encoder (ONNX / TensorRT / PyTorch) 的推理延迟 |

---

## 一键命令（推荐）

在项目根目录执行：

```bash
# 1. 导出 ONNX（vision + LLM-input_ids + LLM-inputs_embeds）
bash run.sh deploy-internvl2-onnx

# 2. 构建 TensorRT engine
bash run.sh deploy-internvl2-trt

# 3. TRT 端到端推理
bash run.sh run-internvl2-trt --prompt "Describe this image in detail." --max_new_tokens 64

# 4. 与 PyTorch 严格对比
bash run.sh compare-internvl2-trt --max_new_tokens 32

# 5. Vision Encoder 单模块延迟测试（ONNX vs TensorRT，支持多图 batch）
bash run.sh bench-internvl2-vision-latency --onnx_provider CUDAExecutionProvider --batch_size 1
bash run.sh bench-internvl2-vision-latency --onnx_provider CUDAExecutionProvider --batch_size 6
bash run.sh bench-internvl2-vision-latency --onnx_provider CUDAExecutionProvider --batch_size 12
bash run.sh bench-internvl2-vision-latency --onnx_provider CUDAExecutionProvider --batch_size 24
```

---

## Vision Encoder 延迟测试

使用 `deploy/internvl/benchmark_vision_latency.py` 单独测量 ViT 单模块的推理延迟，方便在板端做性能摸底。脚本支持 **batch 推理**（1 / 6 / 12 / 24 张图），可一键切换 ONNX Runtime 与 TensorRT。

### 动态 batch engine 准备

一键命令默认使用动态 batch engine（覆盖 1~24 张图）：

```bash
# 导出动态 batch ONNX
/home/jovyan/work/envs/myenv/bin/python3 deploy/internvl/export_vision_onnx.py \
  --dynamic_batch \
  --output deploy/outputs/internvl2_vision_dynamic.onnx

# 构建动态 batch TensorRT engine
/home/jovyan/work/envs/myenv/bin/python3 deploy/internvl/export_vision_trt.py \
  --onnx_path deploy/outputs/internvl2_vision_dynamic.onnx \
  --output deploy/outputs/internvl2_vision_dynamic.trt \
  --min_batch 1 --opt_batch 12 --max_batch 24
```

### 单 backend 测试

```bash
cd /home/jovyan/work/vla_unified
PYTHON=/home/jovyan/work/envs/myenv/bin/python3

# ONNX Runtime (CUDA)，batch=6
$PYTHON deploy/internvl/benchmark_vision_latency.py \
  --skip_pytorch --skip_trt \
  --onnx_path deploy/outputs/internvl2_vision_dynamic.onnx \
  --onnx_provider CUDAExecutionProvider \
  --batch_size 6 \
  --image_path 20260610151154.jpg \
  --warmup 30 --iterations 100

# TensorRT，batch=12
$PYTHON deploy/internvl/benchmark_vision_latency.py \
  --skip_pytorch --skip_onnx \
  --engine_path deploy/outputs/internvl2_vision_dynamic.trt \
  --batch_size 12 \
  --image_path 20260610151154.jpg \
  --warmup 30 --iterations 100

# PyTorch eager baseline (GPU)，batch=1
$PYTHON deploy/internvl/benchmark_vision_latency.py \
  --skip_onnx --skip_trt \
  --batch_size 1 \
  --image_path 20260610151154.jpg \
  --warmup 50 --iterations 200
```

### 板端示例

板端如果只装了一种 runtime，可以只跑对应后端：

```bash
# CPU-only 板端
python3 deploy/internvl/benchmark_vision_latency.py \
  --skip_pytorch --skip_trt \
  --onnx_path /path/to/internvl2_vision_dynamic.onnx \
  --onnx_provider CPUExecutionProvider \
  --batch_size 6 \
  --image_path /path/to/board/image.jpg \
  --warmup 20 --iterations 100

# Jetson (TensorRT)
python3 deploy/internvl/benchmark_vision_latency.py \
  --skip_pytorch --skip_onnx \
  --engine_path /path/to/internvl2_vision_dynamic.trt \
  --batch_size 6 \
  --image_path /path/to/board/image.jpg \
  --warmup 20 --iterations 100
```

---

## 手动运行示例

```bash
cd /home/jovyan/work/vla_unified
PYTHON=/home/jovyan/work/envs/myenv/bin/python3

# ONNX 导出
$PYTHON deploy/internvl/export_vision_onnx.py
$PYTHON deploy/internvl/export_llm_embed_onnx.py

# TRT 建 engine
$PYTHON deploy/internvl/export_vision_trt.py
$PYTHON deploy/internvl/export_llm_embed_trt.py

# 验证
$PYTHON deploy/internvl/verify_vision_onnx.py --n_random 50
$PYTHON deploy/internvl/verify_vision_trt.py --n_random 10
$PYTHON deploy/internvl/verify_llm_onnx.py
$PYTHON deploy/internvl/verify_llm_trt.py

# 端到端
$PYTHON deploy/internvl/run_trt_internvl.py \
  --image_path 20260610151154.jpg \
  --prompt "描述这张图片" \
  --max_new_tokens 32

# 对比
$PYTHON deploy/internvl/compare_torch_trt.py \
  --image_path 20260610151154.jpg \
  --max_new_tokens 32
```

---

## 输出产物

```text
deploy/outputs/
├── internvl2_vision.onnx              # Vision ONNX，固定 batch=1 (~1.2GB，含外部权重)
├── internvl2_vision.trt               # Vision TensorRT engine，固定 batch=1 (~1.2GB，FP32)
├── internvl2_vision_dynamic.onnx      # Vision ONNX，动态 batch (1~24)
├── internvl2_vision_dynamic.trt       # Vision TensorRT engine，动态 batch (1~24) (~1.2GB，FP32)
├── internvl2_llm.onnx                 # LLM input_ids ONNX
├── internvl2_llm.trt                  # LLM input_ids TRT engine (~2.4GB，FP32)
├── internvl2_llm_embed.onnx           # LLM inputs_embeds ONNX (~1.9GB)
├── internvl2_llm_embed.trt            # LLM inputs_embeds TRT engine (~1.9GB，FP32)
└── onnx__MatMul_* / *.weight          # ONNX 外部权重文件
```

---

## 精度指标参考（RTX 4090）

### Vision ONNX
```text
max_abs_error ~1e-4, mean_abs_error ~2e-6, rel_error < 1e-3, cosine_sim > 0.9999
```

### Vision TRT
```text
vs PyTorch: rel_error < 5%, cosine_sim > 0.999
```

### LLM ONNX
```text
vs PyTorch: max_abs_error < 1e-3, token_match = 100%, greedy 序列完全一致
```

### LLM TRT
```text
vs PyTorch: rel_error < 5%, token_match = 100%, greedy 序列完全一致
```

### End-to-end
```text
PyTorch model.chat() vs TensorRT: token_match = 100%
```

---

## Vision 多图 Batch 延迟参考（RTX 4090）

使用动态 batch engine 实测，输入真实图片 `20260610151154.jpg`，`warmup=30, iterations=100`：

| Batch (图数) | ONNX-CUDA 总延迟 | ONNX-CUDA 单图延迟 | ONNX-CUDA 吞吐 | TensorRT 总延迟 | TensorRT 单图延迟 | TensorRT 吞吐 |
|-------------|-----------------|-------------------|---------------|----------------|------------------|--------------|
| 1  | 20.80 ms | 20.80 ms | 48.07 images/s | 20.87 ms | 20.87 ms | 47.91 images/s |
| 6  | 123.14 ms | 20.52 ms | 48.72 images/s | 110.45 ms | 18.41 ms | 54.32 images/s |
| 12 | 249.75 ms | 20.81 ms | 48.05 images/s | 220.74 ms | 18.40 ms | 54.36 images/s |
| 24 | 498.34 ms | 20.76 ms | 48.16 images/s | 446.97 ms | 18.62 ms | 53.70 images/s |

结论：
- ONNX-CUDA 基本随 batch 线性增长，单图延迟约 20.8 ms，吞吐约 48 images/s。
- TensorRT 在 batch≥6 时单图延迟下降到约 18.4 ms，吞吐提升到约 54 images/s，相比 batch=1 有约 **11~12%** 的收益。
- 动态 batch TRT engine 的 batch=1 性能略低于固定 batch=1 engine（~19.4 ms），这是 dynamic shape 的额外开销。

---

## 实现细节

### 1. 关闭 Flash Attention
ONNX / TensorRT 不支持 `flash_attn` 自定义算子。导出时通过 `use_flash_attn=False` 让模型退化为 eager attention。

### 2. Vision Encoder
- 默认固定 batch=1，输入 `(1, 3, 448, 448)`，输出 `visual_features (1, 256, 896)`
- 通过 `--dynamic_batch` 可导出动态 batch ONNX，配合 `--min_batch/--opt_batch/--max_batch` 构建动态 batch TRT engine
- 导出时把 `model.extract_feature()` 整体封装为子图

### 3. LLM (inputs_embeds 模式)
- 支持动态 `seq_len ∈ [1, 512]`
- 不启用 KV-cache，每次 forward 重新计算整个 prefix
- 输入为 `inputs_embeds (1, seq_len, 896)`、`attention_mask`、`position_ids`

### 4. 图像特征注入
- Prompt 经 tokenizer 得到 `input_ids`
- 用 `embed_tokens(input_ids)` 得到 text embeddings
- 找到 `input_ids == img_context_token_id` 的位置
- 将 TRT Vision 输出的 `visual_features` 替换到这些位置
- 再把融合后的 `inputs_embeds` 送入 TRT LLM

### 5. 生成策略
- 贪心解码（greedy）
- 遇到 `eos_token_id=151645` 停止
- 每次生成新 token 后，将其 embedding 拼接到 `inputs_embeds` 末尾继续自回归

---

## 已知限制

1. **LLM 无 KV-cache**：当前 TRT LLM 每次都要重新计算整段上下文，因此端到端吞吐量低于 PyTorch（生成 32 token 约 2.6s vs PyTorch 1.3s）。输出结果与 PyTorch 完全一致。
2. **FP32 engine**：FP16 在 vision encoder 上相对误差过大（~27%），因此统一使用 FP32。
3. **batch 固定为 1**：当前 TRT engine 未做多 batch 支持。

---

## 后续优化方向

- LLM 改为 single-step KV-cache engine，显著提升生成速度
- 尝试 INT8/FP8 weight-only quantization 减小 engine 体积
- 把 text embedding 也导出为 TensorRT / ONNX，减少 PyTorch 依赖
- 支持多图输入（当前只支持单图）
