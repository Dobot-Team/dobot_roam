# SmolVLM2 ONNX + TensorRT 部署说明

本目录包含 SmolVLM2-256M / SmolVLM2-500M 的完整 ONNX / TensorRT 部署链路脚本。

> **设计原则**：把 SmolVLM2 拆成 **Vision Encoder (vision_model + connector)** 与 **LLM (text_model + lm_head)** 两个模块分别导出/转 TRT，最后在 Python 生成循环里通过 `inputs_merger` 把图像特征注入到 text embedding 中。
>
> 本 README 中的命令和指标均已在当前环境实际跑通验证。

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

当前验证环境：

```text
tensorrt    10.7.0
onnxruntime 1.23.2
```

---

## 文件说明

### 导出脚本

| 脚本 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `export_vision_onnx.py` | `pixel_values (1, 9, 3, 512, 512)` | `smolvlm2_*_vision.onnx` | Vision Encoder (vision_model + connector) |
| `export_vision_trt.py` | ONNX | `smolvlm2_*_vision.trt` | Vision TensorRT engine |
| `export_llm_embed_onnx.py` | `inputs_embeds (1, seq_len, hidden_size)` | `smolvlm2_*_llm_embed.onnx` | LLM (inputs_embeds 模式，端到端用) |
| `export_llm_embed_trt.py` | ONNX | `smolvlm2_*_llm_embed.trt` | LLM TensorRT engine (inputs_embeds) |

导出脚本内部自带 **CPU ONNX vs CPU PyTorch** 校验，会打印 `max_abs_error / mean_abs_error / rel_error` 并判断是否在容忍范围内。

### 端到端 / 对比脚本

| 脚本 | 用途 |
|------|------|
| `run_trt_smolvlm.py` | 图片 + prompt -> TRT Vision -> `inputs_merger` 注入 image feature -> TRT LLM 贪心生成 |
| `compare_torch_trt.py` | 同一张图/同一个 prompt 分别跑 PyTorch `model.generate()` 与 TRT，输出 token 匹配率与耗时 |
| `benchmark_vision_latency.py` | 单独测试 Vision Encoder (ONNX / TensorRT / PyTorch) 的推理延迟 |

---

## 一键命令（推荐）

在项目根目录执行：

```bash
# 1. 导出 ONNX（vision + LLM-embed）
bash run.sh deploy-smolvlm2-256m-onnx
bash run.sh deploy-smolvlm2-500m-onnx

# 2. 构建 TensorRT engine
bash run.sh deploy-smolvlm2-256m-trt
bash run.sh deploy-smolvlm2-500m-trt

# 3. TRT 端到端推理
bash run.sh run-smolvlm2-256m-trt --prompt "Describe this image in detail." --max_new_tokens 64
bash run.sh run-smolvlm2-500m-trt --prompt "Describe this image in detail." --max_new_tokens 64

# 4. 与 PyTorch 严格对比
bash run.sh compare-smolvlm2-256m-trt --max_new_tokens 32
bash run.sh compare-smolvlm2-500m-trt --max_new_tokens 32

# 5. Vision Encoder 单模块延迟测试（ONNX vs TensorRT，支持多 sample）
bash run.sh bench-smolvlm2-256m-vision-latency --batch_size 1
bash run.sh bench-smolvlm2-256m-vision-latency --batch_size 6
bash run.sh bench-smolvlm2-500m-vision-latency --batch_size 12
bash run.sh bench-smolvlm2-500m-vision-latency --batch_size 24
```

---

## Vision Encoder 延迟测试

使用 `deploy/smolvlm/benchmark_vision_latency.py` 单独测量 SmolVLM2 Vision Encoder 的推理延迟，方便在板端做性能摸底。

> **重要说明**：SmolVLM2 的 `vision_model` 内部把 `num_images=9` 这一维与 batch 维混在一起处理（position embedding 的 split 沿 axis=0 固定为 9 份），因此导出的 ONNX / TRT engine **本质上只支持一个 outer sample**（`pixel_values` 形状为 `(1, 9, 3, 512, 512)`）。当 `--batch_size > 1` 时，benchmark 脚本会跑 **N 次 batch=1 的串行推理** 来模拟 N 个 sample，并统计总耗时与等效单 sample 耗时。

### 单 backend 测试

```bash
cd /home/jovyan/work/vla_unified
PYTHON=/home/jovyan/work/envs/myenv/bin/python3
MODEL=/home/jovyan/work/vla_smolvlm2_256m/models/HuggingFaceTB/SmolVLM2-256M-Video-Instruct

# ONNX Runtime (CUDA)，batch=6（实际为 6 次串行 batch=1）
$PYTHON deploy/smolvlm/benchmark_vision_latency.py \
  --skip_pytorch --skip_trt \
  --model_path $MODEL \
  --onnx_provider CUDAExecutionProvider \
  --batch_size 6 \
  --warmup 20 --iterations 80

# TensorRT，batch=12（实际为 12 次串行 batch=1）
$PYTHON deploy/smolvlm/benchmark_vision_latency.py \
  --skip_pytorch --skip_onnx \
  --model_path $MODEL \
  --batch_size 12 \
  --warmup 20 --iterations 80
```

### 板端示例

```bash
python3 deploy/smolvlm/benchmark_vision_latency.py \
  --skip_pytorch --skip_trt \
  --model_path /path/to/SmolVLM2-256M-Video-Instruct \
  --onnx_provider CPUExecutionProvider \
  --batch_size 6 \
  --warmup 10 --iterations 50
```

---

## 手动运行示例

### SmolVLM2-256M

```bash
cd /home/jovyan/work/vla_unified
PYTHON=/home/jovyan/work/envs/myenv/bin/python3
MODEL=/home/jovyan/work/vla_smolvlm2_256m/models/HuggingFaceTB/SmolVLM2-256M-Video-Instruct

# ONNX 导出
$PYTHON deploy/smolvlm/export_vision_onnx.py --model_path $MODEL
$PYTHON deploy/smolvlm/export_llm_embed_onnx.py --model_path $MODEL

# TRT 建 engine
$PYTHON deploy/smolvlm/export_vision_trt.py \
  --onnx_path deploy/outputs/smolvlm2_256m_vision.onnx \
  --output deploy/outputs/smolvlm2_256m_vision.trt
$PYTHON deploy/smolvlm/export_llm_embed_trt.py \
  --onnx_path deploy/outputs/smolvlm2_256m_llm_embed.onnx \
  --output deploy/outputs/smolvlm2_256m_llm_embed.trt

# 端到端
$PYTHON deploy/smolvlm/run_trt_smolvlm.py \
  --model_path $MODEL \
  --vision_engine deploy/outputs/smolvlm2_256m_vision.trt \
  --llm_engine deploy/outputs/smolvlm2_256m_llm_embed.trt \
  --image_path 20260610151154.jpg \
  --prompt "描述这张图片" \
  --max_new_tokens 32

# 对比
$PYTHON deploy/smolvlm/compare_torch_trt.py \
  --model_path $MODEL \
  --image_path 20260610151154.jpg \
  --max_new_tokens 32
```

### SmolVLM2-500M

将上面命令中的 `256M` 替换为 `500M`，模型路径替换为：

```bash
MODEL=/home/jovyan/work/vla_smolvlm2_500m/models/HuggingFaceTB/SmolVLM2-500M-Video-Instruct
```

---

## 输出产物

```text
deploy/outputs/
├── smolvlm2_256m_vision.onnx         # Vision ONNX (~358MB)
├── smolvlm2_256m_vision.trt          # Vision TensorRT engine (~360MB，FP32)
├── smolvlm2_256m_llm_embed.onnx      # LLM inputs_embeds ONNX (~515MB)
├── smolvlm2_256m_llm_embed.trt       # LLM inputs_embeds TRT engine (~520MB，FP32)
├── smolvlm2_500m_vision.onnx         # Vision ONNX (~376MB)
├── smolvlm2_500m_vision.trt          # Vision TensorRT engine (~378MB，FP32)
├── smolvlm2_500m_llm_embed.onnx      # LLM inputs_embeds ONNX (~1.4GB)
├── smolvlm2_500m_llm_embed.trt       # LLM inputs_embeds TRT engine (~1.4GB，FP32)
└── onnx__MatMul_* / *.weight         # ONNX 外部权重文件
```

---

## 精度指标参考

以下数据在当前环境（RTX 4090，TensorRT 10.7.0，FP32 engine）实测得到。

### ONNX 导出自校验

导出脚本最后会执行 CPU ONNX vs CPU PyTorch，默认阈值：

```text
max_abs_error < 1e-2
mean_abs_error < 1e-4
rel_error     < 1e-3
```

Vision 与 LLM 导出均会打印 `max_abs_error / mean_abs_error`。

### TRT 端到端对比（图片：`20260610151154.jpg`，`max_new_tokens=32`）

#### SmolVLM2-256M

```text
PyTorch time:     1.246s
TRT time:         5.950s
Token match rate: 96.9%
Speedup:          0.21x
Status:           🎉 Comparison PASSED.
```

#### SmolVLM2-500M

```text
PyTorch time:     1.200s
TRT time:         6.846s
Token match rate: 96.9%
Speedup:          0.18x
Status:           🎉 Comparison PASSED.
```

> 注：TRT 当前未使用 KV-cache，每次 forward 重新计算整段上下文，因此端到端耗时高于 PyTorch；文本输出与 PyTorch 高度一致。

---

## Vision 多 Sample 延迟参考（RTX 4090）

由于 SmolVLM2 Vision Encoder 只支持单个 outer sample，以下数据通过 **N 次 batch=1 串行推理** 测得。每个 sample 内部含 9 张 `512×512` 图像 patch，即输入形状为 `(1, 9, 3, 512, 512)`。

### SmolVLM2-256M

| Sample 数 | ONNX-CUDA 总延迟 | ONNX-CUDA 单 sample | ONNX-CUDA 吞吐 | TensorRT 总延迟 | TensorRT 单 sample | TensorRT 吞吐 |
|----------|-----------------|--------------------|---------------|----------------|-------------------|--------------|
| 1  | 87.98 ms | 87.98 ms | 11.37 samples/s | 57.27 ms | 57.27 ms | 17.46 samples/s |
| 6  | 528.69 ms | 88.12 ms | 11.35 samples/s | 345.43 ms | 57.57 ms | 17.37 samples/s |
| 12 | 1057.31 ms | 88.11 ms | 11.35 samples/s | 695.74 ms | 57.98 ms | 17.25 samples/s |
| 24 | 2117.52 ms | 88.23 ms | 11.33 samples/s | 1380.52 ms | 57.52 ms | 17.38 samples/s |

### SmolVLM2-500M

| Sample 数 | ONNX-CUDA 总延迟 | ONNX-CUDA 单 sample | ONNX-CUDA 吞吐 | TensorRT 总延迟 | TensorRT 单 sample | TensorRT 吞吐 |
|----------|-----------------|--------------------|---------------|----------------|-------------------|--------------|
| 1  | 88.30 ms | 88.30 ms | 11.32 samples/s | 57.88 ms | 57.88 ms | 17.28 samples/s |
| 6  | 529.12 ms | 88.19 ms | 11.34 samples/s | 347.64 ms | 57.94 ms | 17.26 samples/s |
| 12 | 1056.73 ms | 88.06 ms | 11.36 samples/s | 695.09 ms | 57.92 ms | 17.27 samples/s |
| 24 | 2113.60 ms | 88.07 ms | 11.36 samples/s | 1389.36 ms | 57.89 ms | 17.27 samples/s |

结论：
- 256M 与 500M 的 Vision Encoder 结构相同，因此延迟几乎一致。
- ONNX-CUDA 单 sample 约 **88 ms**，TensorRT 单 sample 约 **57~58 ms**，TRT 比 ONNX-CUDA 快约 **34%**。
- 由于只能串行推理，总延迟基本随 sample 数线性增长。

---

## 实现细节

### 1. 输入规格

- **Vision Encoder**
  - 固定 `batch=1`，固定 `num_images=9`
  - 输入：`pixel_values (1, 9, 3, 512, 512)`
  - 输出：`image_hidden_states (1, 9, 64, hidden_size)`
  - `hidden_size`：256M 为 576，500M 为 960

- **LLM (inputs_embeds 模式)**
  - 支持动态 `seq_len ∈ [1, 1024]`
  - 输入：`inputs_embeds (1, seq_len, hidden_size)`、`attention_mask (1, seq_len)`、`position_ids (1, seq_len)`
  - 输出：`logits (1, seq_len, vocab_size)`

### 2. 图像特征注入

SmolVLM2 通过 `model.model.inputs_merger` 完成图像与文本特征的融合：

1. Prompt 经 processor 得到 `input_ids`
2. `embed_tokens(input_ids)` 得到 text embeddings
3. TRT Vision 输出 `image_hidden_states (1, 9, 64, hidden_size)`
4. 调用 `inputs_merger(input_ids, inputs_embeds, image_hidden_states)` 得到融合后的 `inputs_embeds`
5. 将融合后的 `inputs_embeds` 送入 TRT LLM

### 3. 生成策略

- 贪心解码（greedy）
- 遇到 `eos_token_id` 停止
- 每次生成新 token 后，将其 embedding 拼接到 `inputs_embeds` 末尾继续自回归
- 上下文长度上限 1024，超过时做尾部截断

---

## 已知限制

1. **LLM 无 KV-cache**：当前 TRT LLM 每次都要重新计算整段上下文，因此端到端吞吐量低于 PyTorch（见上方对比指标）。输出结果与 PyTorch 基本一致。
2. **FP32 engine**：默认使用 FP32，未启用 FP16。
3. **Vision Encoder 仅支持 outer batch=1 / 图像帧数固定为 9**：SmolVLM2 的 `vision_model` 内部 position embedding 按 `num_images=9` 切分，导致 ONNX / TRT engine 无法直接处理 outer batch>1。多 sample 场景需串行调用 batch=1 engine。
4. **无独立 verify 脚本**：ONNX 精度校验内嵌在导出脚本中；端到端精度通过 `compare_torch_trt.py` 验证。

---

## 后续优化方向

- LLM 改为 single-step KV-cache engine，显著提升生成速度
- 尝试 INT8/FP8 weight-only quantization 减小 engine 体积
- 把 text embedding / inputs_merger 也导出为 TensorRT / ONNX，进一步减少 PyTorch 依赖
- 支持动态图像帧数与多 batch
