# Qwen2-VL-2B TensorRT 部署

本目录提供 Qwen2-VL-2B-Instruct 的 PyTorch → ONNX → TensorRT 部署链路。

> ⚠️ 当前 Vision Engine 为**固定分辨率**导出：导出时会根据一张样例图片计算 `image_grid_thw`，并用该值固定 RoPE / cu_seqlens。因此导出的 ONNX / TRT engine 只能处理与样例图相同尺寸的图片。后续如需多分辨率，需要把 `image_grid_thw` 作为动态 shape tensor 处理。

## 文件说明

| 脚本 | 用途 |
|------|------|
| `export_vision_onnx.py` | 导出 Vision Encoder ONNX（固定分辨率） |
| `export_vision_trt.py` | 构建 Vision Encoder TensorRT engine |
| `export_llm_embed_onnx.py` | 导出 LLM ONNX（inputs_embeds → logits） |
| `export_llm_embed_trt.py` | 构建 LLM TensorRT engine |
| `run_trt_qwen2vl.py` | 端到端 TRT 推理 |
| `compare_torch_trt.py` | 对比 PyTorch generate() 与 TRT 推理结果 |

## 环境准备

项目根目录已创建 `.venv`，所有脚本默认使用：

```bash
cd /root/embodvla
.venv/bin/python -m pip install -r requirements.txt
```

> 注意：原仓库硬编码的 `/home/jovyan/work/envs/myenv/bin/python3` 在当前机器不存在，`run.sh` 已改为优先使用该路径、不存在时回退到 `.venv/bin/python`。

## 快速开始（推荐用 `run.sh`）

```bash
cd /root/embodvla

# 1. 导出 ONNX（vision + LLM）
bash run.sh deploy-qwen2vl-2b-onnx

# 2. 构建 TensorRT engines
bash run.sh deploy-qwen2vl-2b-trt

# 3. 端到端 TRT 推理
bash run.sh run-qwen2vl-2b-trt --max_new_tokens 32

# 4. 与 PyTorch 对比
bash run.sh compare-qwen2vl-2b-trt --max_new_tokens 32
```

默认使用项目本地的：
- 模型：`models/Qwen/Qwen2-VL-2B-Instruct`
- 测试图：`assets/test_image.jpg`
- 输出：`deploy/outputs/`

## 手动分步运行

```bash
cd /root/embodvla
PYTHON=.venv/bin/python
MODEL=models/Qwen/Qwen2-VL-2B-Instruct
IMAGE=assets/test_image.jpg

# 1. Vision ONNX + TRT
$PYTHON deploy/qwen2vl/export_vision_onnx.py \
  --model_path $MODEL --image_path $IMAGE
$PYTHON deploy/qwen2vl/export_vision_trt.py \
  --model_path $MODEL --image_path $IMAGE

# 2. LLM ONNX + TRT
$PYTHON deploy/qwen2vl/export_llm_embed_onnx.py --model_path $MODEL
$PYTHON deploy/qwen2vl/export_llm_embed_trt.py \
  --onnx_path deploy/outputs/qwen2vl_2b_llm_embed.onnx \
  --hidden_size 1536 --max_seq 2048

# 3. 端到端推理
$PYTHON deploy/qwen2vl/run_trt_qwen2vl.py \
  --model_path $MODEL --image_path $IMAGE --max_new_tokens 32

# 4. 对比
$PYTHON deploy/qwen2vl/compare_torch_trt.py \
  --model_path $MODEL --image_path $IMAGE --max_new_tokens 32
```

## 本次修复点

1. **路径可配置**：所有脚本默认改为项目本地路径，不再依赖 `/home/jovyan/work/...`。
2. **支持任意样例图**：`export_vision_onnx.py` / `export_vision_trt.py` / `run_trt_qwen2vl.py` / `compare_torch_trt.py` 新增 `--image_path` 参数。
3. **修正 MRoPE position_ids**：`run_trt_qwen2vl.py` 现在调用 Qwen2-VL 官方的 `get_rope_index` 生成 3-D position_ids，而不是简单把 1-D arange 复制 3 份。
4. **截断移到 TRT 推理前**：避免序列超过 engine `max_seq=2048` 时直接 shape 报错。
5. **`run.sh` 增加 Qwen2-VL 部署说明**：`usage()` 里新增 `deploy-qwen2vl-2b-*` 命令列表。

## 实测结果

环境：
- GPU：NVIDIA（CUDA 12.4）
- PyTorch：`2.6.0+cu124`
- TensorRT：`10.7.0`
- Transformers：`5.12.1`
- 模型：`models/Qwen/Qwen2-VL-2B-Instruct`
- 测试图：`assets/test_image.jpg`（山景，从 Unsplash 下载）

### 导出产物

| 产物 | 路径 | 大小 |
|------|------|------|
| Vision ONNX | `deploy/outputs/qwen2vl_2b_vision.onnx` | ~2.5 GB |
| Vision TRT Engine | `deploy/outputs/qwen2vl_2b_vision.trt` | ~2.5 GB |
| LLM ONNX | `deploy/outputs/qwen2vl_2b_llm_embed.onnx` | ~5.9 GB |
| LLM TRT Engine | `deploy/outputs/qwen2vl_2b_llm_embed.trt` | ~5.9 GB |

### 端到端 TRT 推理

```bash
.venv/bin/python deploy/qwen2vl/run_trt_qwen2vl.py \
  --model_path models/Qwen/Qwen2-VL-2B-Instruct \
  --image_path assets/test_image.jpg \
  --max_new_tokens 32
```

输出示例（提示词：描述这张图片）：

```text
这张图片展示了一幅壮丽的山景。画面中，雄伟的山脉在夕阳的映照下，山峰被染上了一层
```

关键日志：
- `input_ids shape: (1, 1023), pixel_values shape: (3996, 1176)`
- `Vision TRT output shape: (999, 1536)`
- 首 token 耗时：~0.40s
- 总耗时：~13.4s / 32 tokens，吞吐 ~2.39 tokens/s

### PyTorch vs TensorRT 对比

```bash
.venv/bin/python deploy/qwen2vl/compare_torch_trt.py \
  --model_path models/Qwen/Qwen2-VL-2B-Instruct \
  --image_path assets/test_image.jpg \
  --max_new_tokens 32
```

结果：
- Token 匹配率：**100%**
- PyTorch 推理时间：~1.17s
- TensorRT 推理时间：~20.15s
- 两者生成的中文描述完全一致，无 `<image>` / `<unk>` 泄漏

#### 为什么 TRT 比 PyTorch 慢？

当前 TRT 实现**没有启用 KV-cache**，而 PyTorch 的 `generate()` 默认使用了 KV-cache：

| 阶段 | PyTorch `generate()` | 当前 TRT 实现 |
|------|---------------------|---------------|
| 第 1 个 token | 完整跑一遍图像 + 文本前缀，保存 KV | 完整跑一遍图像 + 文本前缀 |
| 第 2~N 个 token | 只输入**最新 1 个 token**，复用前面 KV | 把**整个序列**重新送进 engine |

因此 TRT 生成 32 个 token 要做 32 次全序列前向，计算量随长度平方增长；而 PyTorch 每次只需处理新增 token，总体更快。

后续优化方向：把 LLM engine 改成两阶段（context 阶段一次算 KV，decode 阶段逐 token 更新 KV），这是 TensorRT-LLM 等框架的标准做法。

## 已知限制

- Vision Engine 目前固定分辨率，换不同尺寸图片需要重新导出。
- LLM Engine 未使用 KV-cache，每步重新计算全序列，速度较慢。
- 当前 TRT 部署以“跑通 + 结果对齐”为首要目标，性能优化尚未完成。
