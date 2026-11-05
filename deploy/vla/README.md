# VLA (InternVL2.5-1B / InternVL3-1B) TensorRT 部署

本目录提供 InternVL2.5-1B 与 InternVL3-1B 底座 VLA 模型的 TensorRT 部署链路，与 `deploy/internvl/` 风格保持一致：

- **Vision Encoder**: 复用 `deploy/outputs/internvl2_vision_dynamic.trt` 或 `deploy/outputs/internvl3_vision_dynamic.trt`（支持动态 batch 1~24）
- **LLM**: 导出为 hidden-state 模式 ONNX/TRT（非 generation 模式）
- **Action Head + Query Tokens**: 从训练 checkpoint 导出为 ONNX/TRT
- **端到端**: TRT Vision → 构建 inputs_embeds（text + image + 24 query）→ TRT/PyTorch LLM → TRT Action Head → 24 个 (x, y) 轨迹点

> 所有部署脚本会自动从 checkpoint 的 `hyper_parameters` 中读取 `model_path`，因此 InternVL2.5 与 InternVL3 的 checkpoint 可以无缝切换，无需手动指定 `--base_model_path`。

## 文件说明

| 脚本 | 用途 |
|------|------|
| `export_action_head_onnx.py` | 从训练 checkpoint 导出 MLP Action Head ONNX |
| `export_action_head_trt.py` | 构建 Action Head TensorRT engine |
| `export_vla_llm_hidden_onnx.py` | 导出 LLM hidden-state ONNX（inputs_embeds → hidden_states） |
| `export_vla_llm_hidden_trt.py` | 构建 LLM hidden-state TensorRT engine |
| `run_trt_vla.py` | 端到端 TRT VLA 推理 |
| `compare_torch_trt.py` | 对比 PyTorch VLA 与 TRT VLA 输出 |

## 环境要求

与 `deploy/internvl/` 一致，需已安装 ONNX / TensorRT：

```bash
/home/jovyan/work/envs/myenv/bin/python3 -m pip install \
  onnx>=1.16 \
  onnxruntime-gpu>=1.23.0 \
  tensorrt==10.7.0 \
  cuda-python \
  --extra-index-url https://pypi.nvidia.com
```

> INT8 量化使用 `cuda-python` 做 GPU 内存管理，不再需要 `pycuda`。

## 一键部署命令

项目根目录已更新 `run.sh`，支持：

```bash
# 1. 导出 VLA ONNX（action head + LLM hidden）
bash run.sh deploy-vla-onnx

# 2. 构建 VLA TensorRT engines
bash run.sh deploy-vla-trt

# 3. 端到端 TRT 推理（需 4 张历史图片）
bash run.sh run-vla-trt \
  --image_paths \
    /home/jovyan/work/data/nuscenes-mini/samples/CAM_FRONT/scene000_frame0000.jpg \
    /home/jovyan/work/data/nuscenes-mini/samples/CAM_FRONT/scene000_frame0005.jpg \
    /home/jovyan/work/data/nuscenes-mini/samples/CAM_FRONT/scene000_frame0010.jpg \
    /home/jovyan/work/data/nuscenes-mini/samples/CAM_FRONT/scene000_frame0015.jpg

# 4. PyTorch vs TRT 对比
bash run.sh compare-vla-trt

# 5. INT8 量化部署（需先跑完步骤 1~2 的 ONNX 导出）
bash run.sh deploy-vla-int8
```

## 手动分步运行

```bash
cd /home/jovyan/work/vla_unified
PYTHON=/home/jovyan/work/envs/myenv/bin/python3
CHECKPOINT=./outputs/nuscenes_mini_internvl2_5_vla_smoke/checkpoints/last.ckpt

# 1. Action Head ONNX + TRT
$PYTHON deploy/vla/export_action_head_onnx.py --checkpoint $CHECKPOINT
$PYTHON deploy/vla/export_action_head_trt.py

# 2. LLM hidden-state ONNX + TRT（可选，耗时较长；如跳过则在运行时自动回退到 PyTorch LLM）
$PYTHON deploy/vla/export_vla_llm_hidden_onnx.py --checkpoint $CHECKPOINT
$PYTHON deploy/vla/export_vla_llm_hidden_trt.py --fp16

# 3. 端到端 TRT 推理（默认 PyTorch LLM + TRT Vision + TRT Action Head）
$PYTHON deploy/vla/run_trt_vla.py \
  --checkpoint $CHECKPOINT \
  --image_paths \
    /home/jovyan/work/data/nuscenes-mini/samples/CAM_FRONT/scene000_frame0000.jpg \
    /home/jovyan/work/data/nuscenes-mini/samples/CAM_FRONT/scene000_frame0005.jpg \
    /home/jovyan/work/data/nuscenes-mini/samples/CAM_FRONT/scene000_frame0010.jpg \
    /home/jovyan/work/data/nuscenes-mini/samples/CAM_FRONT/scene000_frame0015.jpg

# 4. 对比（默认 PyTorch LLM）
$PYTHON deploy/vla/compare_torch_trt.py --checkpoint $CHECKPOINT

# 5. （可选）启用完整 TRT LLM engine
$PYTHON deploy/vla/run_trt_vla.py \
  --checkpoint $CHECKPOINT \
  --image_paths ... \
  --llm_backend trt

# 6. INT8 量化（可选，精度敏感场景请先用 compare 验证）
# 6.1 生成 LLM 校准数据（从验证集采样 100 条）
$PYTHON deploy/vla/calibrate_vla_inputs.py --checkpoint $CHECKPOINT --num_samples 100

# 6.2 构建 INT8 LLM engine
$PYTHON deploy/vla/export_vla_llm_hidden_trt.py --int8

# 6.3 构建 INT8 Action Head engine（收益很小，可选）
$PYTHON deploy/vla/export_action_head_trt.py --int8

# 6.4 使用 INT8 engine 运行端到端推理
$PYTHON deploy/vla/run_trt_vla.py \
  --checkpoint $CHECKPOINT \
  --llm_engine deploy/outputs/vla_llm_hidden_int8.trt \
  --action_engine deploy/outputs/vla_action_head_int8.trt \
  --llm_backend trt \
  --image_paths ...

# 6.5 对比 INT8 与 PyTorch 精度
$PYTHON deploy/vla/compare_torch_trt.py \
  --checkpoint $CHECKPOINT \
  --llm_engine deploy/outputs/vla_llm_hidden_int8.trt \
  --action_engine deploy/outputs/vla_action_head_int8.trt \
  --llm_backend trt
```

### InternVL3-1B 部署示例

流程与 InternVL2.5 完全相同，只需使用 InternVL3 训练出的 checkpoint 与 InternVL3 的 vision engine：

```bash
cd /home/jovyan/work/vla_unified
PYTHON=/home/jovyan/work/envs/myenv/bin/python3
CHECKPOINT=./outputs/nuscenes_mini_internvl3_vla/checkpoints/last.ckpt

# 导出 ONNX
$PYTHON deploy/vla/export_action_head_onnx.py --checkpoint $CHECKPOINT \
    --output deploy/outputs/vla_action_head_internvl3.onnx
$PYTHON deploy/vla/export_vla_llm_hidden_onnx.py --checkpoint $CHECKPOINT \
    --output deploy/outputs/vla_llm_hidden_internvl3.onnx

# 构建 TRT engines
$PYTHON deploy/vla/export_action_head_trt.py \
    --onnx_path deploy/outputs/vla_action_head_internvl3.onnx \
    --output deploy/outputs/vla_action_head_internvl3.trt
$PYTHON deploy/vla/export_vla_llm_hidden_trt.py \
    --onnx_path deploy/outputs/vla_llm_hidden_internvl3.onnx \
    --output deploy/outputs/vla_llm_hidden_internvl3.trt

# 端到端推理（使用 internvl3_vision_dynamic.trt）
$PYTHON deploy/vla/run_trt_vla.py \
  --checkpoint $CHECKPOINT \
  --vision_engine deploy/outputs/internvl3_vision_dynamic.trt \
  --action_engine deploy/outputs/vla_action_head_internvl3.trt \
  --llm_backend pytorch \
  --image_paths ...

# 对比
$PYTHON deploy/vla/compare_torch_trt.py \
  --checkpoint $CHECKPOINT \
  --vision_engine deploy/outputs/internvl3_vision_dynamic.trt \
  --action_engine deploy/outputs/vla_action_head_internvl3.trt \
  --llm_backend pytorch
```

实测（InternVL3-1B smoke checkpoint，PyTorch LLM）：

```text
PyTorch ADE/FDE: 5.2099 / 6.9703
TRT     ADE/FDE: 5.1776 / 7.0063
PyTorch vs TRT max abs diff: 4.31e-01
PyTorch vs TRT mean abs diff: 1.21e-01
PyTorch time: 0.781s
TRT time:     0.443s
Speedup:      1.76x
🎉 Comparison PASSED.
```

## 输出产物

```text
deploy/outputs/
├── internvl2_vision_dynamic.trt          # InternVL2.5 动态 batch vision engine
├── internvl3_vision_dynamic.trt          # InternVL3 动态 batch vision engine
├── vla_action_head.onnx                  # Action Head ONNX (InternVL2.5)
├── vla_action_head.trt                   # Action Head TensorRT engine (InternVL2.5, FP16)
├── vla_action_head_int8.trt              # Action Head TensorRT engine (InternVL2.5, INT8, optional)
├── vla_action_head_internvl3.onnx        # Action Head ONNX (InternVL3)
├── vla_action_head_internvl3.trt         # Action Head TensorRT engine (InternVL3, FP16)
├── vla_llm_hidden.onnx                   # LLM hidden-state ONNX (InternVL2.5)
├── vla_llm_hidden.trt                    # LLM hidden-state TensorRT engine (InternVL2.5)
├── vla_llm_hidden_int8.trt               # LLM hidden-state TensorRT engine (InternVL2.5, INT8, optional)
├── vla_llm_hidden_internvl3.onnx         # LLM hidden-state ONNX (InternVL3)
├── vla_llm_hidden_internvl3.trt          # LLM hidden-state TensorRT engine (InternVL3)
├── vla_llm_calib_data.npz                # INT8 校准数据
└── vla_llm_hidden.cache                  # TensorRT INT8 校准缓存
```

## 设计说明

- **为什么 LLM 导出为 hidden-state 模式？**
  VLA 不需要自回归生成文本，只需要对输入序列做一次前向传播，取最后 24 个 query 位置的 hidden state。因此 LLM ONNX 输出 `hidden_states` 而非 `logits`，action head 在此基础上回归轨迹。

- **默认使用 PyTorch LLM（混合部署）**
  当前 TRT LLM hidden-state engine 与 PyTorch GPU 路径在数值上存在可观测偏差（对回归任务较敏感），因此 `run_trt_vla.py` 与 `compare_torch_trt.py` 默认使用 **PyTorch LLM + TRT Vision + TRT Action Head** 的混合部署。LLM TRT engine 仍可通过 `--llm_backend trt` 启用，作为进一步优化的可选路径。如果 `vla_llm_hidden.trt` 不存在，会自动回退到 PyTorch LLM。

- **运行时不做固定 padding**
  `run_trt_vla.py` 对 prompt 使用 `padding=False`，序列长度随实际输入变化，避免超过默认 LLM engine 的 `max_seq=2048`。

- **默认复用训练 checkpoint**
  Action head 与 query token 必须从训练 checkpoint 加载；LLM 默认也从 checkpoint 加载。如果只想快速验证 base InternVL 的部署通路，可对 LLM 导出脚本使用 `--base_model_path` 指定原始模型路径。

- **INT8 量化说明**
  LLM hidden-state engine 支持 INT8 量化，但需要先用 `calibrate_vla_inputs.py` 在真实验证数据上生成校准数据。由于 INT8 对回归任务精度敏感，建议先跑 `compare-vla-trt` 对比 FP32/FP16/INT8 的轨迹误差，再决定是否启用。Action Head 本身仅约 2 MB，INT8 收益很小，一般保持 FP16 即可。

## INT8 量化效果确认

建议按以下步骤确认 INT8 量化是否满足精度要求：

```bash
cd /home/jovyan/work/vla_unified
PYTHON=/home/jovyan/work/envs/myenv/bin/python3
CHECKPOINT=./outputs/nuscenes_mini_internvl2_5_vla_smoke/checkpoints/last.ckpt

# 1. 生成 INT8 校准数据
$PYTHON deploy/vla/calibrate_vla_inputs.py --checkpoint $CHECKPOINT --num_samples 100

# 2. 构建 INT8 engines
$PYTHON deploy/vla/export_vla_llm_hidden_trt.py --int8 \
    --output deploy/outputs/vla_llm_hidden_int8.trt
$PYTHON deploy/vla/export_action_head_trt.py --int8 \
    --output deploy/outputs/vla_action_head_int8.trt

# 3. 对比 PyTorch vs INT8 TRT（回归任务关键指标：ADE / FDE / 逐点误差）
$PYTHON deploy/vla/compare_torch_trt.py \
    --checkpoint $CHECKPOINT \
    --llm_engine deploy/outputs/vla_llm_hidden_int8.trt \
    --action_engine deploy/outputs/vla_action_head_int8.trt \
    --llm_backend trt

# 4. 与 FP16/FP32 TRT 横向对比（如已构建）
$PYTHON deploy/vla/compare_torch_trt.py \
    --checkpoint $CHECKPOINT \
    --llm_engine deploy/outputs/vla_llm_hidden.trt \
    --action_engine deploy/outputs/vla_action_head.trt \
    --llm_backend trt
```

### 评估标准参考

- **ADE/FDE 相对增量 < 5%**：INT8 可接受
- **ADE/FDE 相对增量 5%~15%**：需检查是否校准样本不足或分布偏移
- **ADE/FDE 相对增量 > 15%**：建议保持 FP16/FP32，或改用 QAT

### 实测结果（NuScenes-mini 验证集单样本）

| 配置 | PyTorch ADE/FDE | TRT ADE/FDE | mean diff | max diff | 结论 |
|------|-----------------|-------------|-----------|----------|------|
| PyTorch LLM + FP16 Action Head | 10.1674 / 3.2771 | 10.0824 / 3.2302 | 0.10 | 0.41 | ✅ 推荐 |
| PyTorch LLM + **INT8** Action Head | 10.1674 / 3.2771 | 10.0824 / 3.2301 | 0.10 | 0.41 | ✅ 可用 |
| **INT8** LLM + INT8 Action Head | 10.1674 / 3.2771 | **13.4956 / 32.3216** | 11.26 | 30.08 | ❌ 不推荐 |
| FP32 LLM + FP16 Action Head | 10.1674 / 3.2771 | **13.4956 / 32.3214** | 11.26 | 30.08 | ❌ 不推荐 |

> 说明：INT8 LLM engine 虽然构建成功，但端到端精度与 PyTorch 差距较大。意外的是，**FP32 TRT LLM engine 也有几乎相同的偏差**，说明误差主要来源于 TRT LLM hidden-state engine 本身，而非 INT8 量化。因此当前建议保持 **PyTorch LLM + TRT Vision + INT8 Action Head** 的混合部署，只把 Action Head 做 INT8 量化。

### 总结与建议

1. **INT8 Action Head 没问题**  
   端到端误差几乎和 FP16 一致（mean diff 0.10，max diff 0.41），engine 大小保持约 2 MB（本身太小，体积收益可忽略）。

2. **INT8 LLM 当前不推荐**  
   虽然 engine 构建成功（`vla_llm_hidden_int8.trt`，1.4 GB），但端到端 FDE 从 3.28 暴涨到 32.32。

3. **问题不在 INT8，而在 TRT LLM engine 本身**  
   FP32 TRT LLM engine 也有几乎相同的偏差，说明误差主要来自 TensorRT hidden-state LLM engine 与 PyTorch LLM 的数值差异，而非 INT8 量化。

### 推荐部署方案

保持 **PyTorch LLM + TRT Vision + INT8 Action Head** 的混合部署：

```bash
cd /home/jovyan/work/vla_unified
PYTHON=/home/jovyan/work/envs/myenv/bin/python3
CHECKPOINT=./outputs/nuscenes_mini_internvl2_5_vla_smoke/checkpoints/last.ckpt

# 1. 生成校准数据（验证集有效样本可能不足 100，实际以数据集为准）
$PYTHON deploy/vla/calibrate_vla_inputs.py --checkpoint $CHECKPOINT --num_samples 100

# 2. 只构建 INT8 Action Head（LLM 保持 PyTorch）
$PYTHON deploy/vla/export_action_head_trt.py --int8 \
    --output deploy/outputs/vla_action_head_int8.trt

# 3. 运行混合推理（不指定 --llm_engine，自动使用 PyTorch LLM）
$PYTHON deploy/vla/run_trt_vla.py \
  --checkpoint $CHECKPOINT \
  --action_engine deploy/outputs/vla_action_head_int8.trt \
  --image_paths ...
```

### 已知限制

- Vision Encoder 保持 FP32，不参与 INT8 量化
- INT8 / FP32 TRT LLM hidden-state engine 与 PyTorch LLM 在数值上存在可观测偏差，对回归任务敏感
- 当前校准默认从验证集采样 100 条；如精度不达标，可尝试增加到 200~500 条或使用训练集采样
- 如确实需要压缩 LLM，建议尝试 TensorRT-LLM、AWQ、GPTQ 等 LLM 专用量化方案，或量化感知训练（QAT）
