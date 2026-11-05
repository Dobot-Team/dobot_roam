#!/bin/bash
# Unified VLA base model sanity check & deployment launcher

cd "$(dirname "$0")"

# Prefer the original project environment if it exists, otherwise use the local venv.
PYTHON="${PYTHON:-/home/jovyan/work/envs/myenv/bin/python3}"
if [ ! -x "$PYTHON" ]; then
    PYTHON="$(pwd)/.venv/bin/python"
fi
SCRIPT="tests/manual/base_model_smoke.py"
IMAGE="assets/test_image.jpg"
DEPLOY_DIR="deploy/internvl"

usage() {
    echo "Usage:"
    echo "  Base model check: $0 {internvl2|internvl3|smolvlm2-256m|smolvlm2-500m|qwen2vl-2b} [extra_args]"
    echo "  InternVL2.5 deployment:"
    echo "    $0 deploy-internvl2-onnx           # export vision + LLM ONNX"
    echo "    $0 deploy-internvl2-trt            # build TensorRT engines"
    echo "    $0 run-internvl2-trt               # run TRT end-to-end"
    echo "    $0 compare-internvl2-trt           # compare TRT vs PyTorch"
    echo "    $0 bench-internvl2-vision-latency  # benchmark ViT ONNX/TRT latency"
    echo ""
    echo "  VLA (InternVL2.5-based) deployment:"
    echo "    $0 deploy-vla-onnx                 # export action head + LLM hidden ONNX (InternVL2.5)"
    echo "    $0 deploy-vla-trt                  # build VLA TensorRT engines (InternVL2.5)"
    echo "    $0 deploy-vla-int8                 # build VLA INT8 TensorRT engines (InternVL2.5)"
    echo "    $0 run-vla-trt --image_paths ...   # run TRT end-to-end VLA inference (InternVL2.5)"
    echo "    $0 compare-vla-trt                 # compare TRT vs PyTorch VLA (InternVL2.5)"
    echo "    $0 deploy-vla-internvl3-onnx       # export action head + LLM hidden ONNX (InternVL3)"
    echo "    $0 deploy-vla-internvl3-trt        # build VLA TensorRT engines (InternVL3)"
    echo "    $0 run-vla-internvl3-trt --image_paths ..."
    echo "    $0 compare-vla-internvl3-trt       # compare TRT vs PyTorch VLA (InternVL3)"
    echo ""
    echo "  VLA (Qwen2-VL-2B-based) deployment:"
    echo "    $0 train-qwen2vl-vla               # train Qwen2-VL VLA on NuScenes-mini (smoke: 10 steps)"
    echo "    $0 eval-qwen2vl-vla                # evaluate Qwen2-VL VLA checkpoint"
    echo "    $0 smoke-qwen2vl-vla               # synthetic smoke test (no real data needed)"
    echo "    $0 deploy-qwen2vl-vla-onnx         # export action head + LLM hidden ONNX"
    echo "    $0 deploy-qwen2vl-vla-trt          # build Qwen2-VL VLA TensorRT engines"
    echo "    $0 run-qwen2vl-vla-trt --image_paths ..."
    echo "    $0 compare-qwen2vl-vla-trt         # compare TRT vs PyTorch Qwen2-VL VLA"
    echo ""
    echo "  InternVL3 deployment:"
    echo "    $0 deploy-internvl3-onnx           # export vision + LLM ONNX"
    echo "    $0 deploy-internvl3-trt            # build TensorRT engines"
    echo "    $0 run-internvl3-trt               # run TRT end-to-end"
    echo "    $0 compare-internvl3-trt           # compare TRT vs PyTorch"
    echo "    $0 bench-internvl3-vision-latency  # benchmark ViT ONNX/TRT latency"
    echo ""
    echo "  SmolVLM2 deployment:"
    echo "    $0 deploy-smolvlm2-256m-onnx           # export vision + LLM ONNX"
    echo "    $0 deploy-smolvlm2-256m-trt            # build TensorRT engines"
    echo "    $0 run-smolvlm2-256m-trt               # run TRT end-to-end"
    echo "    $0 compare-smolvlm2-256m-trt           # compare TRT vs PyTorch"
    echo "    $0 bench-smolvlm2-256m-vision-latency  # benchmark ViT ONNX/TRT latency"
    echo "    $0 deploy-smolvlm2-500m-onnx           # export vision + LLM ONNX"
    echo "    $0 deploy-smolvlm2-500m-trt            # build TensorRT engines"
    echo "    $0 run-smolvlm2-500m-trt               # run TRT end-to-end"
    echo "    $0 compare-smolvlm2-500m-trt           # compare TRT vs PyTorch"
    echo "    $0 bench-smolvlm2-500m-vision-latency  # benchmark ViT ONNX/TRT latency"
    echo ""
    echo "  Qwen2-VL deployment:"
    echo "    $0 deploy-qwen2vl-2b-onnx           # export vision + LLM ONNX"
    echo "    $0 deploy-qwen2vl-2b-trt            # build TensorRT engines"
    echo "    $0 run-qwen2vl-2b-trt               # run TRT end-to-end"
    echo "    $0 compare-qwen2vl-2b-trt           # compare TRT vs PyTorch"
    echo ""
    echo "Examples:"
    echo "  $0 internvl2"
    echo "  $0 internvl3 --prompt \"描述这张图片\" --chinese"
    echo "  $0 smolvlm2-500m --max_new_tokens 256"
    echo "  $0 qwen2vl-2b --prompt \"描述这张图片\""
    echo "  $0 run-internvl2-trt --prompt \"描述这张图片\" --max_new_tokens 64"
    echo "  $0 compare-vla-trt"
    echo "  $0 compare-vla-internvl3-trt"
    exit 1
}

if [ $# -eq 0 ]; then
    usage
fi

MODEL=$1
shift

case "$MODEL" in
    internvl2|internvl3|smolvlm2-256m|smolvlm2-500m|qwen2vl-2b)
        echo "================================"
        echo "Unified VLA Base Model Check"
        echo "Model: $MODEL"
        echo "================================"
        $PYTHON $SCRIPT \
          --model $MODEL \
          --image_path $IMAGE \
          --device cuda \
          --max_new_tokens 128 \
          "$@"
        ;;
    deploy-internvl2-onnx)
        echo "================================"
        echo "InternVL2.5 ONNX Export"
        echo "================================"
        $PYTHON $DEPLOY_DIR/export_vision_onnx.py && \
        $PYTHON $DEPLOY_DIR/export_llm_onnx.py && \
        $PYTHON $DEPLOY_DIR/export_llm_embed_onnx.py
        ;;
    deploy-internvl2-trt)
        echo "================================"
        echo "InternVL2.5 TensorRT Engine Build"
        echo "================================"
        $PYTHON $DEPLOY_DIR/export_vision_trt.py && \
        $PYTHON $DEPLOY_DIR/export_llm_trt.py && \
        $PYTHON $DEPLOY_DIR/export_llm_embed_trt.py
        ;;
    run-internvl2-trt)
        echo "================================"
        echo "InternVL2.5 TensorRT Inference"
        echo "================================"
        $PYTHON $DEPLOY_DIR/run_trt_internvl.py \
          --image_path $IMAGE \
          "$@"
        ;;
    compare-internvl2-trt)
        echo "================================"
        echo "InternVL2.5 PyTorch vs TensorRT"
        echo "================================"
        $PYTHON $DEPLOY_DIR/compare_torch_trt.py \
          --image_path $IMAGE \
          "$@"
        ;;
    bench-internvl2-vision-latency)
        echo "================================"
        echo "InternVL2.5 Vision Latency Benchmark"
        echo "================================"
        $PYTHON $DEPLOY_DIR/benchmark_vision_latency.py \
          --skip_pytorch \
          --onnx_path deploy/outputs/internvl2_vision_dynamic.onnx \
          --engine_path deploy/outputs/internvl2_vision_dynamic.trt \
          --onnx_provider CUDAExecutionProvider \
          --image_path $IMAGE \
          --warmup 50 \
          --iterations 200 \
          --output_json deploy/outputs/internvl2_vision_latency.json \
          "$@"
        ;;
    deploy-vla-onnx)
        echo "================================"
        echo "VLA ONNX Export"
        echo "================================"
        $PYTHON deploy/vla/export_action_head_onnx.py && \
        $PYTHON deploy/vla/export_vla_llm_hidden_onnx.py
        ;;
    deploy-vla-trt)
        echo "================================"
        echo "VLA TensorRT Engine Build"
        echo "================================"
        # Action head uses FP16 by default (small, accurate).
        # LLM hidden-state engine uses FP32 for better accuracy on regression task.
        $PYTHON deploy/vla/export_action_head_trt.py && \
        $PYTHON deploy/vla/export_vla_llm_hidden_trt.py
        ;;
    deploy-vla-int8)
        echo "================================"
        echo "VLA INT8 TensorRT Engine Build"
        echo "================================"
        # 1. Generate calibration data from validation split.
        $PYTHON deploy/vla/calibrate_vla_inputs.py --num_samples 100 && \
        # 2. Build INT8 LLM engine.
        $PYTHON deploy/vla/export_vla_llm_hidden_trt.py --int8 \
            --output deploy/outputs/vla_llm_hidden_int8.trt && \
        # 3. Build INT8 action head engine (optional, small benefit).
        $PYTHON deploy/vla/export_action_head_trt.py --int8 \
            --output deploy/outputs/vla_action_head_int8.trt
        ;;
    run-vla-trt)
        echo "================================"
        echo "VLA TensorRT Inference"
        echo "================================"
        $PYTHON deploy/vla/run_trt_vla.py \
          "$@"
        ;;
    compare-vla-trt)
        echo "================================"
        echo "VLA PyTorch vs TensorRT"
        echo "================================"
        $PYTHON deploy/vla/compare_torch_trt.py \
          "$@"
        ;;
    deploy-vla-internvl3-onnx)
        echo "================================"
        echo "VLA InternVL3 ONNX Export"
        echo "================================"
        CHECKPOINT=${CHECKPOINT:-./outputs/nuscenes_mini_internvl3_vla/checkpoints/last.ckpt}
        $PYTHON deploy/vla/export_action_head_onnx.py \
          --checkpoint $CHECKPOINT \
          --output deploy/outputs/vla_action_head_internvl3.onnx && \
        $PYTHON deploy/vla/export_vla_llm_hidden_onnx.py \
          --checkpoint $CHECKPOINT \
          --output deploy/outputs/vla_llm_hidden_internvl3.onnx
        ;;
    deploy-vla-internvl3-trt)
        echo "================================"
        echo "VLA InternVL3 TensorRT Engine Build"
        echo "================================"
        $PYTHON deploy/vla/export_action_head_trt.py \
          --onnx_path deploy/outputs/vla_action_head_internvl3.onnx \
          --output deploy/outputs/vla_action_head_internvl3.trt && \
        $PYTHON deploy/vla/export_vla_llm_hidden_trt.py \
          --onnx_path deploy/outputs/vla_llm_hidden_internvl3.onnx \
          --output deploy/outputs/vla_llm_hidden_internvl3.trt
        ;;
    run-vla-internvl3-trt)
        echo "================================"
        echo "VLA InternVL3 TensorRT Inference"
        echo "================================"
        CHECKPOINT=${CHECKPOINT:-./outputs/nuscenes_mini_internvl3_vla/checkpoints/last.ckpt}
        $PYTHON deploy/vla/run_trt_vla.py \
          --checkpoint $CHECKPOINT \
          --vision_engine deploy/outputs/internvl3_vision_dynamic.trt \
          --action_engine deploy/outputs/vla_action_head_internvl3.trt \
          --llm_backend pytorch \
          "$@"
        ;;
    compare-vla-internvl3-trt)
        echo "================================"
        echo "VLA InternVL3 PyTorch vs TensorRT"
        echo "================================"
        CHECKPOINT=${CHECKPOINT:-./outputs/nuscenes_mini_internvl3_vla/checkpoints/last.ckpt}
        $PYTHON deploy/vla/compare_torch_trt.py \
          --checkpoint $CHECKPOINT \
          --vision_engine deploy/outputs/internvl3_vision_dynamic.trt \
          --action_engine deploy/outputs/vla_action_head_internvl3.trt \
          --llm_backend pytorch \
          "$@"
        ;;
    train-qwen2vl-vla)
        echo "================================"
        echo "Train Qwen2-VL VLA"
        echo "================================"
        $PYTHON scripts/train.py \
          --config configs/nuscenes_mini_qwen2vl_2b_vla.yaml \
          "$@"
        ;;
    eval-qwen2vl-vla)
        echo "================================"
        echo "Evaluate Qwen2-VL VLA"
        echo "================================"
        CHECKPOINT=${CHECKPOINT:-./outputs/nuscenes_mini_qwen2vl_2b_vla/checkpoints/last.ckpt}
        $PYTHON scripts/eval.py \
          --config configs/nuscenes_mini_qwen2vl_2b_vla.yaml \
          --checkpoint $CHECKPOINT \
          "$@"
        ;;
    smoke-qwen2vl-vla)
        echo "================================"
        echo "Qwen2-VL VLA Synthetic Smoke Test"
        echo "================================"
        $PYTHON scripts/smoke_test_qwen2vl_vla.py \
          "$@"
        ;;
    deploy-qwen2vl-vla-onnx)
        echo "================================"
        echo "Qwen2-VL VLA ONNX Export"
        echo "================================"
        CHECKPOINT=${CHECKPOINT:-./outputs/nuscenes_mini_qwen2vl_2b_vla/checkpoints/last.ckpt}
        $PYTHON deploy/vla/export_qwen2vl_action_head_onnx.py \
          --checkpoint $CHECKPOINT && \
        $PYTHON deploy/vla/export_qwen2vl_llm_hidden_onnx.py \
          --checkpoint $CHECKPOINT
        ;;
    deploy-qwen2vl-vla-trt)
        echo "================================"
        echo "Qwen2-VL VLA TensorRT Engine Build"
        echo "================================"
        # Qwen2-VL LLM uses FP32 engine to avoid FP16 layernorm overflow.
        $PYTHON deploy/vla/export_qwen2vl_action_head_trt.py \
          --onnx_path deploy/outputs/qwen2vl_vla_action_head.onnx \
          --output deploy/outputs/qwen2vl_vla_action_head.trt && \
        $PYTHON deploy/vla/export_qwen2vl_llm_hidden_trt.py \
          --onnx_path deploy/outputs/qwen2vl_vla_llm_hidden.onnx \
          --output deploy/outputs/qwen2vl_vla_llm_hidden.trt \
          --max_seq 8192 --opt_seq 4096
        ;;
    run-qwen2vl-vla-trt)
        echo "================================"
        echo "Qwen2-VL VLA TensorRT Inference"
        echo "================================"
        CHECKPOINT=${CHECKPOINT:-./outputs/nuscenes_mini_qwen2vl_2b_vla/checkpoints/last.ckpt}
        $PYTHON deploy/vla/run_trt_qwen2vl_vla.py \
          --checkpoint $CHECKPOINT \
          --llm_backend trt \
          "$@"
        ;;
    compare-qwen2vl-vla-trt)
        echo "================================"
        echo "Qwen2-VL VLA PyTorch vs TensorRT"
        echo "================================"
        CHECKPOINT=${CHECKPOINT:-./outputs/nuscenes_mini_qwen2vl_2b_vla/checkpoints/last.ckpt}
        $PYTHON deploy/vla/compare_torch_trt_qwen2vl_vla.py \
          --checkpoint $CHECKPOINT \
          --llm_backend trt \
          "$@"
        ;;
    deploy-internvl3-onnx)
        echo "================================"
        echo "InternVL3 ONNX Export"
        echo "================================"
        $PYTHON deploy/internvl3/export_vision_onnx.py && \
        $PYTHON deploy/internvl3/export_llm_onnx.py && \
        $PYTHON deploy/internvl3/export_llm_embed_onnx.py
        ;;
    deploy-internvl3-trt)
        echo "================================"
        echo "InternVL3 TensorRT Engine Build"
        echo "================================"
        $PYTHON deploy/internvl3/export_vision_trt.py && \
        $PYTHON deploy/internvl3/export_llm_trt.py && \
        $PYTHON deploy/internvl3/export_llm_embed_trt.py
        ;;
    run-internvl3-trt)
        echo "================================"
        echo "InternVL3 TensorRT Inference"
        echo "================================"
        $PYTHON deploy/internvl3/run_trt_internvl.py \
          --image_path $IMAGE \
          "$@"
        ;;
    compare-internvl3-trt)
        echo "================================"
        echo "InternVL3 PyTorch vs TensorRT"
        echo "================================"
        $PYTHON deploy/internvl3/compare_torch_trt.py \
          --image_path $IMAGE \
          "$@"
        ;;
    bench-internvl3-vision-latency)
        echo "================================"
        echo "InternVL3 Vision Latency Benchmark"
        echo "================================"
        $PYTHON deploy/internvl3/benchmark_vision_latency.py \
          --skip_pytorch \
          --onnx_path deploy/outputs/internvl3_vision_dynamic.onnx \
          --engine_path deploy/outputs/internvl3_vision_dynamic.trt \
          --onnx_provider CUDAExecutionProvider \
          --image_path $IMAGE \
          --warmup 50 \
          --iterations 200 \
          --output_json deploy/outputs/internvl3_vision_latency.json \
          "$@"
        ;;
    deploy-smolvlm2-256m-onnx)
        echo "================================"
        echo "SmolVLM2-256M ONNX Export"
        echo "================================"
        $PYTHON deploy/smolvlm/export_vision_onnx.py \
          --model_path /home/jovyan/work/vla_smolvlm2_256m/models/HuggingFaceTB/SmolVLM2-256M-Video-Instruct && \
        $PYTHON deploy/smolvlm/export_llm_embed_onnx.py \
          --model_path /home/jovyan/work/vla_smolvlm2_256m/models/HuggingFaceTB/SmolVLM2-256M-Video-Instruct
        ;;
    deploy-smolvlm2-256m-trt)
        echo "================================"
        echo "SmolVLM2-256M TensorRT Engine Build"
        echo "================================"
        $PYTHON deploy/smolvlm/export_vision_trt.py \
          --onnx_path deploy/outputs/smolvlm2_256m_vision.onnx \
          --output deploy/outputs/smolvlm2_256m_vision.trt && \
        $PYTHON deploy/smolvlm/export_llm_embed_trt.py \
          --onnx_path deploy/outputs/smolvlm2_256m_llm_embed.onnx \
          --output deploy/outputs/smolvlm2_256m_llm_embed.trt \
          --hidden_size 576 --max_seq 1024
        ;;
    run-smolvlm2-256m-trt)
        echo "================================"
        echo "SmolVLM2-256M TensorRT Inference"
        echo "================================"
        $PYTHON deploy/smolvlm/run_trt_smolvlm.py \
          --model_path /home/jovyan/work/vla_smolvlm2_256m/models/HuggingFaceTB/SmolVLM2-256M-Video-Instruct \
          --image_path $IMAGE \
          "$@"
        ;;
    compare-smolvlm2-256m-trt)
        echo "================================"
        echo "SmolVLM2-256M PyTorch vs TensorRT"
        echo "================================"
        $PYTHON deploy/smolvlm/compare_torch_trt.py \
          --model_path /home/jovyan/work/vla_smolvlm2_256m/models/HuggingFaceTB/SmolVLM2-256M-Video-Instruct \
          --image_path $IMAGE \
          "$@"
        ;;
    bench-smolvlm2-256m-vision-latency)
        echo "================================"
        echo "SmolVLM2-256M Vision Latency Benchmark"
        echo "================================"
        $PYTHON deploy/smolvlm/benchmark_vision_latency.py \
          --skip_pytorch \
          --model_path /home/jovyan/work/vla_smolvlm2_256m/models/HuggingFaceTB/SmolVLM2-256M-Video-Instruct \
          --onnx_provider CUDAExecutionProvider \
          --image_path $IMAGE \
          --warmup 20 \
          --iterations 80 \
          --output_json deploy/outputs/smolvlm2_256m_vision_latency.json \
          "$@"
        ;;
    deploy-smolvlm2-500m-onnx)
        echo "================================"
        echo "SmolVLM2-500M ONNX Export"
        echo "================================"
        $PYTHON deploy/smolvlm/export_vision_onnx.py \
          --model_path /home/jovyan/work/vla_smolvlm2_500m/models/HuggingFaceTB/SmolVLM2-500M-Video-Instruct && \
        $PYTHON deploy/smolvlm/export_llm_embed_onnx.py \
          --model_path /home/jovyan/work/vla_smolvlm2_500m/models/HuggingFaceTB/SmolVLM2-500M-Video-Instruct
        ;;
    deploy-smolvlm2-500m-trt)
        echo "================================"
        echo "SmolVLM2-500M TensorRT Engine Build"
        echo "================================"
        $PYTHON deploy/smolvlm/export_vision_trt.py \
          --onnx_path deploy/outputs/smolvlm2_500m_vision.onnx \
          --output deploy/outputs/smolvlm2_500m_vision.trt && \
        $PYTHON deploy/smolvlm/export_llm_embed_trt.py \
          --onnx_path deploy/outputs/smolvlm2_500m_llm_embed.onnx \
          --output deploy/outputs/smolvlm2_500m_llm_embed.trt \
          --hidden_size 960 --max_seq 1024
        ;;
    run-smolvlm2-500m-trt)
        echo "================================"
        echo "SmolVLM2-500M TensorRT Inference"
        echo "================================"
        $PYTHON deploy/smolvlm/run_trt_smolvlm.py \
          --model_path /home/jovyan/work/vla_smolvlm2_500m/models/HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
          --image_path $IMAGE \
          "$@"
        ;;
    compare-smolvlm2-500m-trt)
        echo "================================"
        echo "SmolVLM2-500M PyTorch vs TensorRT"
        echo "================================"
        $PYTHON deploy/smolvlm/compare_torch_trt.py \
          --model_path /home/jovyan/work/vla_smolvlm2_500m/models/HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
          --image_path $IMAGE \
          "$@"
        ;;
    bench-smolvlm2-500m-vision-latency)
        echo "================================"
        echo "SmolVLM2-500M Vision Latency Benchmark"
        echo "================================"
        $PYTHON deploy/smolvlm/benchmark_vision_latency.py \
          --skip_pytorch \
          --model_path /home/jovyan/work/vla_smolvlm2_500m/models/HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
          --onnx_provider CUDAExecutionProvider \
          --image_path $IMAGE \
          --warmup 20 \
          --iterations 80 \
          --output_json deploy/outputs/smolvlm2_500m_vision_latency.json \
          "$@"
        ;;
    deploy-qwen2vl-2b-onnx)
        echo "================================"
        echo "Qwen2-VL-2B ONNX Export"
        echo "================================"
        $PYTHON deploy/qwen2vl/export_vision_onnx.py --image_path $IMAGE && \
        $PYTHON deploy/qwen2vl/export_llm_embed_onnx.py
        ;;
    deploy-qwen2vl-2b-trt)
        echo "================================"
        echo "Qwen2-VL-2B TensorRT Engine Build"
        echo "================================"
        $PYTHON deploy/qwen2vl/export_vision_trt.py \
          --onnx_path deploy/outputs/qwen2vl_2b_vision.onnx \
          --output deploy/outputs/qwen2vl_2b_vision.trt \
          --image_path $IMAGE && \
        $PYTHON deploy/qwen2vl/export_llm_embed_trt.py \
          --onnx_path deploy/outputs/qwen2vl_2b_llm_embed.onnx \
          --output deploy/outputs/qwen2vl_2b_llm_embed.trt \
          --hidden_size 1536 --max_seq 2048
        ;;
    run-qwen2vl-2b-trt)
        echo "================================"
        echo "Qwen2-VL-2B TensorRT Inference"
        echo "================================"
        $PYTHON deploy/qwen2vl/run_trt_qwen2vl.py \
          --image_path $IMAGE \
          "$@"
        ;;
    compare-qwen2vl-2b-trt)
        echo "================================"
        echo "Qwen2-VL-2B PyTorch vs TensorRT"
        echo "================================"
        $PYTHON deploy/qwen2vl/compare_torch_trt.py \
          --image_path $IMAGE \
          "$@"
        ;;
    *)
        usage
        ;;
esac
