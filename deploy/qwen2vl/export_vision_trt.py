#!/usr/bin/env python3
"""
Build TensorRT engine from Qwen2-VL vision ONNX.
"""
import argparse
import os
import sys
import warnings

import numpy as np
import tensorrt as trt
import torch
from PIL import Image
from transformers import Qwen2VLProcessor

warnings.filterwarnings("ignore", category=DeprecationWarning)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "Qwen", "Qwen2-VL-2B-Instruct")
# Export for a single NuScenes CAM_FRONT image; VLA concatenates 4 of these per-frame.
DEFAULT_IMAGE_PATHS = [
    "/root/nuscenes/samples/CAM_FRONT/n008-2018-08-27-11-48-51-0400__CAM_FRONT__1535385105412404.jpg",
]
DEFAULT_ONNX = os.path.join(PROJECT_ROOT, "deploy", "outputs", "qwen2vl_2b_vision.onnx")
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "deploy", "outputs", "qwen2vl_2b_vision.trt")


def get_sample_image_grid_thw(model_path: str, image_paths):
    """Compute the image_grid_thw for the fixed sample image(s) used during ONNX export."""
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    processor = Qwen2VLProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    images = [Image.open(p).convert("RGB") for p in image_paths]
    content = [{"type": "image", "image": img} for img in images]
    content.append({"type": "text", "text": "Describe"})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, return_tensors="pt", padding=True)
    return inputs["image_grid_thw"].to(torch.int64).numpy()


def build_engine(onnx_path: str, output_path: str, model_path: str = DEFAULT_MODEL_PATH,
                 image_paths=DEFAULT_IMAGE_PATHS, fp16: bool = False, workspace_mb: int = 4096):
    if not os.path.exists(onnx_path):
        print(f"[Error] ONNX not found: {onnx_path}")
        return 1

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    print(f"[Parse] Reading ONNX: {onnx_path}")
    if not parser.parse_from_file(onnx_path):
        for i in range(parser.num_errors):
            print(f"[ONNX Parse Error] {parser.get_error(i)}")
        return 1

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * (1 << 20))
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("[Build] FP16 mode enabled.")

    # Compute the fixed image_grid_thw values for the sample image(s).
    grid_thw = get_sample_image_grid_thw(model_path, image_paths)

    # Fixed-shape vision encoder. image_grid_thw is a shape input tensor and
    # must be bound with set_shape_input.
    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        shape = tuple(input_tensor.shape)
        if input_tensor.is_shape_tensor:
            # TensorRT shape input values must be provided as a flat list of ints.
            shape_values = grid_thw.reshape(-1).astype(np.int32).tolist()
            profile.set_shape_input(input_tensor.name, shape_values, shape_values, shape_values)
            print(f"[Profile] {input_tensor.name} (shape tensor): {shape_values}")
        else:
            profile.set_shape(input_tensor.name, shape, shape, shape)
            print(f"[Profile] {input_tensor.name}: {shape}")
    config.add_optimization_profile(profile)

    print(f"[Build] Building TensorRT engine (workspace={workspace_mb}MB)...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("[Error] Engine build failed.")
        return 1

    with open(output_path, "wb") as f:
        f.write(serialized)
    del serialized

    print(f"[Build] TensorRT engine saved: {output_path}")
    print(f"[Build] Engine size: {os.path.getsize(output_path) / (1<<20):.1f} MB")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Build TensorRT engine for Qwen2-VL vision encoder")
    parser.add_argument("--onnx_path", type=str, default=DEFAULT_ONNX)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--image_paths", type=str, nargs="+", default=DEFAULT_IMAGE_PATHS,
                        help="Sample image(s) used to compute the fixed image_grid_thw shape tensor")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--workspace_mb", type=int, default=4096)
    args = parser.parse_args()

    sys.exit(build_engine(args.onnx_path, args.output, args.model_path, args.image_paths,
                          args.fp16, args.workspace_mb))


if __name__ == "__main__":
    main()
