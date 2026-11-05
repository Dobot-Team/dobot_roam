#!/usr/bin/env python3
"""
Build TensorRT engine from InternVL vision ONNX.

Supports fp32 and fp16. Output: internvl3_vision.trt
"""
import argparse
import os
import sys
import warnings

import tensorrt as trt

warnings.filterwarnings("ignore", category=DeprecationWarning)

DEFAULT_ONNX = os.path.join(os.path.dirname(__file__), "..", "outputs", "internvl3_vision.onnx")
DEFAULT_OUTPUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "internvl3_vision.trt")


def build_engine(onnx_path: str, output_path: str, fp16: bool = False, workspace_mb: int = 4096,
                 min_batch: int = 1, opt_batch: int = 1, max_batch: int = 1):
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

    # Build optimization profile for input tensor.
    # For dynamic-batch ONNX, override the batch dimension with min/opt/max.
    profile = builder.create_optimization_profile()
    input_tensor = network.get_input(0)
    shape = list(input_tensor.shape)
    if shape[0] == -1 or min_batch != opt_batch or opt_batch != max_batch:
        # Dynamic or explicit range
        min_shape = [min_batch] + shape[1:]
        opt_shape = [opt_batch] + shape[1:]
        max_shape = [max_batch] + shape[1:]
        profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
        print(f"[Profile] {input_tensor.name}: min={min_shape} opt={opt_shape} max={max_shape}")
    else:
        static_shape = tuple(shape)
        profile.set_shape(input_tensor.name, static_shape, static_shape, static_shape)
        print(f"[Profile] {input_tensor.name}: static={static_shape}")
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
    parser = argparse.ArgumentParser(description="Build TensorRT engine for InternVL3 vision encoder")
    parser.add_argument("--onnx_path", type=str, default=DEFAULT_ONNX)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 precision")
    parser.add_argument("--workspace_mb", type=int, default=4096)
    parser.add_argument("--min_batch", type=int, default=1, help="Minimum batch size for dynamic profile")
    parser.add_argument("--opt_batch", type=int, default=1, help="Optimal batch size for dynamic profile")
    parser.add_argument("--max_batch", type=int, default=1, help="Maximum batch size for dynamic profile")
    args = parser.parse_args()

    sys.exit(build_engine(args.onnx_path, args.output, args.fp16, args.workspace_mb,
                          args.min_batch, args.opt_batch, args.max_batch))


if __name__ == "__main__":
    main()
