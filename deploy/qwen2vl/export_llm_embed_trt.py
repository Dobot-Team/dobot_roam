#!/usr/bin/env python3
"""
Build TensorRT engine from Qwen2-VL LLM (inputs_embeds mode) ONNX.
"""
import argparse
import os
import sys
import warnings

import tensorrt as trt

warnings.filterwarnings("ignore", category=DeprecationWarning)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_ONNX = os.path.join(PROJECT_ROOT, "deploy", "outputs", "qwen2vl_2b_llm_embed.onnx")
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "deploy", "outputs", "qwen2vl_2b_llm_embed.trt")


def build_engine(onnx_path: str, output_path: str, hidden_size: int = 1536,
                 min_seq: int = 1, opt_seq: int = 512, max_seq: int = 2048,
                 fp16: bool = False, workspace_mb: int = 4096):
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

    profile = builder.create_optimization_profile()
    profile.set_shape("inputs_embeds", (1, min_seq, hidden_size), (1, opt_seq, hidden_size), (1, max_seq, hidden_size))
    profile.set_shape("attention_mask", (1, min_seq), (1, opt_seq), (1, max_seq))
    # position_ids: (3, batch, seq_len)
    profile.set_shape("position_ids", (3, 1, min_seq), (3, 1, opt_seq), (3, 1, max_seq))
    config.add_optimization_profile(profile)

    print(f"[Profile] inputs_embeds: min=(1,{min_seq},{hidden_size}) opt=(1,{opt_seq},{hidden_size}) max=(1,{max_seq},{hidden_size})")
    print(f"[Profile] attention_mask: min=(1,{min_seq}) opt=(1,{opt_seq}) max=(1,{max_seq})")
    print(f"[Profile] position_ids: min=(3,1,{min_seq}) opt=(3,1,{opt_seq}) max=(3,1,{max_seq})")

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
    parser = argparse.ArgumentParser(description="Build TensorRT engine for Qwen2-VL LLM (embed mode)")
    parser.add_argument("--onnx_path", type=str, default=DEFAULT_ONNX)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--hidden_size", type=int, default=1536)
    parser.add_argument("--min_seq", type=int, default=1)
    parser.add_argument("--opt_seq", type=int, default=512)
    parser.add_argument("--max_seq", type=int, default=2048)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--workspace_mb", type=int, default=4096)
    args = parser.parse_args()

    sys.exit(build_engine(args.onnx_path, args.output, args.hidden_size, args.min_seq, args.opt_seq, args.max_seq,
                          args.fp16, args.workspace_mb))


if __name__ == "__main__":
    main()
