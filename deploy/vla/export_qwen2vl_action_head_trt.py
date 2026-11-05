#!/usr/bin/env python3
"""
Build TensorRT engine from Qwen2-VL VLA action head ONNX.

Supports FP32, FP16 (default), and INT8.
"""
import argparse
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_ONNX = os.path.join(_PROJECT_ROOT, "deploy", "outputs", "qwen2vl_vla_action_head.onnx")
DEFAULT_OUTPUT = os.path.join(_PROJECT_ROOT, "deploy", "outputs", "qwen2vl_vla_action_head.trt")
DEFAULT_CALIB_CACHE = os.path.join(_PROJECT_ROOT, "deploy", "outputs", "qwen2vl_vla_action_head.cache")


def _init_cuda(must_succeed: bool = False):
    """Initialize cuda-python driver API and make a context current."""
    try:
        from cuda.bindings import driver as cuda
    except ImportError:
        if must_succeed:
            raise
        return None

    err = cuda.cuInit(0)
    if isinstance(err, tuple):
        err = err[0]
    if err != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuInit failed: {err}")
    err, dev = cuda.cuDeviceGet(0)
    if err != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuDeviceGet failed: {err}")
    err, ctx = cuda.cuCtxCreate(None, 0, dev)
    if err != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuCtxCreate failed: {err}")
    return cuda


_CUDA_MODULE = _init_cuda(must_succeed=False)

import tensorrt as trt  # noqa: E402


class Int8Calibrator(trt.IInt8EntropyCalibrator2):
    """TensorRT INT8 calibrator using synthetic query_hidden samples."""

    def __init__(self, cache_path: str, hidden_size: int, batch_size: int = 8, num_batches: int = 20):
        super().__init__()
        self.cache_path = cache_path
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.current_index = 0

        rng = np.random.default_rng(42)
        self.data = rng.normal(loc=0.0, scale=0.5, size=(num_batches * batch_size, 24, hidden_size)).astype(np.float32)

        if _CUDA_MODULE is None:
            raise RuntimeError("cuda-python is not available, cannot create INT8 calibrator.")
        self.cuda = _CUDA_MODULE
        _, self.buffer = self.cuda.cuMemAlloc(self.data[:batch_size].nbytes)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_index >= self.num_batches:
            return None
        batch = self.data[self.current_index * self.batch_size:(self.current_index + 1) * self.batch_size]
        self.current_index += 1
        self.cuda.cuMemcpyHtoD(self.buffer, batch.ctypes.data, batch.nbytes)
        return [int(self.buffer)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_path):
            print(f"[Calibrator] Reading calibration cache: {self.cache_path}")
            with open(self.cache_path, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        print(f"[Calibrator] Writing calibration cache: {self.cache_path}")
        with open(self.cache_path, "wb") as f:
            f.write(cache)


def build_engine(onnx_path: str, output_path: str, hidden_size: int = 1536,
                 fp16: bool = True, int8: bool = False,
                 calib_cache_path: str = DEFAULT_CALIB_CACHE, workspace_mb: int = 512):
    if not os.path.exists(onnx_path):
        print(f"[Error] ONNX not found: {onnx_path}")
        return 1

    if int8 and _CUDA_MODULE is None:
        print(f"[Error] INT8 calibration requires cuda-python. Install it with: "
              f"{sys.executable} -m pip install cuda-python")
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
    if int8:
        config.set_flag(trt.BuilderFlag.INT8)
        print("[Build] INT8 mode enabled.")
        calibrator = Int8Calibrator(calib_cache_path, hidden_size=hidden_size, batch_size=8, num_batches=20)
        config.int8_calibrator = calibrator

    profile = builder.create_optimization_profile()
    profile.set_shape("query_hidden", (1, 24, hidden_size), (1, 24, hidden_size), (8, 24, hidden_size))
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
    parser = argparse.ArgumentParser(description="Build TensorRT engine for Qwen2-VL VLA action head")
    parser.add_argument("--onnx_path", type=str, default=DEFAULT_ONNX)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--hidden_size", type=int, default=1536)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--int8", action="store_true", help="Enable INT8 quantization")
    parser.add_argument("--calib_cache", type=str, default=DEFAULT_CALIB_CACHE,
                        help="Path to write/read TensorRT calibration cache")
    parser.add_argument("--workspace_mb", type=int, default=512)
    args = parser.parse_args()

    fp16 = args.fp16 and not args.fp32
    sys.exit(build_engine(args.onnx_path, args.output, args.hidden_size, fp16=fp16, int8=args.int8,
                          calib_cache_path=args.calib_cache, workspace_mb=args.workspace_mb))


if __name__ == "__main__":
    main()
