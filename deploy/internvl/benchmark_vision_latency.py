#!/usr/bin/env python3
"""
Benchmark InternVL2.5 vision encoder (ViT) inference latency.

Supports three backends:
  - ONNX Runtime (CPU / CUDA / TensorRT execution provider)
  - Native TensorRT engine
  - PyTorch eager baseline (optional)

The script is designed to be board-friendly: dependencies are imported lazily,
and missing backends are skipped with a warning rather than crashing.

Default input: random tensor of shape (1, 3, 448, 448) float32.
Use --image_path to benchmark real image preprocessing + inference end-to-end.
"""
import argparse
import json
import os
import sys
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

DEFAULT_MODEL_PATH = "/home/jovyan/work/vla_internvl2/models/OpenGVLab/InternVL2_5-1B"
DEFAULT_ONNX = os.path.join(os.path.dirname(__file__), "..", "outputs", "internvl2_vision.onnx")
DEFAULT_ENGINE = os.path.join(os.path.dirname(__file__), "..", "outputs", "internvl2_vision.trt")
DEFAULT_IMAGE = os.path.join(os.path.dirname(__file__), "..", "..", "20260610151154.jpg")


def now_ms() -> float:
    return time.perf_counter() * 1000.0


def gpu_time_ms() -> float:
    """Return synchronized GPU time in ms if CUDA is available, else host time."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return now_ms()


def stats_ms(latencies_ms: List[float], batch_size: int = 1) -> Dict[str, float]:
    arr = np.array(latencies_ms, dtype=np.float64)
    avg_ms = float(np.mean(arr))
    return {
        "n": int(len(arr)),
        "avg_ms": avg_ms,
        "avg_ms_per_image": avg_ms / batch_size,
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "std_ms": float(np.std(arr)),
        "throughput_imgs_per_s": float(batch_size * 1000.0 / avg_ms),
    }


def build_transform(input_size: int = 448):
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize(input_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def prepare_input(image_path: Optional[str], input_size: int, seed: int, device: torch.device,
                   batch_size: int = 1) -> Tuple[np.ndarray, str]:
    if image_path and os.path.exists(image_path):
        from PIL import Image
        transform = build_transform(input_size)
        img = Image.open(image_path).convert("RGB")
        x = transform(img).unsqueeze(0).to(torch.float32)
        if batch_size > 1:
            x = x.repeat(batch_size, 1, 1, 1)
        return x.cpu().numpy(), "real_image"
    rng = np.random.RandomState(seed)
    x = rng.randn(batch_size, 3, input_size, input_size).astype(np.float32)
    return x, "random"


# --------------------------------------------------------------------------- #
# ONNX Runtime benchmark
# --------------------------------------------------------------------------- #
def benchmark_onnx(onnx_path: str, x: np.ndarray, provider: str,
                   warmup: int, iterations: int) -> Optional[Dict[str, Any]]:
    try:
        import onnxruntime as ort
    except ImportError:
        print("[ONNX] onnxruntime not installed, skipping.")
        return None

    if not os.path.exists(onnx_path):
        print(f"[ONNX] Model not found: {onnx_path}, skipping.")
        return None

    available = ort.get_available_providers()
    if provider not in available:
        print(f"[ONNX] Requested provider '{provider}' not available. Available: {available}")
        if provider == "CUDAExecutionProvider" and "CPUExecutionProvider" in available:
            provider = "CPUExecutionProvider"
        else:
            return None

    session = ort.InferenceSession(onnx_path, providers=[provider])
    input_name = session.get_inputs()[0].name
    print(f"[ONNX] Provider: {provider}")
    print(f"[ONNX] Input name: {input_name}, shape: {session.get_inputs()[0].shape}")

    # Warmup
    for _ in range(warmup):
        _ = session.run(None, {input_name: x})[0]
    if provider == "CUDAExecutionProvider" and torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark
    latencies = []
    for _ in range(iterations):
        t0 = gpu_time_ms()
        _ = session.run(None, {input_name: x})[0]
        t1 = gpu_time_ms()
        latencies.append(t1 - t0)

    result = stats_ms(latencies, x.shape[0])
    result["backend"] = "onnxruntime"
    result["provider"] = provider
    result["batch_size"] = x.shape[0]
    return result


# --------------------------------------------------------------------------- #
# TensorRT benchmark
# --------------------------------------------------------------------------- #
class TRTVisionRunner:
    def __init__(self, engine_path: str):
        import tensorrt as trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)
        self.engine_input_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        self.engine_output_shape = tuple(self.engine.get_tensor_shape(self.output_name))
        self.dtype = trt.nptype(self.engine.get_tensor_dtype(self.input_name))
        print(f"[TRT] Engine loaded: input_shape={self.engine_input_shape} output_shape={self.engine_output_shape} dtype={self.dtype}")

    def infer(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(self.dtype)
        batch_size = x.shape[0]

        # Handle dynamic batch: set actual input shape and compute output shape
        input_shape = list(x.shape)
        if self.engine_input_shape[0] == -1:
            self.context.set_input_shape(self.input_name, input_shape)

        out_shape = list(self.engine_output_shape)
        if out_shape[0] == -1:
            out_shape[0] = batch_size
        out = np.empty(out_shape, dtype=self.dtype)

        d_input = torch.from_numpy(x).cuda()
        d_output = torch.from_numpy(out).cuda()

        self.context.set_tensor_address(self.input_name, d_input.data_ptr())
        self.context.set_tensor_address(self.output_name, d_output.data_ptr())
        self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()

        return d_output.cpu().numpy().astype(np.float32)


def benchmark_trt(engine_path: str, x: np.ndarray, warmup: int, iterations: int) -> Optional[Dict[str, Any]]:
    try:
        import tensorrt as trt  # noqa: F401
    except ImportError:
        print("[TRT] tensorrt not installed, skipping.")
        return None

    if not os.path.exists(engine_path):
        print(f"[TRT] Engine not found: {engine_path}, skipping.")
        return None

    runner = TRTVisionRunner(engine_path)

    # Warmup
    for _ in range(warmup):
        _ = runner.infer(x)

    # Benchmark
    latencies = []
    for _ in range(iterations):
        t0 = gpu_time_ms()
        _ = runner.infer(x)
        t1 = gpu_time_ms()
        latencies.append(t1 - t0)

    result = stats_ms(latencies, x.shape[0])
    result["backend"] = "tensorrt"
    result["provider"] = "tensorrt"
    result["batch_size"] = x.shape[0]
    return result


# --------------------------------------------------------------------------- #
# PyTorch benchmark
# --------------------------------------------------------------------------- #
def load_pt_model(model_path: str, device: torch.device):
    from transformers import AutoModel
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        device_map=None,
        low_cpu_mem_usage=True,
        local_files_only=True,
        use_flash_attn=False,
    )
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False
    return model


def benchmark_pytorch(model_path: str, x: np.ndarray, device: torch.device,
                      warmup: int, iterations: int) -> Optional[Dict[str, Any]]:
    try:
        from transformers import AutoModel  # noqa: F401
    except ImportError:
        print("[PyTorch] transformers not installed, skipping.")
        return None

    if not os.path.exists(model_path):
        print(f"[PyTorch] Model path not found: {model_path}, skipping.")
        return None

    print(f"[PyTorch] Loading model from {model_path}...")
    model = load_pt_model(model_path, device)
    x_tensor = torch.from_numpy(x).to(device)

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model.extract_feature(x_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    latencies = []
    with torch.no_grad():
        for _ in range(iterations):
            t0 = gpu_time_ms()
            _ = model.extract_feature(x_tensor)
            t1 = gpu_time_ms()
            latencies.append(t1 - t0)

    result = stats_ms(latencies, x.shape[0])
    result["backend"] = "pytorch"
    result["provider"] = device.type
    result["batch_size"] = x.shape[0]
    return result


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def print_result(name: str, result: Dict[str, Any]):
    print(f"\n[{name}]")
    print(f"  backend          : {result.get('backend', 'N/A')}")
    print(f"  provider/device  : {result.get('provider', 'N/A')}")
    print(f"  input type       : {result.get('input_type', 'N/A')}")
    print(f"  batch size       : {result.get('batch_size', 1)}")
    print(f"  iterations       : {result['n']}")
    print(f"  avg latency      : {result['avg_ms']:.3f} ms")
    print(f"  avg per image    : {result['avg_ms_per_image']:.3f} ms")
    print(f"  min latency      : {result['min_ms']:.3f} ms")
    print(f"  max latency      : {result['max_ms']:.3f} ms")
    print(f"  p50 latency      : {result['p50_ms']:.3f} ms")
    print(f"  p95 latency      : {result['p95_ms']:.3f} ms")
    print(f"  p99 latency      : {result['p99_ms']:.3f} ms")
    print(f"  std              : {result['std_ms']:.3f} ms")
    print(f"  throughput       : {result['throughput_imgs_per_s']:.2f} images/s")


def main():
    parser = argparse.ArgumentParser(description="Benchmark InternVL2.5 vision encoder latency")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="Local InternVL model path (for PyTorch baseline)")
    parser.add_argument("--onnx_path", type=str, default=DEFAULT_ONNX, help="ONNX model path")
    parser.add_argument("--engine_path", type=str, default=DEFAULT_ENGINE, help="TensorRT engine path")
    parser.add_argument("--image_path", type=str, default=None, help="Real image path; if not set, use random tensor")
    parser.add_argument("--input_size", type=int, default=448, help="Input image size")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=100, help="Benchmark iterations")
    parser.add_argument("--onnx_provider", type=str, default="CUDAExecutionProvider",
                        choices=["CPUExecutionProvider", "CUDAExecutionProvider", "TensorrtExecutionProvider"],
                        help="ONNX Runtime execution provider")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for synthetic input")
    parser.add_argument("--skip_pytorch", action="store_true", help="Skip PyTorch baseline")
    parser.add_argument("--skip_onnx", action="store_true", help="Skip ONNX benchmark")
    parser.add_argument("--skip_trt", action="store_true", help="Skip TensorRT benchmark")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="PyTorch device")
    parser.add_argument("--output_json", type=str, default=None, help="Save results to JSON file")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size (number of images per inference)")
    args = parser.parse_args()

    device = torch.device(args.device)
    x, input_type = prepare_input(args.image_path, args.input_size, args.seed, device, args.batch_size)
    print(f"[Config] Input shape: {x.shape}, dtype: {x.dtype}, type: {input_type}")
    print(f"[Config] Batch size: {args.batch_size}, Warmup: {args.warmup}, Iterations: {args.iterations}, Device: {device}")

    results = {
        "config": {
            "input_shape": list(x.shape),
            "input_type": input_type,
            "input_size": args.input_size,
            "batch_size": args.batch_size,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "device": str(device),
        },
        "backends": [],
    }

    if not args.skip_pytorch:
        r = benchmark_pytorch(args.model_path, x, device, args.warmup, args.iterations)
        if r:
            r["input_type"] = input_type
            print_result("PyTorch", r)
            results["backends"].append(r)

    if not args.skip_onnx:
        r = benchmark_onnx(args.onnx_path, x, args.onnx_provider, args.warmup, args.iterations)
        if r:
            r["input_type"] = input_type
            print_result("ONNX Runtime", r)
            results["backends"].append(r)

    if not args.skip_trt:
        r = benchmark_trt(args.engine_path, x, args.warmup, args.iterations)
        if r:
            r["input_type"] = input_type
            print_result("TensorRT", r)
            results["backends"].append(r)

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n[Save] Results written to {args.output_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
