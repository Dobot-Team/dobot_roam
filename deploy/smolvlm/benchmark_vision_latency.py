#!/usr/bin/env python3
"""
Benchmark SmolVLM2 vision encoder (vision_model + connector) inference latency.

Input shape:  pixel_values (batch, 9, 3, 512, 512) float32
Output shape: image_hidden_states (batch, 9, 64, hidden_size) float32

Supports:
  - ONNX Runtime (CPU / CUDA / TensorRT execution provider)
  - Native TensorRT engine
  - PyTorch eager baseline (optional)

The script is board-friendly: dependencies are imported lazily.

Default input: random tensor of shape (1, 9, 3, 512, 512) float32.
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

DEFAULT_MODEL_PATH = "/home/jovyan/work/vla_smolvlm2_256m/models/HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
DEFAULT_IMAGE = os.path.join(os.path.dirname(__file__), "..", "..", "20260610151154.jpg")


def resolve_default_onnx(model_path: str) -> str:
    # SmolVLM2 vision ONNX only supports a single outer sample due to the
    # vision_model's internal position-embedding split along num_images=9.
    if "256M" in model_path:
        name = "smolvlm2_256m_vision.onnx"
    elif "500M" in model_path:
        name = "smolvlm2_500m_vision.onnx"
    else:
        name = "smolvlm2_custom_vision.onnx"
    return os.path.join(os.path.dirname(__file__), "..", "outputs", name)


def resolve_default_engine(model_path: str) -> str:
    # Use the fixed batch=1 TensorRT engine; multi-sample benchmarks run
    # sequential inferences.
    if "256M" in model_path:
        name = "smolvlm2_256m_vision.trt"
    elif "500M" in model_path:
        name = "smolvlm2_500m_vision.trt"
    else:
        name = "smolvlm2_custom_vision.trt"
    return os.path.join(os.path.dirname(__file__), "..", "outputs", name)


def now_ms() -> float:
    return time.perf_counter() * 1000.0


def gpu_time_ms() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return now_ms()


def stats_ms(latencies_ms: List[float], batch_size: int = 1) -> Dict[str, float]:
    arr = np.array(latencies_ms, dtype=np.float64)
    avg_ms = float(np.mean(arr))
    return {
        "n": int(len(arr)),
        "avg_ms": avg_ms,
        "avg_ms_per_sample": avg_ms / batch_size,
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "std_ms": float(np.std(arr)),
        "throughput_samples_per_s": float(batch_size * 1000.0 / avg_ms),
    }


def prepare_input(image_path: Optional[str], seed: int, device: torch.device,
                  batch_size: int = 1) -> Tuple[np.ndarray, str]:
    if image_path and os.path.exists(image_path):
        from PIL import Image
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(
            DEFAULT_MODEL_PATH, trust_remote_code=True, local_files_only=True
        )
        img = Image.open(image_path).convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image", "image": img}]}]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        pv = inputs["pixel_values"].to(torch.float32)
        if batch_size > 1:
            pv = pv.repeat(batch_size, 1, 1, 1, 1)
        return pv.cpu().numpy(), "real_image"
    rng = np.random.RandomState(seed)
    x = rng.randn(batch_size, 9, 3, 512, 512).astype(np.float32)
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
    input_shape = tuple(session.get_inputs()[0].shape)
    print(f"[ONNX] Provider: {provider}")
    print(f"[ONNX] Input name: {input_name}, shape: {input_shape}")

    # SmolVLM2 vision model assumes the flattened batch dimension equals num_images (9),
    # so the exported ONNX only supports a single outer sample. For batch>1 we run
    # sequential inferences and measure total time.
    single_x = x[:1]
    n_samples = x.shape[0]
    if n_samples > 1:
        print(f"[ONNX] Will run {n_samples} sequential inferences per iteration.")

    for _ in range(warmup):
        _ = session.run(None, {input_name: single_x})[0]
    if provider == "CUDAExecutionProvider" and torch.cuda.is_available():
        torch.cuda.synchronize()

    latencies = []
    if n_samples > 1:
        for _ in range(iterations):
            t0 = gpu_time_ms()
            for _ in range(n_samples):
                _ = session.run(None, {input_name: single_x})[0]
            t1 = gpu_time_ms()
            latencies.append(t1 - t0)
    else:
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
        self.dynamic_batch = self.engine_input_shape[0] == -1
        print(f"[TRT] Engine loaded: input_shape={self.engine_input_shape} output_shape={self.engine_output_shape} dtype={self.dtype}")

    def infer(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(self.dtype)
        batch_size = x.shape[0]

        input_shape = list(x.shape)
        if self.dynamic_batch:
            self.context.set_input_shape(self.input_name, input_shape)
        else:
            assert x.shape[0] == self.engine_input_shape[0], \
                f"Static engine expects batch={self.engine_input_shape[0]}, got {x.shape[0]}"

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

    # SmolVLM2 static engine only supports batch=1; run sequential inferences for batch>1.
    single_x = x[:1]
    n_samples = x.shape[0]
    if not runner.dynamic_batch and n_samples > 1:
        print(f"[TRT] Static batch=1 engine; will run {n_samples} sequential inferences per iteration.")

    for _ in range(warmup):
        runner.infer(single_x)

    latencies = []
    if not runner.dynamic_batch and n_samples > 1:
        for _ in range(iterations):
            t0 = gpu_time_ms()
            for _ in range(n_samples):
                runner.infer(single_x)
            t1 = gpu_time_ms()
            latencies.append(t1 - t0)
    else:
        for _ in range(iterations):
            t0 = gpu_time_ms()
            runner.infer(x)
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
class SmolVLMVisionWrapper(torch.nn.Module):
    def __init__(self, vision_model, connector, patch_size: int):
        super().__init__()
        self.vision_model = vision_model
        self.connector = connector
        self.patch_size = patch_size
        self.num_patches_h = 512 // patch_size
        self.num_patches_w = 512 // patch_size

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size, num_images, C, H, W = pixel_values.shape
        pixel_values = pixel_values.view(batch_size * num_images, C, H, W)
        patch_attention_mask = torch.ones(
            batch_size * num_images,
            self.num_patches_h,
            self.num_patches_w,
            dtype=torch.bool,
            device=pixel_values.device,
        )
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            patch_attention_mask=patch_attention_mask,
        )
        image_hidden_states = self.connector(vision_outputs.last_hidden_state)
        return image_hidden_states.view(batch_size, num_images, *image_hidden_states.shape[1:])


def load_pt_model(model_path: str, device: torch.device):
    from transformers import AutoModelForImageTextToText
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        device_map=None,
        local_files_only=True,
    )
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False
    patch_size = model.model.vision_model.config.patch_size
    wrapper = SmolVLMVisionWrapper(
        model.model.vision_model,
        model.model.connector,
        patch_size,
    ).to(device).eval()
    return wrapper


def benchmark_pytorch(model_path: str, x: np.ndarray, device: torch.device,
                      warmup: int, iterations: int) -> Optional[Dict[str, Any]]:
    try:
        from transformers import AutoModelForImageTextToText  # noqa: F401
    except ImportError:
        print("[PyTorch] transformers not installed, skipping.")
        return None

    if not os.path.exists(model_path):
        print(f"[PyTorch] Model path not found: {model_path}, skipping.")
        return None

    print(f"[PyTorch] Loading model from {model_path}...")
    wrapper = load_pt_model(model_path, device)
    x_tensor = torch.from_numpy(x).to(device)

    for _ in range(warmup):
        with torch.no_grad():
            _ = wrapper(x_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()

    latencies = []
    with torch.no_grad():
        for _ in range(iterations):
            t0 = gpu_time_ms()
            _ = wrapper(x_tensor)
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
    print(f"  avg per sample   : {result['avg_ms_per_sample']:.3f} ms")
    print(f"  min latency      : {result['min_ms']:.3f} ms")
    print(f"  max latency      : {result['max_ms']:.3f} ms")
    print(f"  p50 latency      : {result['p50_ms']:.3f} ms")
    print(f"  p95 latency      : {result['p95_ms']:.3f} ms")
    print(f"  p99 latency      : {result['p99_ms']:.3f} ms")
    print(f"  std              : {result['std_ms']:.3f} ms")
    print(f"  throughput       : {result['throughput_samples_per_s']:.2f} samples/s")


def main():
    parser = argparse.ArgumentParser(description="Benchmark SmolVLM2 vision encoder latency")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="Local SmolVLM2 model path (for PyTorch baseline)")
    parser.add_argument("--onnx_path", type=str, default=None, help="ONNX model path")
    parser.add_argument("--engine_path", type=str, default=None, help="TensorRT engine path")
    parser.add_argument("--image_path", type=str, default=None, help="Real image path; if not set, use random tensor")
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
                        help="Outer batch size (number of samples per inference, each sample has 9 image patches)")
    args = parser.parse_args()

    onnx_path = args.onnx_path or resolve_default_onnx(args.model_path)
    engine_path = args.engine_path or resolve_default_engine(args.model_path)

    device = torch.device(args.device)
    x, input_type = prepare_input(args.image_path, args.seed, device, args.batch_size)
    print(f"[Config] Input shape: {x.shape}, dtype: {x.dtype}, type: {input_type}")
    print(f"[Config] Batch size: {args.batch_size}, Warmup: {args.warmup}, Iterations: {args.iterations}, Device: {device}")

    results = {
        "config": {
            "input_shape": list(x.shape),
            "input_type": input_type,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "batch_size": args.batch_size,
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
        r = benchmark_onnx(onnx_path, x, args.onnx_provider, args.warmup, args.iterations)
        if r:
            r["input_type"] = input_type
            print_result("ONNX Runtime", r)
            results["backends"].append(r)

    if not args.skip_trt:
        r = benchmark_trt(engine_path, x, args.warmup, args.iterations)
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
