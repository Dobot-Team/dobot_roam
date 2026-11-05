#!/usr/bin/env python3
"""
Verify TensorRT vision engine against PyTorch and ONNX.

Runs random + real image inputs and compares TRT output to CPU references.
"""
import argparse
import os
import sys
import warnings

import numpy as np
import onnxruntime as ort
import tensorrt as trt
import torch
from PIL import Image
from transformers import AutoModel

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

DEFAULT_MODEL_PATH = "/home/jovyan/work/vla_internvl2/models/OpenGVLab/InternVL2_5-1B"
DEFAULT_ONNX = os.path.join(os.path.dirname(__file__), "..", "outputs", "internvl2_vision.onnx")
DEFAULT_ENGINE = os.path.join(os.path.dirname(__file__), "..", "outputs", "internvl2_vision.trt")
DEFAULT_IMAGE = os.path.join(os.path.dirname(__file__), "..", "..", "20260610151154.jpg")


def build_transform(input_size: int = 448):
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize(input_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_pt_model(model_path: str):
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        device_map=None,
        low_cpu_mem_usage=True,
        local_files_only=True,
        use_flash_attn=False,
    )
    model.eval().cpu()
    return model


def compare_outputs(ref: np.ndarray, trt_out: np.ndarray, tag: str):
    diff = np.abs(ref - trt_out)
    max_err = float(diff.max())
    mean_err = float(diff.mean())
    rel_err = max_err / (np.abs(ref).max() + 1e-12)
    cos_sim = float(np.sum(ref * trt_out) / (np.linalg.norm(ref) * np.linalg.norm(trt_out) + 1e-12))

    print(f"[{tag}] max_abs_error={max_err:.6e} mean_abs_error={mean_err:.6e} rel_error={rel_err:.6e} cosine_sim={cos_sim:.8f}")

    ok = True
    # TensorRT uses fused GPU kernels that introduce larger numeric differences than eager CPU.
    # For vision encoders, relative error < 5% and cosine similarity > 0.999 are empirically
    # sufficient to preserve downstream LLM token generation (verified separately in end-to-end test).
    if max_err >= 5e-1:
        print(f"[{tag}] FAIL: max_abs_error >= 0.5")
        ok = False
    if mean_err >= 1e-2:
        print(f"[{tag}] FAIL: mean_abs_error >= 1e-2")
        ok = False
    if rel_err >= 5e-2:
        print(f"[{tag}] FAIL: relative_error >= 5e-2")
        ok = False
    if cos_sim <= 0.999:
        print(f"[{tag}] FAIL: cosine_similarity <= 0.999")
        ok = False
    return ok


class TRTVisionRunner:
    def __init__(self, engine_path: str):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)
        self.input_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        self.output_shape = tuple(self.engine.get_tensor_shape(self.output_name))
        self.dtype = trt.nptype(self.engine.get_tensor_dtype(self.input_name))
        print(f"[TRT] Engine loaded: input_shape={self.input_shape} output_shape={self.output_shape} dtype={self.dtype}")

    def infer(self, x: np.ndarray) -> np.ndarray:
        assert x.shape == self.input_shape, f"Input shape mismatch: {x.shape} vs {self.input_shape}"
        x = x.astype(self.dtype)
        out = np.empty(self.output_shape, dtype=self.dtype)

        # Use torch cuda tensors as device buffers
        d_input = torch.from_numpy(x).cuda()
        d_output = torch.from_numpy(out).cuda()

        self.context.set_tensor_address(self.input_name, d_input.data_ptr())
        self.context.set_tensor_address(self.output_name, d_output.data_ptr())
        self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()

        return d_output.cpu().numpy().astype(np.float32)


def verify(model_path: str, onnx_path: str, engine_path: str, image_path: str, n_random: int = 20):
    if not os.path.exists(engine_path):
        print(f"[Error] Engine not found: {engine_path}")
        return 1

    print("[Load] Loading PyTorch model on CPU...")
    pt_model = load_pt_model(model_path)
    image_size = pt_model.config.force_image_size or pt_model.config.vision_config.image_size

    print("[Load] Loading ONNX Runtime CPU session...")
    session_cpu = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_input_name = session_cpu.get_inputs()[0].name

    print("[Load] Loading TensorRT engine...")
    trt_runner = TRTVisionRunner(engine_path)

    all_ok = True

    print(f"\n[Random] Running {n_random} random inputs...")
    for i in range(n_random):
        x = torch.randn(1, 3, image_size, image_size, dtype=torch.float32, device="cpu")
        with torch.no_grad():
            pt_out = pt_model.extract_feature(x).cpu().numpy()
        onnx_out = session_cpu.run(None, {onnx_input_name: x.numpy()})[0]
        trt_out = trt_runner.infer(x.numpy())

        ok1 = compare_outputs(pt_out, trt_out, f"TRT-vs-PT-{i+1}")
        ok2 = compare_outputs(onnx_out, trt_out, f"TRT-vs-ONNX-{i+1}")
        if not (ok1 and ok2):
            all_ok = False
            break

    if all_ok and os.path.exists(image_path):
        print(f"\n[Image] Running real image: {image_path}")
        transform = build_transform(image_size)
        img = Image.open(image_path).convert("RGB")
        x = transform(img).unsqueeze(0).to(torch.float32).cpu()
        with torch.no_grad():
            pt_out = pt_model.extract_feature(x).cpu().numpy()
        onnx_out = session_cpu.run(None, {onnx_input_name: x.numpy()})[0]
        trt_out = trt_runner.infer(x.numpy())

        ok1 = compare_outputs(pt_out, trt_out, "TRT-vs-PT-RealImage")
        ok2 = compare_outputs(onnx_out, trt_out, "TRT-vs-ONNX-RealImage")
        if not (ok1 and ok2):
            all_ok = False

    if all_ok:
        print("\n[Result] Vision TensorRT verification PASSED.")
        return 0
    else:
        print("\n[Result] Vision TensorRT verification FAILED.")
        return 1


def main():
    parser = argparse.ArgumentParser(description="Verify InternVL2.5 vision TensorRT engine")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--onnx_path", type=str, default=DEFAULT_ONNX)
    parser.add_argument("--engine_path", type=str, default=DEFAULT_ENGINE)
    parser.add_argument("--image_path", type=str, default=DEFAULT_IMAGE)
    parser.add_argument("--n_random", type=int, default=10)
    args = parser.parse_args()

    sys.exit(verify(args.model_path, args.onnx_path, args.engine_path, args.image_path, args.n_random))


if __name__ == "__main__":
    main()
