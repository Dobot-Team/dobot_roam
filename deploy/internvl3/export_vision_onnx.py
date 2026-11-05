#!/usr/bin/env python3
"""
Export InternVL3-1B vision encoder (ViT + pixel_shuffle + mlp1) to ONNX.

Input:  pixel_values (1, 3, 448, 448) float32
Output: visual_features (1, num_image_token, llm_hidden_size) float32

For InternVL3-1B: output shape is (1, 256, 896).
"""
import argparse
import os
import sys
import warnings

import torch
import torch.nn as nn
from transformers import AutoModel

# Suppress common noisy warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


DEFAULT_MODEL_PATH = "/home/jovyan/work/vla_internvl3/models/OpenGVLab/InternVL3-1B"
DEFAULT_OUTPUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "internvl3_vision.onnx")
DEFAULT_DYNAMIC_OUTPUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "internvl3_vision_dynamic.onnx")


class VisionEncoderWrapper(nn.Module):
    """Wraps InternVLChatModel.extract_feature for ONNX export."""

    def __init__(self, internvl_model):
        super().__init__()
        self.model = internvl_model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model.extract_feature(pixel_values)


def load_internvl_for_export(model_path: str, device: torch.device):
    print(f"[Load] Loading InternVL model from {model_path} for export...")
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        device_map=None,
        low_cpu_mem_usage=True,
        local_files_only=True,
        use_flash_attn=False,  # ONNX/TRT does not support flash_attn custom ops
    )
    model.eval()
    model.to(device)
    # Disable gradient for all parameters
    for p in model.parameters():
        p.requires_grad = False
    print(f"[Load] Model loaded. num_image_token={model.num_image_token}")
    return model


def export_vision_onnx(model_path: str, output_path: str, opset: int = 17,
                       batch_size: int = 1, dynamic_batch: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_internvl_for_export(model_path, device)

    # Infer input size from config
    image_size = model.config.force_image_size or model.config.vision_config.image_size
    dummy_input = torch.randn(batch_size, 3, image_size, image_size, dtype=torch.float32, device=device)

    wrapper = VisionEncoderWrapper(model).to(device).eval()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "pixel_values": {0: "batch"},
            "visual_features": {0: "batch"},
        }
        print(f"[Export] Exporting vision encoder with DYNAMIC batch to {output_path}")
    else:
        print(f"[Export] Exporting vision encoder with STATIC batch={batch_size} to {output_path}")
    print(f"[Export] Dummy input shape:  {tuple(dummy_input.shape)}")

    with torch.no_grad():
        # Test forward before export
        test_out = wrapper(dummy_input)
        print(f"[Export] Output shape: {tuple(test_out.shape)}")

        torch.onnx.export(
            wrapper,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["pixel_values"],
            output_names=["visual_features"],
            dynamic_axes=dynamic_axes,
        )

    print(f"[Export] ONNX saved: {output_path}")

    # Verify with onnxruntime if available
    try:
        import onnxruntime as ort
        import numpy as np

        # Use CPU for strict graph-correctness check (GPU kernels can introduce small numeric differences)
        session_cpu = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
        cpu_input = dummy_input.detach().cpu().numpy()
        ort_out_cpu = session_cpu.run(None, {session_cpu.get_inputs()[0].name: cpu_input})[0]
        model_cpu = model.cpu()
        with torch.no_grad():
            pt_out_cpu = model_cpu.extract_feature(dummy_input.cpu()).cpu().numpy()
        diff_cpu = np.abs(pt_out_cpu - ort_out_cpu)
        print(f"[Verify] CPU ONNX vs CPU PyTorch max_abs_error={diff_cpu.max():.6e} mean_abs_error={diff_cpu.mean():.6e}")

        # Also report GPU comparison for information
        if torch.cuda.is_available():
            session_gpu = ort.InferenceSession(output_path, providers=["CUDAExecutionProvider"])
            ort_out_gpu = session_gpu.run(None, {session_gpu.get_inputs()[0].name: dummy_input.cpu().numpy()})[0]
            diff_gpu = np.abs(test_out.cpu().numpy() - ort_out_gpu)
            print(f"[Verify] GPU ONNX vs GPU PyTorch max_abs_error={diff_gpu.max():.6e} mean_abs_error={diff_gpu.mean():.6e} (info only)")

        max_err = diff_cpu.max()
        mean_err = diff_cpu.mean()
        rel_err = max_err / (np.abs(pt_out_cpu).max() + 1e-12)
        if max_err > 1e-2 or mean_err > 1e-4 or rel_err > 1e-3:
            print(f"[Warn] CPU ONNX output differs from CPU PyTorch beyond tolerance (max={max_err:.6e}, mean={mean_err:.6e}, rel={rel_err:.6e}).")
            return 1
        print("[Verify] ONNX export PASSED (CPU reference).")
    except Exception as e:
        import traceback
        print(f"[Warn] Could not verify with onnxruntime: {e}")
        traceback.print_exc()

    return 0


def main():
    parser = argparse.ArgumentParser(description="Export InternVL3 vision encoder to ONNX")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="Local InternVL model path")
    parser.add_argument("--output", type=str, default=None, help="Output ONNX path")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--batch_size", type=int, default=1, help="Static batch size (ignored if --dynamic_batch)")
    parser.add_argument("--dynamic_batch", action="store_true", help="Export with dynamic batch dimension")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"[Error] Model path not found: {args.model_path}")
        sys.exit(1)

    if args.output is None:
        args.output = DEFAULT_DYNAMIC_OUTPUT if args.dynamic_batch else DEFAULT_OUTPUT

    sys.exit(export_vision_onnx(args.model_path, args.output, args.opset, args.batch_size, args.dynamic_batch))


if __name__ == "__main__":
    main()
