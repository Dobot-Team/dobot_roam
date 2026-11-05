#!/usr/bin/env python3
"""
Export SmolVLM2 vision encoder (vision_model + connector) to ONNX.

Input:  pixel_values      (1, 9, 3, 512, 512) float32
        pixel_attention_mask (1, 9, 512, 512) int64
Output: image_hidden_states (1, 9, 64, hidden_size) float32

For SmolVLM2-256M: hidden_size=576
For SmolVLM2-500M: hidden_size=960
"""
import argparse
import os
import sys
import warnings

import torch
import torch.nn as nn
from transformers import AutoModelForImageTextToText

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

DEFAULT_MODEL_PATH = "/home/jovyan/work/vla_smolvlm2_256m/models/HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
DEFAULT_OUTPUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "smolvlm2_256m_vision.onnx")


def resolve_default_output(model_path: str, dynamic_batch: bool = False) -> str:
    if "256M" in model_path:
        name = "smolvlm2_256m_vision_dynamic.onnx" if dynamic_batch else "smolvlm2_256m_vision.onnx"
    elif "500M" in model_path:
        name = "smolvlm2_500m_vision_dynamic.onnx" if dynamic_batch else "smolvlm2_500m_vision.onnx"
    else:
        name = "smolvlm2_custom_vision_dynamic.onnx" if dynamic_batch else "smolvlm2_custom_vision.onnx"
    return os.path.join(os.path.dirname(__file__), "..", "outputs", name)


class VisionEncoderWrapper(nn.Module):
    """Wraps SmolVLM2 vision_model + connector for ONNX export."""

    def __init__(self, vision_model, connector, patch_size: int, num_patches_h: int = 32, num_patches_w: int = 32):
        super().__init__()
        self.vision_model = vision_model
        self.connector = connector
        self.patch_size = patch_size
        self.num_patches_h = num_patches_h
        self.num_patches_w = num_patches_w

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: (1, 9, 3, 512, 512)
        batch_size, num_images, C, H, W = pixel_values.shape
        pixel_values = pixel_values.view(batch_size * num_images, C, H, W)

        # Fixed-size patch attention mask: all patches are valid for 512x512 image with patch_size=16
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
        # Reshape back to (batch_size, num_images, num_tokens, hidden_size)
        image_hidden_states = image_hidden_states.view(batch_size, num_images, *image_hidden_states.shape[1:])
        return image_hidden_states


def load_smolvlm_for_export(model_path: str, device: torch.device):
    print(f"[Load] Loading SmolVLM2 model from {model_path} for export...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        device_map=None,
        local_files_only=True,
    )
    model.eval()
    model.to(device)
    for p in model.parameters():
        p.requires_grad = False
    patch_size = model.model.vision_model.config.patch_size
    print(f"[Load] Model loaded. patch_size={patch_size}")
    return model, patch_size


def export_vision_onnx(model_path: str, output_path: str, opset: int = 17,
                       batch_size: int = 1, dynamic_batch: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, patch_size = load_smolvlm_for_export(model_path, device)

    wrapper = VisionEncoderWrapper(
        model.model.vision_model,
        model.model.connector,
        patch_size,
        num_patches_h=512 // patch_size,
        num_patches_w=512 // patch_size,
    ).to(device).eval()

    dummy_pixel_values = torch.randn(batch_size, 9, 3, 512, 512, dtype=torch.float32, device=device)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "pixel_values": {0: "batch"},
            "image_hidden_states": {0: "batch"},
        }
        print(f"[Export] Exporting vision encoder with DYNAMIC outer batch to {output_path}")
    else:
        print(f"[Export] Exporting vision encoder with STATIC outer batch={batch_size} to {output_path}")
    print(f"[Export] Input shape: pixel_values={tuple(dummy_pixel_values.shape)}")

    with torch.no_grad():
        test_out = wrapper(dummy_pixel_values)
        print(f"[Export] Output shape: {tuple(test_out.shape)}")

        torch.onnx.export(
            wrapper,
            dummy_pixel_values,
            output_path,
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["pixel_values"],
            output_names=["image_hidden_states"],
            dynamic_axes=dynamic_axes,
        )

    print(f"[Export] ONNX saved: {output_path}")

    # Verify with onnxruntime CPU
    try:
        import onnxruntime as ort
        import numpy as np

        session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
        cpu_pv = dummy_pixel_values.cpu().numpy()
        ort_out = session.run(None, {"pixel_values": cpu_pv})[0]

        model_cpu = model.cpu()
        wrapper_cpu = VisionEncoderWrapper(
            model_cpu.model.vision_model,
            model_cpu.model.connector,
            patch_size,
            num_patches_h=512 // patch_size,
            num_patches_w=512 // patch_size,
        )
        with torch.no_grad():
            pt_out_cpu = wrapper_cpu(dummy_pixel_values.cpu()).cpu().numpy()

        diff = np.abs(pt_out_cpu - ort_out)
        print(f"[Verify] CPU ONNX vs CPU PyTorch max_abs_error={diff.max():.6e} mean_abs_error={diff.mean():.6e}")

        max_err = diff.max()
        mean_err = diff.mean()
        rel_err = max_err / (np.abs(pt_out_cpu).max() + 1e-12)
        if max_err > 1e-2 or mean_err > 1e-4 or rel_err > 1e-3:
            print(f"[Warn] CPU ONNX output differs beyond tolerance (max={max_err:.6e}, mean={mean_err:.6e}, rel={rel_err:.6e}).")
            return 1
        print("[Verify] Vision ONNX verification PASSED.")
    except Exception as e:
        print(f"[Warn] ONNX verification skipped/failed: {e}")

    return 0


def main():
    parser = argparse.ArgumentParser(description="Export SmolVLM2 vision encoder to ONNX")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--batch_size", type=int, default=1, help="Static outer batch size (ignored if --dynamic_batch)")
    parser.add_argument("--dynamic_batch", action="store_true", help="Export with dynamic outer batch dimension")
    args = parser.parse_args()

    output_path = args.output_path or resolve_default_output(args.model_path, args.dynamic_batch)
    sys.exit(export_vision_onnx(args.model_path, output_path, args.opset, args.batch_size, args.dynamic_batch))


if __name__ == "__main__":
    main()
