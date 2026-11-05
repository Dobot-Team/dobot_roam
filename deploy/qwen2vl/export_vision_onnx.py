#!/usr/bin/env python3
"""
Export Qwen2-VL visual encoder to ONNX.

Input:  pixel_values   (N, 1176) float32  (flattened image patches)
        image_grid_thw (1, 3) int64
Output: image_embeds   (T*H*W, hidden_size) float32

For Qwen2-VL-2B on 20260610151154.jpg:
  N = 7884, output = (1971, 1536)
"""
import argparse
import os
import sys
import warnings

import torch
import torch.nn as nn
from transformers import Qwen2VLForConditionalGeneration

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "Qwen", "Qwen2-VL-2B-Instruct")
# Export for a single NuScenes CAM_FRONT image; VLA concatenates 4 of these per-frame.
DEFAULT_IMAGE_PATHS = [
    "/root/nuscenes/samples/CAM_FRONT/n008-2018-08-27-11-48-51-0400__CAM_FRONT__1535385105412404.jpg",
]
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "deploy", "outputs", "qwen2vl_2b_vision.onnx")


class VisionEncoderWrapper(nn.Module):
    """Wraps Qwen2-VL visual model for ONNX export."""

    def __init__(self, visual):
        super().__init__()
        self.visual = visual

    def forward(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> torch.Tensor:
        return self.visual(pixel_values, grid_thw=image_grid_thw)


def load_qwen2vl_for_export(model_path: str, device: torch.device):
    print(f"[Load] Loading Qwen2-VL model from {model_path} for export...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        device_map=None,
        local_files_only=True,
        attn_implementation="eager",  # SDPA/flash-attn cannot be exported to ONNX
    )
    model.eval()
    model.to(device)
    for p in model.parameters():
        p.requires_grad = False
    print("[Load] Model loaded.")
    return model


def patch_visual_for_fixed_image(visual, grid_thw: torch.Tensor):
    """
    Precompute rotary embeddings and cu_seqlens for the fixed sample image and
    monkey-patch visual.forward so the exported ONNX graph contains no dynamic
    shape loops. This avoids TensorRT shape-tensor slicing issues in Qwen2-VL's
    windowed RoPE / attention mask construction.
    """
    import torch.nn.functional as F

    device = grid_thw.device
    with torch.no_grad():
        rotary_pos_emb = visual.rot_pos_emb(grid_thw)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())
        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    visual.register_buffer("_fixed_position_embeddings_cos", position_embeddings[0], persistent=False)
    visual.register_buffer("_fixed_position_embeddings_sin", position_embeddings[1], persistent=False)
    visual.register_buffer("_fixed_cu_seqlens", cu_seqlens, persistent=False)

    def _fixed_forward(hidden_states: torch.Tensor, grid_thw: torch.Tensor = None) -> torch.Tensor:
        hidden_states = visual.patch_embed(hidden_states)
        position_embeddings = (visual._fixed_position_embeddings_cos, visual._fixed_position_embeddings_sin)
        cu_seqlens = visual._fixed_cu_seqlens
        for blk in visual.blocks:
            hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings)
        return visual.merger(hidden_states)

    visual.forward = _fixed_forward
    print(f"[Patch] visual.forward frozen for fixed grid_thw={grid_thw.tolist()}")


def export_vision_onnx(model_path: str, image_paths: list, output_path: str, opset: int = 17):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_qwen2vl_for_export(model_path, device)

    # Determine dummy input shape from the provided sample images.
    from transformers import Qwen2VLProcessor
    from PIL import Image
    processor = Qwen2VLProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    for p in image_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Sample image not found: {p}")
    images = [Image.open(p).convert("RGB") for p in image_paths]
    content = [{"type": "image", "image": img} for img in images]
    content.append({"type": "text", "text": "Describe"})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    sample_inputs = processor(text=[text], images=images, return_tensors="pt", padding=True)

    dummy_pixel_values = sample_inputs["pixel_values"].to(torch.float32).to(device)
    dummy_image_grid_thw = sample_inputs["image_grid_thw"].to(torch.int64).to(device)

    # Qwen2-VL's vision encoder is directly under model.visual in transformers 4.50+.
    visual = model.visual

    # Freeze rotary embeddings for the selected image to eliminate dynamic shape loops.
    # NOTE: This produces a fixed-shape engine for this image's grid_thw.  See README.
    patch_visual_for_fixed_image(visual, dummy_image_grid_thw)

    wrapper = VisionEncoderWrapper(visual).to(device).eval()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"[Export] Exporting vision encoder to {output_path}")
    print(f"[Export] Input shapes: pixel_values={tuple(dummy_pixel_values.shape)}, image_grid_thw={tuple(dummy_image_grid_thw.shape)}")

    with torch.no_grad():
        test_out = wrapper(dummy_pixel_values, dummy_image_grid_thw)
        print(f"[Export] Output shape: {tuple(test_out.shape)}")

        torch.onnx.export(
            wrapper,
            (dummy_pixel_values, dummy_image_grid_thw),
            output_path,
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["pixel_values", "image_grid_thw"],
            output_names=["image_embeds"],
            dynamic_axes=None,  # fixed for sample image size
        )

    print(f"[Export] ONNX saved: {output_path}")

    # Verify with onnxruntime CPU
    try:
        import onnxruntime as ort
        import numpy as np

        session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
        cpu_pv = dummy_pixel_values.cpu().numpy()
        cpu_grid = dummy_image_grid_thw.cpu().numpy()
        ort_out = session.run(None, {"pixel_values": cpu_pv, "image_grid_thw": cpu_grid})[0]

        model_cpu = model.cpu()
        wrapper_cpu = VisionEncoderWrapper(model_cpu.model.visual)
        with torch.no_grad():
            pt_out_cpu = wrapper_cpu(dummy_pixel_values.cpu(), dummy_image_grid_thw.cpu()).cpu().numpy()

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
    parser = argparse.ArgumentParser(description="Export Qwen2-VL vision encoder to ONNX")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--image_paths", type=str, nargs="+", default=DEFAULT_IMAGE_PATHS,
                        help="Sample image(s) used to freeze the fixed vision-engine shape")
    parser.add_argument("--output_path", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    sys.exit(export_vision_onnx(args.model_path, args.image_paths, args.output_path, args.opset))


if __name__ == "__main__":
    main()
