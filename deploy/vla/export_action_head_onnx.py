#!/usr/bin/env python3
"""
Export VLA action head (MLP) to ONNX.

Input:  query_hidden (B, num_queries, hidden_size)
Output: pred_traj    (B, num_queries, 2)
"""
import argparse
import os
import sys
import warnings

import torch
import torch.nn as nn

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Allow importing vla and deploy packages from project root.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from deploy.vla.utils import load_checkpoint
from vla.models import InternVLVLA

DEFAULT_CHECKPOINT = os.path.join(_PROJECT_ROOT, "outputs", "nuscenes_internvl2_5_real_v2", "checkpoints", "last.ckpt")
DEFAULT_OUTPUT = os.path.join(_PROJECT_ROOT, "deploy", "outputs", "vla_action_head.onnx")
DEFAULT_BASE_MODEL = "/root/embodvla/models/OpenGVLab/InternVL2_5-1B"


def _get_model_path_from_checkpoint(checkpoint_path: str, fallback: str = DEFAULT_BASE_MODEL) -> str:
    """Read the saved model_path from the Lightning checkpoint hyper-parameters."""
    try:
        ckpt = load_checkpoint(checkpoint_path, map_location="cpu")
        hp = ckpt.get("hyper_parameters", {})
        model_path = hp.get("model_path")
        if model_path and os.path.exists(model_path):
            print(f"[Load] Detected backbone model_path from checkpoint: {model_path}")
            return model_path
    except Exception as e:
        print(f"[Warn] Could not read model_path from checkpoint: {e}")
    print(f"[Load] Using fallback base model: {fallback}")
    return fallback


class ActionHeadWrapper(nn.Module):
    def __init__(self, action_head):
        super().__init__()
        self.action_head = action_head

    def forward(self, query_hidden: torch.Tensor) -> torch.Tensor:
        return self.action_head(query_hidden)


def export(checkpoint_path: str, output_path: str, opset: int = 17):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[Load] Loading VLA model from checkpoint {checkpoint_path}...")
    # Load full model to extract action head.  In practice only action_head + query weights matter.
    from vla.lightning import VLAPlModule
    model_path = _get_model_path_from_checkpoint(checkpoint_path)
    pl_module = VLAPlModule.load_from_checkpoint(
        checkpoint_path,
        model_path=model_path,
        num_queries=24,
        freeze_vision=True,
        freeze_llm=False,
        freeze_mlp1=True,
        action_head_dims=(512, 2),
        lr=1e-4,
        weight_decay=0.01,
        warmup_steps=100,
        max_steps=5000,
        dtype=torch.float32,
    )
    pl_module.eval().to(device)
    model = pl_module.model

    hidden_size = model.llm_hidden_size
    num_queries = model.num_queries

    wrapper = ActionHeadWrapper(model.action_head).to(device).eval()
    for p in wrapper.parameters():
        p.requires_grad = False

    dummy_input = torch.randn(1, num_queries, hidden_size, dtype=torch.float32, device=device)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"[Export] Exporting action head to {output_path}")
    print(f"[Export] Input shape: (1, {num_queries}, {hidden_size})")

    with torch.no_grad():
        test_out = wrapper(dummy_input)
        print(f"[Export] Output shape: {tuple(test_out.shape)}")

        torch.onnx.export(
            wrapper,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["query_hidden"],
            output_names=["pred_traj"],
            dynamic_axes={
                "query_hidden": {0: "batch"},
                "pred_traj": {0: "batch"},
            },
        )

    print(f"[Export] ONNX saved: {output_path}")

    # Quick CPU verification.
    try:
        import numpy as np
        import onnxruntime as ort
        session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
        np_input = dummy_input.cpu().numpy()
        ort_out = session.run(None, {"query_hidden": np_input})[0]
        pt_out = test_out.cpu().numpy()
        diff = np.abs(pt_out - ort_out)
        print(f"[Verify] CPU ONNX vs PyTorch max_abs_error={diff.max():.6e} mean_abs_error={diff.mean():.6e}")
    except Exception as e:
        print(f"[Warn] Could not verify action head ONNX: {e}")

    return 0


def main():
    parser = argparse.ArgumentParser(description="Export VLA action head to ONNX")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    sys.exit(export(args.checkpoint, args.output, args.opset))


if __name__ == "__main__":
    main()
