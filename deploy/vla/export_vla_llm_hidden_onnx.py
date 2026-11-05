#!/usr/bin/env python3
"""
Export VLA LLM to ONNX with inputs_embeds input and hidden_states output.

This is different from the generation-oriented internvl2_llm_embed ONNX:
we only need a single forward pass and return the last hidden state,
which is then sliced to the last 24 query positions and fed to the action head.

Inputs:  inputs_embeds (B, seq_len, hidden_size)
         attention_mask (B, seq_len)
         position_ids (B, seq_len)
Outputs: hidden_states (B, seq_len, hidden_size)
"""
import argparse
import os
import sys
import warnings

import torch
import torch.nn as nn
from transformers import AutoModel

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from deploy.vla.utils import load_checkpoint
from vla.lightning import VLAPlModule

DEFAULT_CHECKPOINT = os.path.join(_PROJECT_ROOT, "outputs", "nuscenes_internvl2_5_real_v2", "checkpoints", "last.ckpt")
DEFAULT_OUTPUT = os.path.join(_PROJECT_ROOT, "deploy", "outputs", "vla_llm_hidden.onnx")
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


class LLMHiddenWrapper(nn.Module):
    def __init__(self, llm):
        super().__init__()
        self.llm = llm

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.LongTensor, position_ids: torch.LongTensor):
        out = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        # Return last layer hidden states.
        return out.hidden_states[-1]


def load_llm_from_checkpoint(checkpoint_path: str, device: torch.device):
    print(f"[Load] Loading LLM from VLA checkpoint {checkpoint_path}...")
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
    for p in pl_module.parameters():
        p.requires_grad = False
    llm = pl_module.model.internvl.language_model
    return llm


def load_llm_from_base(model_path: str, device: torch.device):
    print(f"[Load] Loading base InternVL LLM from {model_path}...")
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        device_map=None,
        low_cpu_mem_usage=True,
        local_files_only=True,
        use_flash_attn=False,
    )
    llm = model.language_model
    llm.eval().to(device)
    for p in llm.parameters():
        p.requires_grad = False
    return llm


def export(checkpoint_path: str, output_path: str, base_model_path: str,
           min_seq: int = 64, opt_seq: int = 512, max_seq: int = 2048, opset: int = 17):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if checkpoint_path and os.path.exists(checkpoint_path):
        llm = load_llm_from_checkpoint(checkpoint_path, device)
    else:
        print(f"[Warn] Checkpoint not found, falling back to base model: {base_model_path}")
        llm = load_llm_from_base(base_model_path, device)

    hidden_size = llm.config.hidden_size
    wrapper = LLMHiddenWrapper(llm).to(device).eval()

    dummy_seq_len = 64
    dummy_inputs_embeds = torch.randn(1, dummy_seq_len, hidden_size, dtype=torch.float32, device=device)
    dummy_attention_mask = torch.ones((1, dummy_seq_len), dtype=torch.long, device=device)
    dummy_position_ids = torch.arange(dummy_seq_len, dtype=torch.long, device=device).unsqueeze(0)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"[Export] Exporting LLM (hidden mode) to {output_path}")
    print(f"[Export] Dummy input shape: (1, {dummy_seq_len}, {hidden_size})")

    with torch.no_grad():
        test_out = wrapper(dummy_inputs_embeds, dummy_attention_mask, dummy_position_ids)
        print(f"[Export] Output shape: {tuple(test_out.shape)}")

        torch.onnx.export(
            wrapper,
            (dummy_inputs_embeds, dummy_attention_mask, dummy_position_ids),
            output_path,
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["inputs_embeds", "attention_mask", "position_ids"],
            output_names=["hidden_states"],
            dynamic_axes={
                "inputs_embeds": {0: "batch", 1: "seq_len"},
                "attention_mask": {0: "batch", 1: "seq_len"},
                "position_ids": {0: "batch", 1: "seq_len"},
                "hidden_states": {0: "batch", 1: "seq_len"},
            },
        )

    print(f"[Export] ONNX saved: {output_path}")

    # CPU sanity check with smaller seq_len to avoid huge CPU compute.
    try:
        import numpy as np
        import onnxruntime as ort
        session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
        cpu_embeds = dummy_inputs_embeds.cpu().numpy()
        cpu_mask = dummy_attention_mask.cpu().numpy()
        cpu_pos = dummy_position_ids.cpu().numpy()
        ort_out = session.run(None, {
            "inputs_embeds": cpu_embeds,
            "attention_mask": cpu_mask,
            "position_ids": cpu_pos,
        })[0]
        pt_out = test_out.cpu().numpy()
        diff = np.abs(pt_out - ort_out)
        print(f"[Verify] CPU ONNX vs PyTorch max_abs_error={diff.max():.6e} mean_abs_error={diff.mean():.6e}")
    except Exception as e:
        import traceback
        print(f"[Warn] Could not verify LLM hidden ONNX: {e}")
        traceback.print_exc()

    return 0


def main():
    parser = argparse.ArgumentParser(description="Export VLA LLM (hidden state mode) to ONNX")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--base_model_path", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--min_seq", type=int, default=64)
    parser.add_argument("--opt_seq", type=int, default=512)
    parser.add_argument("--max_seq", type=int, default=2048)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    sys.exit(export(
        args.checkpoint, args.output, args.base_model_path,
        args.min_seq, args.opt_seq, args.max_seq, args.opset
    ))


if __name__ == "__main__":
    main()
