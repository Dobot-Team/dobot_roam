#!/usr/bin/env python3
"""
Export InternVL2.5-1B language model (Qwen2.5-0.5B) to ONNX.

We export a single forward pass WITHOUT KV-cache to keep the graph simple:
  Inputs:  input_ids (1, seq_len), attention_mask (1, seq_len), position_ids (1, seq_len)
  Outputs: logits (1, seq_len, vocab_size)

The generation loop in the TRT runner will feed the full prefix each step.
"""
import argparse
import json
import os
import sys
import warnings

import torch
import torch.nn as nn
from transformers import AutoModel

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

DEFAULT_MODEL_PATH = "/home/jovyan/work/vla_internvl2/models/OpenGVLab/InternVL2_5-1B"
DEFAULT_OUTPUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "internvl2_llm.onnx")


class LLMWrapper(nn.Module):
    def __init__(self, llm):
        super().__init__()
        self.llm = llm

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor, position_ids: torch.LongTensor):
        out = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        )
        return out.logits


def load_llm(model_path: str, device: torch.device):
    print(f"[Load] Loading InternVL model to extract LLM from {model_path}...")
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
    print(f"[Load] LLM extracted: {type(llm).__name__}")
    print(f"[Load] vocab_size={llm.config.vocab_size}, hidden_size={llm.config.hidden_size}, layers={llm.config.num_hidden_layers}")
    return llm


def export_llm_onnx(model_path: str, output_path: str, max_seq_len: int = 512, opset: int = 17):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    llm = load_llm(model_path, device)
    vocab_size = llm.config.vocab_size

    wrapper = LLMWrapper(llm).to(device).eval()

    # Dummy inputs at export time: batch=1, seq_len=1 (min), max_seq_len (opt/max)
    # TensorRT dynamic profile will be set from optimization profile at engine build time.
    dummy_seq_len = 16
    dummy_input_ids = torch.randint(0, vocab_size, (1, dummy_seq_len), dtype=torch.long, device=device)
    dummy_attention_mask = torch.ones((1, dummy_seq_len), dtype=torch.long, device=device)
    dummy_position_ids = torch.arange(dummy_seq_len, dtype=torch.long, device=device).unsqueeze(0)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"[Export] Exporting LLM to {output_path}")
    print(f"[Export] Dummy input shape: (1, {dummy_seq_len})")

    with torch.no_grad():
        test_out = wrapper(dummy_input_ids, dummy_attention_mask, dummy_position_ids)
        print(f"[Export] Output shape: {tuple(test_out.shape)}")

        torch.onnx.export(
            wrapper,
            (dummy_input_ids, dummy_attention_mask, dummy_position_ids),
            output_path,
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["input_ids", "attention_mask", "position_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq_len"},
                "attention_mask": {0: "batch", 1: "seq_len"},
                "position_ids": {0: "batch", 1: "seq_len"},
                "logits": {0: "batch", 1: "seq_len"},
            },
        )

    print(f"[Export] ONNX saved: {output_path}")

    # Verify with onnxruntime CPU
    try:
        import onnxruntime as ort
        import numpy as np
        session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
        cpu_ids = dummy_input_ids.cpu().numpy()
        cpu_mask = dummy_attention_mask.cpu().numpy()
        cpu_pos = dummy_position_ids.cpu().numpy()
        ort_out = session.run(None, {
            "input_ids": cpu_ids,
            "attention_mask": cpu_mask,
            "position_ids": cpu_pos,
        })[0]
        llm_cpu = llm.cpu()
        wrapper_cpu = LLMWrapper(llm_cpu)
        with torch.no_grad():
            pt_out = wrapper_cpu(
                torch.from_numpy(cpu_ids),
                torch.from_numpy(cpu_mask),
                torch.from_numpy(cpu_pos),
            ).numpy()
        diff = np.abs(pt_out - ort_out)
        max_err = diff.max()
        mean_err = diff.mean()
        print(f"[Verify] CPU ONNX vs CPU PyTorch max_abs_error={max_err:.6e} mean_abs_error={mean_err:.6e}")
        # Logits can differ slightly due to fused layernorm/softmax; token-level decisions are robust to ~1e-3 differences.
        if max_err > 1e-3 or mean_err > 1e-4:
            print("[Warn] CPU LLM ONNX differs from PyTorch beyond tolerance.")
            return 1
        print("[Verify] LLM ONNX export PASSED (CPU reference).")
    except Exception as e:
        import traceback
        print(f"[Warn] Could not verify LLM ONNX: {e}")
        traceback.print_exc()

    return 0


def main():
    parser = argparse.ArgumentParser(description="Export InternVL2.5 LLM to ONNX")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    sys.exit(export_llm_onnx(args.model_path, args.output, args.max_seq_len, args.opset))


if __name__ == "__main__":
    main()
