#!/usr/bin/env python3
"""
Export Qwen2-VL language model to ONNX with inputs_embeds as input.

Inputs:  inputs_embeds   (1, seq_len, hidden_size)
         attention_mask  (1, seq_len)
         position_ids    (3, 1, seq_len)   # mrope 3-D position ids
Outputs: logits          (1, seq_len, vocab_size)
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
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "deploy", "outputs", "qwen2vl_2b_llm_embed.onnx")


class LLMEmbedWrapper(nn.Module):
    def __init__(self, qwen_model, lm_head):
        super().__init__()
        self.qwen_model = qwen_model
        self.lm_head = lm_head

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.LongTensor, position_ids: torch.LongTensor):
        out = self.qwen_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        )
        logits = self.lm_head(out.last_hidden_state)
        return logits


def load_llm(model_path: str, device: torch.device):
    print(f"[Load] Loading Qwen2-VL model to extract LLM from {model_path}...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        device_map=None,
        local_files_only=True,
        attn_implementation="eager",  # SDPA/flash-attn cannot be exported to ONNX
    )
    qwen_model = model.model
    lm_head = model.lm_head
    qwen_model.eval().to(device)
    lm_head.eval().to(device)
    for p in list(qwen_model.parameters()) + list(lm_head.parameters()):
        p.requires_grad = False
    # Use the top-level config (Qwen2VLConfig) for hidden_size / num_hidden_layers.
    print(f"[Load] LLM hidden_size={model.config.text_config.hidden_size}, layers={model.config.text_config.num_hidden_layers}")
    return qwen_model, lm_head


def export_llm_embed_onnx(model_path: str, output_path: str, opset: int = 17):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    qwen_model, lm_head = load_llm(model_path, device)
    # qwen_model.config is the top-level Qwen2VLConfig; hidden_size lives in text_config.
    hidden_size = qwen_model.config.text_config.hidden_size
    wrapper = LLMEmbedWrapper(qwen_model, lm_head).to(device).eval()

    dummy_seq_len = 16
    dummy_inputs_embeds = torch.randn(1, dummy_seq_len, hidden_size, dtype=torch.float32, device=device)
    dummy_attention_mask = torch.ones((1, dummy_seq_len), dtype=torch.long, device=device)
    # mrope position_ids: (3, batch, seq_len)
    dummy_position_ids = torch.arange(dummy_seq_len, dtype=torch.long, device=device).unsqueeze(0).unsqueeze(0).expand(3, 1, -1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"[Export] Exporting LLM (embed mode) to {output_path}")
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
            output_names=["logits"],
            dynamic_axes={
                "inputs_embeds": {0: "batch", 1: "seq_len"},
                "attention_mask": {0: "batch", 1: "seq_len"},
                "position_ids": {1: "batch", 2: "seq_len"},
                "logits": {0: "batch", 1: "seq_len"},
            },
        )

    print(f"[Export] ONNX saved: {output_path}")

    # Verify with onnxruntime CPU
    try:
        import onnxruntime as ort
        import numpy as np

        session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
        cpu_embeds = dummy_inputs_embeds.cpu().numpy()
        cpu_mask = dummy_attention_mask.cpu().numpy()
        cpu_pos = dummy_position_ids.cpu().numpy()
        ort_out = session.run(None, {
            "inputs_embeds": cpu_embeds,
            "attention_mask": cpu_mask,
            "position_ids": cpu_pos,
        })[0]

        qwen_model_cpu = qwen_model.cpu()
        lm_head_cpu = lm_head.cpu()
        wrapper_cpu = LLMEmbedWrapper(qwen_model_cpu, lm_head_cpu)
        with torch.no_grad():
            pt_out_cpu = wrapper_cpu(dummy_inputs_embeds.cpu(), dummy_attention_mask.cpu(), dummy_position_ids.cpu()).cpu().numpy()

        diff = np.abs(pt_out_cpu - ort_out)
        print(f"[Verify] CPU ONNX vs CPU PyTorch max_abs_error={diff.max():.6e} mean_abs_error={diff.mean():.6e}")

        max_err = diff.max()
        mean_err = diff.mean()
        rel_err = max_err / (np.abs(pt_out_cpu).max() + 1e-12)
        if max_err > 1e-2 or mean_err > 1e-4 or rel_err > 1e-3:
            print(f"[Warn] CPU ONNX output differs beyond tolerance (max={max_err:.6e}, mean={mean_err:.6e}, rel={rel_err:.6e}).")
            return 1
        print("[Verify] LLM ONNX verification PASSED.")
    except Exception as e:
        print(f"[Warn] ONNX verification skipped/failed: {e}")

    return 0


def main():
    parser = argparse.ArgumentParser(description="Export Qwen2-VL language model to ONNX (embed mode)")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output_path", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    sys.exit(export_llm_embed_onnx(args.model_path, args.output_path, args.opset))


if __name__ == "__main__":
    main()
