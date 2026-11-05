#!/usr/bin/env python3
"""
Export SmolVLM2 language model to ONNX with inputs_embeds as input.

Inputs:  inputs_embeds   (1, seq_len, hidden_size)
         attention_mask  (1, seq_len)
         position_ids    (1, seq_len)
Outputs: logits          (1, seq_len, vocab_size)

For SmolVLM2-256M: hidden_size=576, vocab_size from text_config
For SmolVLM2-500M: hidden_size=960, vocab_size from text_config
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


def resolve_default_output(model_path: str) -> str:
    if "256M" in model_path:
        name = "smolvlm2_256m_llm_embed.onnx"
    elif "500M" in model_path:
        name = "smolvlm2_500m_llm_embed.onnx"
    else:
        name = "smolvlm2_custom_llm_embed.onnx"
    return os.path.join(os.path.dirname(__file__), "..", "outputs", name)


class LLMEmbedWrapper(nn.Module):
    def __init__(self, text_model, lm_head):
        super().__init__()
        self.text_model = text_model
        self.lm_head = lm_head

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.LongTensor, position_ids: torch.LongTensor):
        out = self.text_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        )
        logits = self.lm_head(out.last_hidden_state)
        return logits


def load_llm(model_path: str, device: torch.device):
    print(f"[Load] Loading SmolVLM2 model to extract text_model + lm_head from {model_path}...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        device_map=None,
        local_files_only=True,
    )
    text_model = model.model.text_model
    lm_head = model.lm_head
    text_model.eval().to(device)
    lm_head.eval().to(device)
    for p in list(text_model.parameters()) + list(lm_head.parameters()):
        p.requires_grad = False
    print(f"[Load] text_model hidden_size={text_model.config.hidden_size}, layers={text_model.config.num_hidden_layers}")
    return text_model, lm_head


def export_llm_embed_onnx(model_path: str, output_path: str, max_seq_len: int = 512, opset: int = 17):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_model, lm_head = load_llm(model_path, device)
    hidden_size = text_model.config.hidden_size
    vocab_size = text_model.config.vocab_size

    wrapper = LLMEmbedWrapper(text_model, lm_head).to(device).eval()

    dummy_seq_len = 16
    dummy_inputs_embeds = torch.randn(1, dummy_seq_len, hidden_size, dtype=torch.float32, device=device)
    dummy_attention_mask = torch.ones((1, dummy_seq_len), dtype=torch.long, device=device)
    dummy_position_ids = torch.arange(dummy_seq_len, dtype=torch.long, device=device).unsqueeze(0)

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
        cpu_embeds = dummy_inputs_embeds.cpu().numpy()
        cpu_mask = dummy_attention_mask.cpu().numpy()
        cpu_pos = dummy_position_ids.cpu().numpy()
        ort_out = session.run(None, {
            "inputs_embeds": cpu_embeds,
            "attention_mask": cpu_mask,
            "position_ids": cpu_pos,
        })[0]

        text_model_cpu = text_model.cpu()
        lm_head_cpu = lm_head.cpu()
        wrapper_cpu = LLMEmbedWrapper(text_model_cpu, lm_head_cpu)
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
    parser = argparse.ArgumentParser(description="Export SmolVLM2 language model to ONNX (embed mode)")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    output_path = args.output_path or resolve_default_output(args.model_path)
    sys.exit(export_llm_embed_onnx(args.model_path, output_path, args.max_seq_len, args.opset))


if __name__ == "__main__":
    main()
