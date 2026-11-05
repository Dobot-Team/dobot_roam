#!/usr/bin/env python3
"""
Strict verification of InternVL LLM ONNX against PyTorch.

Checks:
  - Single-step logits for random prompts (CPU reference)
  - argmax next-token consistency
  - Short greedy generation consistency
"""
import argparse
import os
import sys
import warnings

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModel, AutoTokenizer

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

DEFAULT_MODEL_PATH = "/home/jovyan/work/vla_internvl2/models/OpenGVLab/InternVL2_5-1B"
DEFAULT_ONNX = os.path.join(os.path.dirname(__file__), "..", "outputs", "internvl2_llm.onnx")


def load_llm_tokenizer(model_path: str):
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        device_map=None,
        low_cpu_mem_usage=True,
        local_files_only=True,
        use_flash_attn=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True,
        local_files_only=True,
    )
    llm = model.language_model
    llm.eval().cpu()
    return llm, tokenizer


def compare_logits(pt_logits: np.ndarray, ort_logits: np.ndarray, tag: str):
    diff = np.abs(pt_logits - ort_logits)
    max_err = float(diff.max())
    mean_err = float(diff.mean())
    rel_err = max_err / (np.abs(pt_logits).max() + 1e-12)
    pt_argmax = pt_logits.argmax(-1)
    ort_argmax = ort_logits.argmax(-1)
    token_match = float((pt_argmax == ort_argmax).mean())

    print(f"[{tag}] max_abs_error={max_err:.6e} mean_abs_error={mean_err:.6e} rel_error={rel_err:.6e} token_match={token_match:.4f}")

    ok = True
    if max_err >= 1e-3:
        print(f"[{tag}] FAIL: max_abs_error >= 1e-3")
        ok = False
    if mean_err >= 1e-4:
        print(f"[{tag}] FAIL: mean_abs_error >= 1e-4")
        ok = False
    if token_match < 1.0:
        print(f"[{tag}] FAIL: token_match < 100%")
        ok = False
    return ok


def greedy_generate_pt(llm, input_ids: torch.Tensor, max_new_tokens: int):
    """Greedy generation WITHOUT KV-cache (to match ONNX runner)."""
    generated = input_ids.clone()
    for _ in range(max_new_tokens):
        attention_mask = torch.ones_like(generated)
        position_ids = torch.arange(generated.shape[1], dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            logits = llm(generated, attention_mask=attention_mask, position_ids=position_ids, use_cache=False).logits
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=-1)
    return generated[0].tolist()


def greedy_generate_ort(session, input_ids: np.ndarray, max_new_tokens: int):
    generated = input_ids.copy()
    for _ in range(max_new_tokens):
        attention_mask = np.ones_like(generated)
        position_ids = np.arange(generated.shape[1], dtype=np.int64).reshape(1, -1)
        logits = session.run(None, {
            "input_ids": generated,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        })[0]
        next_token = logits[:, -1, :].argmax(axis=-1, keepdims=True)
        generated = np.concatenate([generated, next_token], axis=-1)
    return generated[0].tolist()


def verify(model_path: str, onnx_path: str, n_prompts: int = 10, gen_len: int = 10):
    if not os.path.exists(onnx_path):
        print(f"[Error] ONNX not found: {onnx_path}")
        return 1

    print("[Load] Loading PyTorch LLM and tokenizer...")
    llm, tokenizer = load_llm_tokenizer(model_path)
    vocab_size = llm.config.vocab_size

    print("[Load] Loading ONNX Runtime CPU session...")
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    all_ok = True

    # 1. Random prompt logits comparison
    print(f"\n[Random] Running {n_prompts} random prompts...")
    for i in range(n_prompts):
        seq_len = np.random.randint(5, 32)
        input_ids = torch.randint(0, vocab_size, (1, seq_len), dtype=torch.long)
        attention_mask = torch.ones((1, seq_len), dtype=torch.long)
        position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

        with torch.no_grad():
            pt_logits = llm(input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False).logits.numpy()
        ort_logits = session.run(None, {
            "input_ids": input_ids.numpy(),
            "attention_mask": attention_mask.numpy(),
            "position_ids": position_ids.numpy(),
        })[0]

        ok = compare_logits(pt_logits, ort_logits, f"Prompt-{i+1}")
        if not ok:
            all_ok = False
            break

    # 2. Greedy generation consistency
    if all_ok:
        print(f"\n[Generation] Greedy generating {gen_len} tokens from a real prompt...")
        prompt_text = "The image shows"
        input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids
        pt_tokens = greedy_generate_pt(llm, input_ids, gen_len)
        ort_tokens = greedy_generate_ort(session, input_ids.numpy(), gen_len)

        print(f"[Generation] PyTorch tokens: {pt_tokens}")
        print(f"[Generation] ONNX tokens:    {ort_tokens}")
        if pt_tokens != ort_tokens:
            print("[Generation] FAIL: token sequences differ")
            all_ok = False
        else:
            print("[Generation] PASS: token sequences match")

    if all_ok:
        print("\n[Result] LLM ONNX verification PASSED.")
        return 0
    else:
        print("\n[Result] LLM ONNX verification FAILED.")
        return 1


def main():
    parser = argparse.ArgumentParser(description="Verify InternVL2.5 LLM ONNX against PyTorch")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--onnx_path", type=str, default=DEFAULT_ONNX)
    parser.add_argument("--n_prompts", type=int, default=10)
    parser.add_argument("--gen_len", type=int, default=10)
    args = parser.parse_args()

    sys.exit(verify(args.model_path, args.onnx_path, args.n_prompts, args.gen_len))


if __name__ == "__main__":
    main()
