#!/usr/bin/env python3
"""
Compare PyTorch SmolVLM2 generate() with TensorRT end-to-end inference.
"""
import argparse
import os
import sys
import time
import warnings

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

DEFAULT_MODEL_PATH = "/home/jovyan/work/vla_smolvlm2_256m/models/HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
DEFAULT_IMAGE = os.path.join(os.path.dirname(__file__), "..", "..", "20260610151154.jpg")


def run_pytorch(model_path: str, image_path: str, prompt: str, max_new_tokens: int, device: torch.device):
    print("[PyTorch] Loading model...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        device_map=None,
    ).to(device)
    model.eval()

    img = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    model_dtype = next(model.parameters()).dtype
    for key in inputs:
        if torch.is_floating_point(inputs[key]):
            inputs[key] = inputs[key].to(model_dtype)

    t0 = time.time()
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    elapsed = time.time() - t0

    input_len = inputs["input_ids"].shape[1]
    generated_ids = generated_ids[:, input_len:]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    print(f"[PyTorch] Inference time: {elapsed:.3f}s")
    return response, elapsed


def run_trt(model_path: str, image_path: str, prompt: str, max_new_tokens: int, model_name: str):
    from run_trt_smolvlm import run_trt_inference
    vision_engine = os.path.join(os.path.dirname(__file__), "..", "outputs", f"{model_name}_vision.trt")
    llm_engine = os.path.join(os.path.dirname(__file__), "..", "outputs", f"{model_name}_llm_embed.trt")
    t0 = time.time()
    response = run_trt_inference(
        model_path=model_path,
        vision_engine_path=vision_engine,
        llm_engine_path=llm_engine,
        image_path=image_path,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
    )
    elapsed = time.time() - t0
    print(f"[TRT] Inference time: {elapsed:.3f}s")
    return response, elapsed


def token_match_rate(text_a: str, text_b: str, tokenizer) -> float:
    toks_a = tokenizer(text_a, add_special_tokens=False)["input_ids"]
    toks_b = tokenizer(text_b, add_special_tokens=False)["input_ids"]
    min_len = min(len(toks_a), len(toks_b))
    if min_len == 0:
        return 0.0
    matches = sum(1 for a, b in zip(toks_a[:min_len], toks_b[:min_len]) if a == b)
    return matches / max(len(toks_a), len(toks_b))


def resolve_model_name(model_path: str) -> str:
    if "256M" in model_path:
        return "smolvlm2_256m"
    if "500M" in model_path:
        return "smolvlm2_500m"
    return "smolvlm2_custom"


def compare(model_path: str, image_path: str, prompt: str, max_new_tokens: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    model_name = resolve_model_name(model_path)

    print("=" * 60)
    print(f"SMOLVLM2 PYTORCH vs TENSORRT COMPARISON ({model_name})")
    print("=" * 60)
    print(f"[Config] Image: {image_path}")
    print(f"[Config] Prompt: {prompt}")
    print(f"[Config] max_new_tokens: {max_new_tokens}")
    print()

    pt_response, pt_time = run_pytorch(model_path, image_path, prompt, max_new_tokens, device)
    print()
    trt_response, trt_time = run_trt(model_path, image_path, prompt, max_new_tokens, model_name)

    match = token_match_rate(pt_response, trt_response, processor.tokenizer)

    print("\n" + "=" * 60)
    print("PYTORCH RESPONSE")
    print("=" * 60)
    print(pt_response)
    print("=" * 60)
    print("\n" + "=" * 60)
    print("TENSORRT RESPONSE")
    print("=" * 60)
    print(trt_response)
    print("=" * 60)

    print("\n[SANITY CHECKS]")
    checks = [
        ("PyTorch response non-empty", len(pt_response.strip()) > 0),
        ("TRT response non-empty", len(trt_response.strip()) > 0),
        ("No <image> leaked in PyTorch", "<image>" not in pt_response),
        ("No <image> leaked in TRT", "<image>" not in trt_response),
        ("No <unk> leaked in PyTorch", "<unk>" not in pt_response),
        ("No <unk> leaked in TRT", "<unk>" not in trt_response),
        ("Token match rate > 50%", match > 0.5),
    ]
    all_ok = True
    for name, ok in checks:
        symbol = "✓" if ok else "✗"
        print(f"  {symbol} {name}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            all_ok = False

    print(f"\n[Metrics]")
    print(f"  Token match rate: {match*100:.1f}%")
    print(f"  PyTorch time:     {pt_time:.3f}s")
    print(f"  TRT time:         {trt_time:.3f}s")
    print(f"  Speedup:          {pt_time/trt_time:.2f}x" if trt_time > 0 else "  Speedup: N/A")

    if all_ok:
        print("\n🎉 Comparison PASSED.")
        return 0
    else:
        print("\n⚠️  Some checks failed.")
        return 1


def main():
    parser = argparse.ArgumentParser(description="Compare PyTorch and TensorRT SmolVLM2 outputs")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--image_path", type=str, default=DEFAULT_IMAGE)
    parser.add_argument("--prompt", type=str, default="Describe this image in detail.")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    args = parser.parse_args()

    sys.exit(compare(args.model_path, args.image_path, args.prompt, args.max_new_tokens))


if __name__ == "__main__":
    main()
