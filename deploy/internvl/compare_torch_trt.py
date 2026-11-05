#!/usr/bin/env python3
"""
Compare PyTorch InternVL chat() with TensorRT end-to-end inference.

Reports:
  - Generated text from both
  - Token-level match rate
  - Latency breakdown
  - Sanity checks
"""
import argparse
import os
import sys
import time
import warnings

import torch
from transformers import AutoModel, AutoTokenizer
from PIL import Image
from torchvision import transforms

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

DEFAULT_MODEL_PATH = "/home/jovyan/work/vla_internvl2/models/OpenGVLab/InternVL2_5-1B"
DEFAULT_IMAGE = os.path.join(os.path.dirname(__file__), "..", "..", "20260610151154.jpg")


def build_transform(input_size: int = 448):
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize(input_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def run_pytorch(model_path: str, image_path: str, prompt: str, max_new_tokens: int, device: torch.device):
    print("[PyTorch] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=True, local_files_only=True
    )
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
        device_map=None,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to(device)
    model.eval()

    img = Image.open(image_path).convert("RGB")
    pixel_values = build_transform(448)(img).unsqueeze(0).to(device)
    if model.dtype == torch.bfloat16:
        pixel_values = pixel_values.to(torch.bfloat16)

    generation_config = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "num_beams": 1,
    }

    t0 = time.time()
    response, _ = model.chat(
        tokenizer=tokenizer,
        pixel_values=pixel_values,
        question=prompt,
        generation_config=generation_config,
        history=None,
        return_history=True,
    )
    elapsed = time.time() - t0
    print(f"[PyTorch] Inference time: {elapsed:.3f}s")
    return response, elapsed


def run_trt(model_path: str, image_path: str, prompt: str, max_new_tokens: int):
    # Import runner inline to avoid heavy imports if not needed
    from run_trt_internvl import run_trt_inference
    t0 = time.time()
    response = run_trt_inference(
        model_path=model_path,
        vision_engine_path=os.path.join(os.path.dirname(__file__), "..", "outputs", "internvl2_vision.trt"),
        llm_engine_path=os.path.join(os.path.dirname(__file__), "..", "outputs", "internvl2_llm_embed.trt"),
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


def compare(model_path: str, image_path: str, prompt: str, max_new_tokens: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=True, local_files_only=True
    )

    print("=" * 60)
    print("INTERNVL2.5 PYTORCH vs TENSORRT COMPARISON")
    print("=" * 60)
    print(f"[Config] Image: {image_path}")
    print(f"[Config] Prompt: {prompt}")
    print(f"[Config] max_new_tokens: {max_new_tokens}")
    print()

    pt_response, pt_time = run_pytorch(model_path, image_path, prompt, max_new_tokens, device)
    print()
    trt_response, trt_time = run_trt(model_path, image_path, prompt, max_new_tokens)

    match = token_match_rate(pt_response, trt_response, tokenizer)

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
    parser = argparse.ArgumentParser(description="Compare PyTorch and TensorRT InternVL2.5 outputs")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--image_path", type=str, default=DEFAULT_IMAGE)
    parser.add_argument("--prompt", type=str, default="Describe this image in detail.")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    args = parser.parse_args()

    sys.exit(compare(args.model_path, args.image_path, args.prompt, args.max_new_tokens))


if __name__ == "__main__":
    main()
