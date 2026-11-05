#!/usr/bin/env python3
"""
Compare PyTorch VLA inference with TensorRT VLA inference.

Reports:
  - Predicted trajectories from both
  - Point-wise and trajectory-level error metrics
  - Latency breakdown
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np
import torch
from PIL import Image

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from deploy.vla.utils import load_checkpoint
from vla.data import NuScenesVLADataModule
from vla.lightning import VLAPlModule
from run_trt_vla import VLATRTInference, build_transform

DEFAULT_CHECKPOINT = os.path.join(_PROJECT_ROOT, "outputs", "nuscenes_internvl2_5_real_v2", "checkpoints", "last.ckpt")
DEFAULT_BASE_MODEL = "/root/embodvla/models/OpenGVLab/InternVL2_5-1B"
DEFAULT_VISION_ENGINE = os.path.join(_PROJECT_ROOT, "deploy", "outputs", "internvl2_vision_dynamic.trt")
DEFAULT_LLM_ENGINE = os.path.join(_PROJECT_ROOT, "deploy", "outputs", "vla_llm_hidden.trt")
DEFAULT_ACTION_ENGINE = os.path.join(_PROJECT_ROOT, "deploy", "outputs", "vla_action_head.trt")


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

IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"


def build_prompt(tokenizer, num_images: int, prompt: str, num_image_token: int):
    image_token_str = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * num_image_token + IMG_END_TOKEN
    image_tokens = "\n".join([image_token_str] * num_images)
    text = f"{image_tokens}\nUser: {prompt}\nAssistant:"
    model_inputs = tokenizer(text, return_tensors="pt", padding=False, truncation=True, max_length=2048)
    return model_inputs["input_ids"], model_inputs["attention_mask"]


def run_pytorch(pl_module: VLAPlModule, image_paths: list, gt: np.ndarray, device: torch.device):
    model = pl_module.model
    model.eval().to(device)

    # Preprocess images.
    transform = build_transform(448)
    images = [Image.open(p).convert("RGB") for p in image_paths]
    pixel_values = torch.stack([transform(img) for img in images], dim=0).to(device)

    # Build natural (unpadded) prompt.
    input_ids, attention_mask = build_prompt(
        model.tokenizer, len(images),
        "Predict the future trajectory of the ego vehicle.",
        model.internvl.num_image_token,
    )
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    t0 = time.time()
    with torch.no_grad():
        if model.dtype == torch.bfloat16 and device.type != "cpu":
            pixel_values = pixel_values.to(torch.bfloat16)
            with torch.amp.autocast(device.type, dtype=torch.bfloat16):
                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    trajectories=None,
                )
        else:
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                trajectories=None,
            )
    elapsed = time.time() - t0
    pred = outputs["pred_traj"].float().cpu().numpy()
    print(f"[PyTorch] Inference time: {elapsed:.3f}s")
    return pred, gt, elapsed


def compute_metrics(pred: np.ndarray, gt: np.ndarray):
    dists = np.linalg.norm(pred - gt, axis=-1)
    return {
        "ADE": float(dists.mean()),
        "FDE": float(dists[:, -1].mean()),
        "max_error": float(dists.max()),
    }


def compare(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("VLA PYTORCH vs TENSORRT COMPARISON")
    print("=" * 60)
    print(f"[Config] Checkpoint: {args.checkpoint}")
    print(f"[Config] PyTorch dtype: {'bfloat16' if args.bf16 else 'float32'}")
    print(f"[Config] TRT-path LLM backend: {args.llm_backend}")
    print()

    dtype = torch.bfloat16 if args.bf16 else torch.float32
    resolved_base_model_path = _get_model_path_from_checkpoint(args.checkpoint, args.base_model_path)
    pl_module = VLAPlModule.load_from_checkpoint(
        args.checkpoint,
        model_path=resolved_base_model_path,
        num_queries=24,
        freeze_vision=True,
        freeze_llm=False,
        freeze_mlp1=True,
        action_head_dims=(512, 2),
        lr=1e-4,
        weight_decay=0.01,
        warmup_steps=100,
        max_steps=5000,
        dtype=dtype,
    )

    # Use data module only to retrieve sample image paths and ground truth.
    dm = NuScenesVLADataModule(
        dataroot=args.dataroot,
        tokenizer=pl_module.model.tokenizer,
        num_history_frames=4,
        n_future=24,
        batch_size=1,
        num_workers=0,
        num_image_tokens=256,
        max_length=2024,
    )
    dm.setup()
    sample_idx = 0
    image_paths = dm.val_dataset.samples[sample_idx]["image_paths"]
    gt = dm.val_dataset.samples[sample_idx]["traj"][np.newaxis, ...]  # (1, 24, 2)

    pt_pred, gt, pt_time = run_pytorch(pl_module, image_paths, gt, device)
    print()

    # Load TRT runner once; inference time is measured separately.
    print("[TRT] Loading deployment runner...")
    trt_runner = VLATRTInference(
        args.checkpoint, resolved_base_model_path,
        args.vision_engine, args.llm_engine, args.action_engine,
        llm_backend=args.llm_backend,
        dtype=torch.float32,
    )
    t0 = time.time()
    trt_result = trt_runner.infer(image_paths)
    trt_time = time.time() - t0
    trt_pred = trt_result["pred_traj"][np.newaxis, ...]
    print(f"[TRT] Inference time: {trt_time:.3f}s")

    pt_metrics = compute_metrics(pt_pred, gt)
    trt_metrics = compute_metrics(trt_pred, gt)
    diff = np.abs(pt_pred - trt_pred)

    print("\n" + "=" * 60)
    print("METRICS")
    print("=" * 60)
    print(f"PyTorch ADE/FDE: {pt_metrics['ADE']:.4f} / {pt_metrics['FDE']:.4f}")
    print(f"TRT     ADE/FDE: {trt_metrics['ADE']:.4f} / {trt_metrics['FDE']:.4f}")
    print(f"PyTorch vs TRT max abs diff: {diff.max():.6e}")
    print(f"PyTorch vs TRT mean abs diff: {diff.mean():.6e}")
    print(f"PyTorch time: {pt_time:.3f}s")
    print(f"TRT time:     {trt_time:.3f}s")
    print(f"Speedup:      {pt_time/trt_time:.2f}x" if trt_time > 0 else "Speedup: N/A")

    # TRT engines introduce small numerical deviations; for a trajectory-regression
    # smoke test the tolerance is relaxed compared with text-generation deployment.
    checks = [
        ("PyTorch vs TRT max diff < 2.0", diff.max() < 2.0),
        ("PyTorch vs TRT mean diff < 0.5", diff.mean() < 0.5),
    ]
    all_ok = True
    print("\n[SANITY CHECKS]")
    for name, ok in checks:
        symbol = "✓" if ok else "✗"
        print(f"  {symbol} {name}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            all_ok = False

    if all_ok:
        print("\n🎉 Comparison PASSED.")
    else:
        print("\n⚠️  Some checks failed. The TRT LLM engine (`--llm_backend trt`) can")
        print("    amplify numerical deviations; use `--llm_backend pytorch` for tighter matching.")


def main():
    parser = argparse.ArgumentParser(description="Compare PyTorch and TensorRT VLA outputs")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--base_model_path", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--dataroot", type=str, default="/root/nuscenes")
    parser.add_argument("--vision_engine", type=str, default=DEFAULT_VISION_ENGINE)
    parser.add_argument("--llm_engine", type=str, default=DEFAULT_LLM_ENGINE)
    parser.add_argument("--action_engine", type=str, default=DEFAULT_ACTION_ENGINE)
    parser.add_argument("--llm_backend", type=str, default="pytorch", choices=["trt", "pytorch"],
                        help="LLM backend for TRT path. PyTorch is recommended for accuracy.")
    parser.add_argument("--bf16", action="store_true", default=True, help="Use bfloat16 for PyTorch (default)")
    parser.add_argument("--fp32", action="store_true", help="Use float32 for PyTorch to match TRT precision")
    args = parser.parse_args()

    args.bf16 = not args.fp32
    compare(args)


if __name__ == "__main__":
    main()
