#!/usr/bin/env python3
"""
Compare PyTorch Qwen2-VL VLA inference with TensorRT VLA inference.

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
from vla.lightning import VLAPlModule
from run_trt_qwen2vl_vla import Qwen2VLVLATRTInference

DEFAULT_CHECKPOINT = os.path.join(_PROJECT_ROOT, "outputs", "nuscenes_qwen2vl_real_v2", "checkpoints", "last.ckpt")
DEFAULT_BASE_MODEL = os.path.join(_PROJECT_ROOT, "models", "Qwen", "Qwen2-VL-2B-Instruct")
DEFAULT_VISION_ENGINE = os.path.join(_PROJECT_ROOT, "deploy", "outputs", "qwen2vl_2b_vision.trt")
DEFAULT_LLM_ENGINE = os.path.join(_PROJECT_ROOT, "deploy", "outputs", "qwen2vl_vla_llm_hidden.trt")
DEFAULT_ACTION_ENGINE = os.path.join(_PROJECT_ROOT, "deploy", "outputs", "qwen2vl_vla_action_head.trt")


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


def run_pytorch(pl_module: VLAPlModule, image_paths: list, device: torch.device):
    model = pl_module.model
    model.eval().to(device)

    processor = model.tokenizer
    images = [Image.open(p).convert("RGB") for p in image_paths]
    content = [{"type": "image", "image": img} for img in images]
    content.append({"type": "text", "text": "Predict the future trajectory of the ego vehicle."})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, return_tensors="pt", padding=True)

    batch = {
        "pixel_values": inputs["pixel_values"].to(device),
        "image_grid_thw": inputs["image_grid_thw"].to(device),
        "input_ids": inputs["input_ids"].to(device),
        "attention_mask": inputs["attention_mask"].to(device),
        "mm_token_type_ids": inputs.get("mm_token_type_ids", torch.zeros_like(inputs["input_ids"], dtype=torch.int64)).to(device),
    }

    t0 = time.time()
    with torch.no_grad():
        if model.dtype == torch.bfloat16 and device.type != "cpu":
            with torch.amp.autocast(device.type, dtype=torch.bfloat16):
                outputs = model(trajectories=None, **batch)
        else:
            outputs = model(trajectories=None, **batch)
    elapsed = time.time() - t0
    pred = outputs["pred_traj"].float().cpu().numpy()
    print(f"[PyTorch] Inference time: {elapsed:.3f}s")
    return pred, elapsed


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
    print("QWEN2-VL VLA PYTORCH vs TENSORRT COMPARISON")
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
        model_type="qwen2vl",
        num_queries=24,
        freeze_vision=True,
        freeze_llm=True,
        freeze_action_head=False,
        action_head_dims=(512, 2),
        lr=1e-4,
        weight_decay=0.01,
        warmup_steps=100,
        max_steps=5000,
        dtype=dtype,
    )

    image_paths = args.image_paths
    if image_paths is None or len(image_paths) == 0:
        # Use random synthetic images for a quick comparison if none provided.
        import tempfile
        from PIL import Image
        image_paths = []
        for i in range(4):
            path = os.path.join(tempfile.mkdtemp(), f"img_{i}.jpg")
            arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            Image.fromarray(arr).convert("RGB").save(path)
            image_paths.append(path)

    pt_pred, pt_time = run_pytorch(pl_module, image_paths, device)
    print()

    print("[TRT] Loading deployment runner...")
    trt_runner = Qwen2VLVLATRTInference(
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

    diff = np.abs(pt_pred - trt_pred)

    print("\n" + "=" * 60)
    print("METRICS")
    print("=" * 60)
    print(f"PyTorch vs TRT max abs diff: {diff.max():.6e}")
    print(f"PyTorch vs TRT mean abs diff: {diff.mean():.6e}")
    print(f"PyTorch time: {pt_time:.3f}s")
    print(f"TRT time:     {trt_time:.3f}s")
    print(f"Speedup:      {pt_time/trt_time:.2f}x" if trt_time > 0 else "Speedup: N/A")

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
    parser = argparse.ArgumentParser(description="Compare PyTorch and TensorRT Qwen2-VL VLA outputs")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--base_model_path", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--vision_engine", type=str, default=DEFAULT_VISION_ENGINE)
    parser.add_argument("--llm_engine", type=str, default=DEFAULT_LLM_ENGINE)
    parser.add_argument("--action_engine", type=str, default=DEFAULT_ACTION_ENGINE)
    parser.add_argument("--image_paths", type=str, nargs="*", default=None)
    parser.add_argument("--llm_backend", type=str, default="pytorch", choices=["trt", "pytorch"],
                        help="LLM backend for TRT path. PyTorch is recommended for accuracy.")
    parser.add_argument("--bf16", action="store_true", default=True, help="Use bfloat16 for PyTorch (default)")
    parser.add_argument("--fp32", action="store_true", help="Use float32 for PyTorch to match TRT precision")
    args = parser.parse_args()

    args.bf16 = not args.fp32
    compare(args)


if __name__ == "__main__":
    main()
