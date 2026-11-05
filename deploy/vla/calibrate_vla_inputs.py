#!/usr/bin/env python3
"""
Generate INT8 calibration data for VLA LLM hidden-state engine.

This script loads a trained VLA checkpoint, samples N examples from the
NuScenes validation set, and runs the PyTorch model up to the
`inputs_embeds` stage (vision encoder + mlp1 + text embedding + query
tokens). The resulting tensors are saved as NumPy arrays and are used by
`export_vla_llm_hidden_trt.py --int8` as calibration input.

Output (default: deploy/outputs/vla_llm_calib_data.npz):
    inputs_embeds:  (N, seq_len, hidden_size)  float32
    attention_mask: (N, seq_len)                int64
    position_ids:   (N, seq_len)                int64
    seq_lens:       (N,)                        int64   # per-sample length before padding
"""
import argparse
import os
import sys
from typing import Dict, List

import numpy as np
import torch
import yaml

import warnings  # noqa
warnings.filterwarnings("ignore", category=FutureWarning)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from deploy.vla.utils import load_checkpoint
from vla.data import NuScenesVLADataModule
from vla.lightning import VLAPlModule


DEFAULT_CHECKPOINT = os.path.join(_PROJECT_ROOT, "outputs", "nuscenes_internvl2_5_real_v2", "checkpoints", "last.ckpt")
DEFAULT_CONFIG = os.path.join(_PROJECT_ROOT, "configs", "nuscenes_mini_internvl2_5_vla.yaml")
DEFAULT_OUTPUT = os.path.join(_PROJECT_ROOT, "deploy", "outputs", "vla_llm_calib_data.npz")
DEFAULT_BASE_MODEL = "/root/embodvla/models/OpenGVLab/InternVL2_5-1B"


def load_config(config_path: str) -> Dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


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


@torch.no_grad()
def extract_inputs_embeds(model, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Run VLA forward up to LLM inputs_embeds and return them along with masks/position_ids."""
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    pixel_values = batch["pixel_values"].to(model.dtype)
    B = input_ids.shape[0]
    device = input_ids.device
    num_queries = model.num_queries

    # Vision + text embedding + visual injection.
    vit_embeds = model.internvl.extract_feature(pixel_values)
    input_embeds = model.internvl.get_input_embeddings()(input_ids)
    input_embeds = model._inject_visual_features(input_embeds, input_ids, vit_embeds)

    # Append query tokens.
    query_ids = torch.arange(num_queries, device=device).unsqueeze(0).expand(B, -1)
    query_embeds = model.query_tokens(query_ids).to(input_embeds.dtype)
    inputs_embeds = torch.cat([input_embeds, query_embeds], dim=1)

    query_attention_mask = torch.ones(B, num_queries, device=device, dtype=attention_mask.dtype)
    full_attention_mask = torch.cat([attention_mask, query_attention_mask], dim=1)

    # position_ids: [0, 1, ..., seq_len-1] for each sample.
    seq_len = inputs_embeds.shape[1]
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(B, -1)

    return {
        "inputs_embeds": inputs_embeds,
        "attention_mask": full_attention_mask,
        "position_ids": position_ids,
    }


def pad_or_truncate_seq(tensor: np.ndarray, target_len: int, pad_value: int = 0) -> np.ndarray:
    """Pad/truncate a (N, seq_len, *) array to target_len on axis 1."""
    cur_len = tensor.shape[1]
    if cur_len == target_len:
        return tensor
    if cur_len > target_len:
        return tensor[:, :target_len]
    pad_width = [(0, 0)] * tensor.ndim
    pad_width[1] = (0, target_len - cur_len)
    return np.pad(tensor, pad_width, mode="constant", constant_values=pad_value)


def collect_calibration_data(
    checkpoint_path: str,
    config_path: str,
    output_path: str,
    num_samples: int = 100,
    base_model_path: str = DEFAULT_BASE_MODEL,
    split: str = "val",
):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

    # Prefer the model_path saved in the checkpoint so different backbones work out-of-the-box.
    resolved_base_model_path = _get_model_path_from_checkpoint(checkpoint_path, base_model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model.
    print(f"[Calibrate] Loading checkpoint from {checkpoint_path} ...")
    pl_module = VLAPlModule.load_from_checkpoint(
        checkpoint_path,
        model_path=resolved_base_model_path,
        num_queries=model_cfg.get("num_queries", 24),
        freeze_vision=True,
        freeze_llm=False,
        freeze_mlp1=True,
        action_head_dims=tuple(model_cfg.get("action_head_dims", [512, 2])),
        lr=1e-4,
        weight_decay=0.01,
        warmup_steps=100,
        max_steps=5000,
        dtype=torch.float32,
        local_files_only=True,
    )
    pl_module.eval().to(device)
    for p in pl_module.parameters():
        p.requires_grad = False

    # Build datamodule.
    print(f"[Calibrate] Loading NuScenes from {data_cfg['dataroot']} ...")
    dm = NuScenesVLADataModule(
        dataroot=data_cfg["dataroot"],
        version=data_cfg["version"],
        tokenizer=pl_module.model.tokenizer,
        input_size=data_cfg["input_size"],
        num_history_frames=data_cfg["num_history_frames"],
        n_future=data_cfg["n_future"],
        dt=data_cfg["dt"],
        cam_channel=data_cfg["cam_channel"],
        batch_size=1,
        num_workers=data_cfg.get("num_workers", 4),
        val_split=data_cfg["val_split"],
        prompt=data_cfg["prompt"],
        cache_dir=data_cfg.get("cache_dir"),
        num_image_tokens=data_cfg.get("num_image_tokens", 256),
        max_length=data_cfg.get("max_length", 2048),
    )
    dm.setup(stage="validate")
    dataset = dm.val_dataset if split == "val" else dm.train_dataset
    dataloader = dm.val_dataloader() if split == "val" else dm.train_dataloader()

    # Sample up to num_samples unique examples.
    num_samples = min(num_samples, len(dataset))
    print(f"[Calibrate] Collecting {num_samples} calibration samples from {split} split ...")

    all_inputs_embeds: List[np.ndarray] = []
    all_attention_mask: List[np.ndarray] = []
    all_position_ids: List[np.ndarray] = []
    seq_lens: List[int] = []

    collected = 0
    for batch in dataloader:
        if collected >= num_samples:
            break
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        out = extract_inputs_embeds(pl_module.model, batch)

        # Convert to float32 / int64 numpy.
        inputs_embeds = out["inputs_embeds"].float().cpu().numpy()
        attention_mask = out["attention_mask"].long().cpu().numpy()
        position_ids = out["position_ids"].long().cpu().numpy()

        B = inputs_embeds.shape[0]
        for i in range(B):
            if collected >= num_samples:
                break
            # Remove padding for this sample to keep calibration data realistic.
            mask_i = attention_mask[i]
            valid_len = int(mask_i.sum())
            if valid_len == 0:
                valid_len = inputs_embeds.shape[1]

            emb = inputs_embeds[i:i + 1, :valid_len]
            msk = mask_i[np.newaxis, :valid_len]
            pos = position_ids[i:i + 1, :valid_len]

            all_inputs_embeds.append(emb)
            all_attention_mask.append(msk)
            all_position_ids.append(pos)
            seq_lens.append(valid_len)
            collected += 1

    # Pad all samples to the same length so we can stack.
    max_len = max(seq_lens)
    print(f"[Calibrate] Max sequence length in calibration data: {max_len}")

    inputs_embeds_arr = np.concatenate(
        [pad_or_truncate_seq(x, max_len, pad_value=0.0) for x in all_inputs_embeds], axis=0
    )
    attention_mask_arr = np.concatenate(
        [pad_or_truncate_seq(x, max_len, pad_value=0) for x in all_attention_mask], axis=0
    )
    position_ids_arr = np.concatenate(
        [pad_or_truncate_seq(x, max_len, pad_value=0) for x in all_position_ids], axis=0
    )
    seq_lens_arr = np.array(seq_lens, dtype=np.int64)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(
        output_path,
        inputs_embeds=inputs_embeds_arr.astype(np.float32),
        attention_mask=attention_mask_arr.astype(np.int64),
        position_ids=position_ids_arr.astype(np.int64),
        seq_lens=seq_lens_arr,
    )
    print(f"[Calibrate] Saved calibration data: {output_path}")
    print(f"[Calibrate] shapes: inputs_embeds={inputs_embeds_arr.shape}, "
          f"attention_mask={attention_mask_arr.shape}, position_ids={position_ids_arr.shape}")


def main():
    parser = argparse.ArgumentParser(description="Generate INT8 calibration data for VLA LLM")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--num_samples", type=int, default=100, help="Number of calibration samples")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--base_model_path", type=str, default=DEFAULT_BASE_MODEL)
    args = parser.parse_args()

    collect_calibration_data(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        output_path=args.output,
        num_samples=args.num_samples,
        base_model_path=args.base_model_path,
        split=args.split,
    )


if __name__ == "__main__":
    main()
