"""Collate for the e2e-style planner dataset.

Handles variable-length prompt sequences via **left-padding** to the batch max
length: the last real token of every sample aligns at the same position, and
padding positions (filled with 0) have attention_mask=0 so they are excluded
from attention computation.

Fixed-shape tensors (rgb, future_trajectory, future_trajectory_mask) are simply
stacked.
"""

from typing import Dict, List

import torch

from vla.omni.ported.constant.data import DataMetricKeys
from vla.omni.ported.projects.vln.constant.keys import OmTrackDataKeys, PlannerDataKeys

PAD_TOKEN_ID = 0


def planner_collate_fn(batch: List[Dict]) -> Dict:
    def stack(key):
        return torch.stack([b[key] for b in batch], dim=0)

    # --- variable-length prompt: left-pad to batch max ---
    prompts = [b[PlannerDataKeys.PROMPT_IDS] for b in batch]
    masks = [b[PlannerDataKeys.ATTENTION_MASK] for b in batch]
    mm_types = [b.get("mm_token_type_ids") for b in batch]
    has_mm_types = all(m is not None for m in mm_types)
    max_len = max(p.size(0) for p in prompts)

    padded_prompts = []
    padded_masks = []
    padded_mm_types = []
    for idx, (p, m) in enumerate(zip(prompts, masks)):
        pad_len = max_len - p.size(0)
        if pad_len > 0:
            padded_prompts.append(
                torch.cat([torch.full((pad_len,), PAD_TOKEN_ID, dtype=p.dtype), p])
            )
            padded_masks.append(
                torch.cat([torch.zeros(pad_len, dtype=m.dtype), m])
            )
            if has_mm_types:
                mt = mm_types[idx]
                padded_mm_types.append(torch.cat([torch.zeros(pad_len, dtype=mt.dtype), mt]))
        else:
            padded_prompts.append(p)
            padded_masks.append(m)
            if has_mm_types:
                padded_mm_types.append(mm_types[idx])

    out = {
        PlannerDataKeys.RGB: stack(PlannerDataKeys.RGB),
        PlannerDataKeys.PROMPT_IDS: torch.stack(padded_prompts, dim=0),
        PlannerDataKeys.ATTENTION_MASK: torch.stack(padded_masks, dim=0),
        PlannerDataKeys.FUTURE_TRAJECTORY: stack(PlannerDataKeys.FUTURE_TRAJECTORY),
        PlannerDataKeys.FUTURE_TRAJECTORY_MASK: stack(PlannerDataKeys.FUTURE_TRAJECTORY_MASK),
    }

    # Optional scalar metadata used by Qwen video/MRoPE alignment. Keep legacy
    # datasets working by forwarding only keys that are present in every sample.
    if has_mm_types:
        out["mm_token_type_ids"] = torch.stack(padded_mm_types, dim=0)

    for key in (OmTrackDataKeys.CURRENT_PATH, OmTrackDataKeys.INSTRUCTION):
        if all(key in item for item in batch):
            out[key] = [item[key] for item in batch]

    for key in (
        "_num_image_tokens",
        "_num_video_tokens",
        "_num_vision_tokens",
        "_video_grid_t",
        "_tokens_per_video_grid",
        "_input_hz",
        "_temporal_patch_size",
    ):
        if all(key in b for b in batch):
            out[key] = torch.as_tensor([b[key] for b in batch])

    for key in (
        DataMetricKeys.RESAMPLE_COUNT,
        DataMetricKeys.READ_ATTEMPT_COUNT,
    ):
        if all(key in b for b in batch):
            out[key] = torch.as_tensor([b[key] for b in batch], dtype=torch.long).sum()

    for key in (DataMetricKeys.LOAD_TIME_S, DataMetricKeys.PROCESS_TIME_S):
        if all(key in b for b in batch):
            out[key] = torch.as_tensor(
                [b[key] for b in batch], dtype=torch.float64
            ).mean()

    return out
