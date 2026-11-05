"""Prompt DataProcessor for the VLN planner (system prompt + instruction + vision).

Builds the model prompt exactly in the layout requested for the quadruped VLN
policy, following the configurable DataProcessor layer of ``dbn.md`` section 5
(a per-sample callable that mutates ``data_dict`` in place, declared entirely in
the config's ``data_processor_list``):

    [ system prompt ] + [ instruction ] + [ vision block ]

* **system prompt** (fixed): the navigation-policy role description.
* **instruction**: by default, ``instruction_template.format(TARGET_DESC=...)``
  where ``TARGET_DESC`` is derived from the dataset's raw tracking instruction
  (e.g. "wearing a black and silver armored outfit"). When
  ``use_origin_prompt=True``, the processor skips regex extraction and template
  formatting, and directly uses the dataset's raw ``instruction_text`` after the
  system prompt.
* **vision block**: ``[vision_start] + pad_token * N + [vision_end]`` using the
  *real* Qwen3.5 special token ids. For video inputs the pad token is
  ``video_token_id`` / ``<|video_pad|>`` exactly like the official chat template;
  for legacy image inputs it remains ``image_token_id`` / ``<|image_pad|>``.

The processor tokenizes with the real Qwen3.5 tokenizer and emits a
**variable-length** ``PROMPT_IDS`` / ``ATTENTION_MASK`` (no truncation, no
padding). Batch-level left-padding to the longest sample is handled by
``planner_collate_fn``.

Pair with ``TrackVLADataset(emit_prompt=False)`` so the dataset provides the raw
``instruction_text`` and ``_num_vision_tokens`` this processor consumes.
"""

import re
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from vla.omni.ported.projects.vln.constant.keys import PlannerDataKeys
from vla.omni.ported.projects.vln.data.dataset.data_processor.base_data_processor import BaseDataProcessor

DEFAULT_SYSTEM_PROMPT = (
    "You are the navigation policy of a quadruped robot. Given the observations "
    "and the instruction, decide where to go next."
)
DEFAULT_INSTRUCTION_TEMPLATE = "Follow the person who is {TARGET_DESC}."

QWEN_IMAGE_TOKEN_ID = 248056
QWEN_VIDEO_TOKEN_ID = 248057
QWEN_VISION_START_ID = 248053
QWEN_VISION_END_ID = 248054

_DESC_RE = re.compile(
    r"\b(?:individual|person|man|woman|people|figure|guy|human)\b\s*(.*)",
    re.IGNORECASE,
)


class VLNPromptProcessor(BaseDataProcessor):
    def __init__(
        self,
        tokenizer_path: str,
        num_image_tokens: Optional[int] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        instruction_template: str = DEFAULT_INSTRUCTION_TEMPLATE,
        image_token_id: int = QWEN_IMAGE_TOKEN_ID,
        video_token_id: int = QWEN_VIDEO_TOKEN_ID,
        vision_start_id: int = QWEN_VISION_START_ID,
        vision_end_id: int = QWEN_VISION_END_ID,
        use_vision_markers: bool = True,
        vision_token_type: str = "image",
        split_video_by_timestamps: bool = False,
        timestamp_template: str = "<{time:.2f}s>",
        use_origin_prompt: bool = False,
    ) -> None:
        from tokenizers import Tokenizer

        self._tok = Tokenizer.from_file(tokenizer_path)
        self.num_image_tokens = num_image_tokens
        self.system_prompt = system_prompt
        self.instruction_template = instruction_template
        if vision_token_type not in {"image", "video"}:
            raise ValueError(f"vision_token_type must be 'image' or 'video', got {vision_token_type!r}")
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_id = vision_start_id
        self.vision_end_id = vision_end_id
        self.use_vision_markers = use_vision_markers
        self.vision_token_type = vision_token_type
        self.split_video_by_timestamps = split_video_by_timestamps
        self.timestamp_template = timestamp_template
        self.use_origin_prompt = use_origin_prompt

    @staticmethod
    def _target_desc(instruction: str) -> str:
        instruction = (instruction or "").strip().rstrip(".")
        m = _DESC_RE.search(instruction)
        desc = m.group(1).strip() if m and m.group(1).strip() else instruction
        return desc

    def _build_instruction(self, raw_instruction: str) -> str:
        raw_instruction = (raw_instruction or "").strip()
        if self.use_origin_prompt:
            return raw_instruction
        return self.instruction_template.format(TARGET_DESC=self._target_desc(raw_instruction))

    def _encode(self, text: str) -> List[int]:
        return self._tok.encode(text).ids

    def __call__(self, data_dict: Dict[str, Any], mode: str = "train") -> None:
        raw = data_dict.get("instruction_text", "")
        n_vision = self.num_image_tokens
        if n_vision is None:
            if self.vision_token_type == "video":
                n_vision = data_dict.get("_num_video_tokens", data_dict.get("_num_vision_tokens", 0))
            else:
                n_vision = data_dict.get("_num_image_tokens", data_dict.get("_num_vision_tokens", 0))
        n_vision = int(n_vision or 0)
        vision_token_id = self.video_token_id if self.vision_token_type == "video" else self.image_token_id

        instruction = self._build_instruction(raw)
        text_ids = self._encode(self.system_prompt + " " + instruction)

        mm_token_type_ids = [0] * len(text_ids)
        prompt_text_parts = [f"{self.system_prompt} {instruction} "]

        if self.vision_token_type == "video" and self.split_video_by_timestamps:
            grid_t = int(data_dict.get("_video_grid_t", 0) or 0)
            tokens_per_grid = int(data_dict.get("_tokens_per_video_grid", 0) or 0)
            input_hz = float(data_dict.get("_input_hz", 1.0) or 1.0)
            temporal_patch_size = int(data_dict.get("_temporal_patch_size", 1) or 1)
            grid_timestamps = data_dict.get("_video_grid_timestamps")
            if grid_t <= 0 or tokens_per_grid <= 0:
                raise ValueError(
                    "split_video_by_timestamps=True requires _video_grid_t and "
                    "_tokens_per_video_grid from the dataset."
                )
            if grid_t * tokens_per_grid != n_vision:
                raise ValueError(
                    f"video token split mismatch: grid_t({grid_t}) * "
                    f"tokens_per_grid({tokens_per_grid}) != n_vision({n_vision})"
                )
            if grid_timestamps is not None:
                if isinstance(grid_timestamps, torch.Tensor):
                    grid_timestamps = grid_timestamps.detach().cpu().flatten().tolist()
                elif isinstance(grid_timestamps, np.ndarray):
                    grid_timestamps = grid_timestamps.reshape(-1).tolist()
                else:
                    grid_timestamps = list(grid_timestamps)
                if len(grid_timestamps) != grid_t:
                    raise ValueError(
                        "video grid timestamp mismatch: "
                        f"len({len(grid_timestamps)}) != grid_t({grid_t})"
                    )

            vision_block = []
            for t_idx in range(grid_t):
                # Qwen3.5 official video prompts are timestamp-separated. The
                # dataset may provide actual timestamps after nearest-frame
                # selection; legacy datasets fall back to their nominal input_hz.
                if grid_timestamps is None:
                    time_sec = (t_idx * temporal_patch_size) / input_hz
                else:
                    time_sec = float(grid_timestamps[t_idx])
                timestamp = self.timestamp_template.format(time=time_sec, index=t_idx)
                ts_ids = self._encode(timestamp + " ")
                vision_block.extend(ts_ids)
                mm_token_type_ids.extend([0] * len(ts_ids))
                if self.use_vision_markers:
                    vision_block.append(self.vision_start_id)
                    mm_token_type_ids.append(0)
                vision_block.extend([vision_token_id] * tokens_per_grid)
                mm_token_type_ids.extend([2] * tokens_per_grid)
                if self.use_vision_markers:
                    vision_block.append(self.vision_end_id)
                    mm_token_type_ids.append(0)
                prompt_text_parts.append(
                    timestamp
                    + " "
                    + ("<vision_start>" if self.use_vision_markers else "")
                    + "<video>" * tokens_per_grid
                    + ("<vision_end>" if self.use_vision_markers else "")
                )
        else:
            if self.use_vision_markers:
                vision_block = [self.vision_start_id] + [vision_token_id] * n_vision + [self.vision_end_id]
                mm_token_type_ids.extend([0] + [2 if self.vision_token_type == "video" else 1] * n_vision + [0])
            else:
                vision_block = [vision_token_id] * n_vision
                mm_token_type_ids.extend([2 if self.vision_token_type == "video" else 1] * n_vision)
            pad_text = "<video>" if self.vision_token_type == "video" else "<image>"
            prompt_text_parts.append(
                ("<vision_start>" if self.use_vision_markers else "")
                + pad_text * n_vision
                + ("<vision_end>" if self.use_vision_markers else "")
            )

        ids = text_ids + vision_block
        actual_len = len(ids)
        if len(mm_token_type_ids) != actual_len:
            raise ValueError(
                f"mm_token_type_ids length mismatch: {len(mm_token_type_ids)} vs prompt length {actual_len}"
            )
        prompt = np.asarray(ids, dtype=np.int64)
        attn = np.ones(actual_len, dtype=np.int64)
        mm_types = np.asarray(mm_token_type_ids, dtype=np.int64)

        data_dict[PlannerDataKeys.PROMPT_IDS] = torch.from_numpy(prompt)
        data_dict[PlannerDataKeys.ATTENTION_MASK] = torch.from_numpy(attn)
        data_dict["mm_token_type_ids"] = torch.from_numpy(mm_types)
        data_dict["prompt_text"] = "".join(prompt_text_parts)
