"""VisionEncoder - wraps vision_model + optional mm_projector + normalize/cast.

Faithful (dependency-light) analogue of ai_foundation's
``model/arch/components/vision_encoder.py``: takes the raw image dict, normalizes
& casts it, runs the vision backbone, then projects tokens into the LLM embedding
space via ``mm_projector``. Produces per-image visual token embeddings the arch
scatters into the prompt at image-token positions.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from vla.omni.ported.projects.vln.constant.keys import PlannerDataKeys
from vla.omni.ported.projects.vln.model.arch.components.image_normalize_and_cast import ImageNormalizeAndCast


class VisionEncoder(nn.Module):
    def __init__(
        self,
        vision_model: nn.Module,
        mm_projector: Optional[nn.Module] = None,
        vision_token_downsampler: Optional[nn.Module] = None,
        image_normalize_and_cast: Optional[ImageNormalizeAndCast] = None,
    ) -> None:
        super().__init__()
        self.vision_model = vision_model
        self.mm_projector = mm_projector
        self.vision_token_downsampler = vision_token_downsampler
        self.image_normalize_and_cast = image_normalize_and_cast or ImageNormalizeAndCast()
        # kept for arch-level branching parity with ai_foundation (unused here)
        self.is_qwen2vl_vision = False
        self.is_qwen3vl_vision = False
        self.is_turing_vision = False

    def forward(self, image_dict: Dict[str, Any]) -> torch.Tensor:
        image_dict = self.image_normalize_and_cast(image_dict)
        rgb = image_dict[PlannerDataKeys.RGB]  # [B, n_img, H, W, 3]
        tokens = self.vision_model(rgb)         # [B, n_img_tokens, vis_dim]
        if self.vision_token_downsampler is not None:
            tokens = self.vision_token_downsampler(tokens)
        if self.mm_projector is not None:
            tokens = self.mm_projector(tokens)  # -> [B, n_img_tokens, token_dim]
        return tokens
