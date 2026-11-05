"""VLM wrappers for the planner (ai_foundation ``model/vlm`` parity).

Mirrors the role of ``TuringVL`` / ``QWen2VL``: a single ``vision_language_model``
config node that holds the vision backbone + multimodal projector + language
model, and exposes ``get_vision_model`` / ``get_mm_projector`` /
``get_language_model`` so the planner arch can decompose it into a VisionEncoder
and an LLMDecoder.

* ``BaseVLMForPlanner`` - interface the planner depends on.
* ``MockVLM``           - dependency-light: registry-built vision_model + llm_model
                          + a linear mm_projector; runnable/testable without Qwen.
* ``QwenVLPlanner``     - real Qwen2.5-VL extension point (documented stub).

Sub-models (``MockVisionModel`` / ``MockLLM``) live here too so the whole VLM is
config-describable exactly like the e2e release yaml.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Sub-models (config-built, like ViTEVA / TuringText01 in the e2e config)
# ---------------------------------------------------------------------------
class MockVisionModel(nn.Module):
    """Tiny CNN vision backbone: [B, n_img, H, W, 3] -> [B, n_img, vis_dim]."""

    def __init__(self, vis_dim: int = 128, in_channels: int = 3, trainable: bool = True) -> None:
        super().__init__()
        self.vis_dim = vis_dim
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, stride=2, padding=2), nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(64, vis_dim),
        )
        for p in self.parameters():
            p.requires_grad_(trainable)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        B, n_img = rgb.shape[0], rgb.shape[1]
        x = rgb.flatten(0, 1).permute(0, 3, 1, 2).contiguous()  # [B*n_img, 3, H, W]
        feat = self.features(x)                                 # [B*n_img, vis_dim]
        return feat.view(B, n_img, self.vis_dim)


class _LLMConfig:
    def __init__(self, hidden_size: int) -> None:
        self.hidden_size = hidden_size


class MockLLM(nn.Module):
    """Tiny transformer LM operating on inputs_embeds -> last hidden states.

    Exposes ``config.hidden_size`` and ``embed_tokens`` like a HF causal LM so the
    planner can infer ``token_dim`` and embed prompt ids.
    """

    def __init__(
        self,
        vocab_size: int = 256,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        max_len: int = 96,
    ) -> None:
        super().__init__()
        self.config = _LLMConfig(hidden_size)
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, hidden_size) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size * 4,
            batch_first=True, activation="gelu", dropout=0.0,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        L = inputs_embeds.size(1)
        x = inputs_embeds + self.pos_embed[:, :L]
        key_padding = (attention_mask == 0) if attention_mask is not None else None
        return self.encoder(x, src_key_padding_mask=key_padding)


# ---------------------------------------------------------------------------
# VLM wrappers
# ---------------------------------------------------------------------------
class BaseVLMForPlanner(nn.Module, ABC):
    @abstractmethod
    def get_vision_model(self) -> nn.Module: ...

    @abstractmethod
    def get_mm_projector(self) -> Optional[nn.Module]: ...

    @abstractmethod
    def get_language_model(self) -> nn.Module: ...


class MockVLM(BaseVLMForPlanner):
    def __init__(
        self,
        vision_model: nn.Module,
        llm_model: nn.Module,
        mm_projector: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.vision_model = vision_model
        self.llm_model = llm_model
        token_dim = llm_model.config.hidden_size
        vis_dim = getattr(vision_model, "vis_dim", token_dim)
        # project visual tokens into the LLM embedding space (like TuringVL merger)
        self.mm_projector = mm_projector or nn.Linear(vis_dim, token_dim)

    def get_vision_model(self) -> nn.Module:
        return self.vision_model

    def get_mm_projector(self) -> Optional[nn.Module]:
        return self.mm_projector

    def get_language_model(self) -> nn.Module:
        return self.llm_model


class QwenVLPlanner(BaseVLMForPlanner):
    """Real Qwen2.5-VL VLM for the planner. Extension point (documented stub)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        raise NotImplementedError(
            "QwenVLPlanner is a stub. Port TuringVL/QWen2VL: hold the real vision "
            "backbone + mm projector + LLM and expose the three getters."
        )

    def get_vision_model(self) -> nn.Module:
        raise NotImplementedError

    def get_mm_projector(self) -> Optional[nn.Module]:
        raise NotImplementedError

    def get_language_model(self) -> nn.Module:
        raise NotImplementedError
