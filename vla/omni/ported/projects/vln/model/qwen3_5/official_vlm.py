"""Transformers-backed Qwen3.5 VLM wrapper for the VLN planner.

This module keeps DBN's planner-facing interface but delegates the actual Qwen3.5
vision and text backbones to the official ``transformers.models.qwen3_5``
implementation. The only DBN-specific logic left here is the bridge from the
planner's sampled image tensor ``[B, T, H, W, 3]`` to Qwen's flattened visual
patch contract, and unwrapping HuggingFace model outputs for ``LLMDecoder``.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from vla.omni.ported.projects.vln.model.qwen3_5.vlm import (
    Qwen35VisionAdapter,
    _load_config_dict_from_pretrained,
    _load_state_dict_any,
)
from vla.omni.ported.projects.vln.model.vlm.mock_vlm import BaseVLMForPlanner


class OfficialQwen35TextAdapter(nn.Module):
    """Adapter returning a tensor from official ``Qwen3_5TextModel.forward``."""

    def __init__(self, text_model: nn.Module) -> None:
        super().__init__()
        self.text_model = text_model
        self.config = text_model.config
        self.embed_tokens = text_model.embed_tokens

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        out = self.text_model(*args, **kwargs)
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state
        if isinstance(out, (tuple, list)):
            return out[0]
        return out


class OfficialQwen35VLM(BaseVLMForPlanner):
    """Official Transformers Qwen3.5 VLM backend for config-level switching.

    It exposes the same three getters as ``Qwen35VLM`` so changing only
    ``vision_language_model.type`` in YAML swaps DBN's dependency-light port for
    the official Transformers implementation.
    """

    def __init__(
        self,
        text_config: Optional[Dict[str, Any]] = None,
        vision_config: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        mm_projector: Optional[nn.Module] = None,
        pretrained_path: Optional[str] = None,
        pretrained_strict: bool = False,
        treat_image_sequence_as_video: bool = False,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        try:
            from transformers.models.qwen3_5.configuration_qwen3_5 import (
                Qwen3_5Config,
                Qwen3_5TextConfig,
                Qwen3_5VisionConfig,
            )
            from transformers.models.qwen3_5.modeling_qwen3_5 import (
                Qwen3_5TextModel,
                Qwen3_5VisionModel,
            )
        except ImportError as e:  # pragma: no cover - depends on external package version
            raise ImportError(
                "OfficialQwen35VLM requires a Transformers build that provides "
                "transformers.models.qwen3_5 (for example >=4.57 / current 5.x)."
            ) from e

        resolved_config = config
        if (
            resolved_config is None
            and text_config is None
            and vision_config is None
            and pretrained_path is not None
        ):
            resolved_config = _load_config_dict_from_pretrained(pretrained_path)

        if resolved_config is not None:
            self.config = Qwen3_5Config.from_dict(resolved_config)
        else:
            text_cfg = Qwen3_5TextConfig.from_dict(text_config or {})
            vision_cfg = Qwen3_5VisionConfig.from_dict(vision_config or {})
            self.config = Qwen3_5Config(text_config=text_cfg, vision_config=vision_cfg)

        self.vision_model = Qwen3_5VisionModel(self.config.vision_config)
        self.vision_adapter = Qwen35VisionAdapter(
            self.vision_model,
            self.config.vision_config,
            treat_image_sequence_as_video=treat_image_sequence_as_video,
        )
        self.language_model = OfficialQwen35TextAdapter(Qwen3_5TextModel(self.config.text_config))

        out_hidden = self.config.vision_config.out_hidden_size
        token_dim = self.config.text_config.hidden_size
        if mm_projector is not None:
            self.mm_projector: nn.Module = mm_projector
        elif out_hidden == token_dim:
            self.mm_projector = nn.Identity()
        else:
            self.mm_projector = nn.Linear(out_hidden, token_dim)

        if pretrained_path:
            self.load_pretrained(pretrained_path, strict=pretrained_strict)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        if self.gradient_checkpointing:
            self._enable_gradient_checkpointing()

    def _enable_gradient_checkpointing(self) -> None:
        """Enable activation recomputation on both official Qwen3.5 towers."""
        checkpoint_kwargs = {"use_reentrant": False}
        text_model = self.language_model.text_model
        text_model.config.use_cache = False
        for name, module in (
            ("language model", text_model),
            ("vision model", self.vision_model),
        ):
            enable = getattr(module, "gradient_checkpointing_enable", None)
            if not callable(enable):
                raise RuntimeError(
                    f"Official Qwen3.5 {name} does not support gradient checkpointing "
                    "in the installed Transformers version."
                )
            enable(gradient_checkpointing_kwargs=checkpoint_kwargs)

    def load_pretrained(self, path: str, strict: bool = False) -> None:
        """Load official Qwen3.5 checkpoint shards into official submodules."""
        state = _load_state_dict_any(path)
        lm_sd, vis_sd, ignored = {}, {}, []
        for k, v in state.items():
            if k.startswith("mtp.") or k.startswith("model.mtp."):
                continue
            if k.startswith("model.language_model."):
                lm_sd[k[len("model.language_model.") :]] = v
            elif k.startswith("language_model."):
                lm_sd[k[len("language_model.") :]] = v
            elif k.startswith("model.visual."):
                vis_sd[k[len("model.visual.") :]] = v
            elif k.startswith("visual."):
                vis_sd[k[len("visual.") :]] = v
            else:
                ignored.append(k)

        lm_res = self.language_model.text_model.load_state_dict(lm_sd, strict=strict)
        vis_res = self.vision_model.load_state_dict(vis_sd, strict=strict)
        print(
            f"[OfficialQwen35VLM] loaded pretrained from {path}: "
            f"language_model(missing={len(getattr(lm_res, 'missing_keys', []))}, "
            f"unexpected={len(getattr(lm_res, 'unexpected_keys', []))}), "
            f"vision_model(missing={len(getattr(vis_res, 'missing_keys', []))}, "
            f"unexpected={len(getattr(vis_res, 'unexpected_keys', []))}), "
            f"unrouted={len(ignored)}"
        )

    def get_vision_model(self) -> nn.Module:
        return self.vision_adapter

    def get_mm_projector(self) -> Optional[nn.Module]:
        return self.mm_projector

    def get_language_model(self) -> nn.Module:
        return self.language_model
