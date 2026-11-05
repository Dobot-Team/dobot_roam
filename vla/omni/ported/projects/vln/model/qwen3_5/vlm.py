"""Qwen3.5 VLM wrapper for the VLN planner (``model/vlm`` parity, real backbone).

Plays the same role as ``MockVLM`` / ``TuringVL`` but wraps the *real* Qwen3.5
vision + language stacks (``model/qwen3_5``). It exposes the three getters the
planner decomposes on:

* ``get_vision_model()``    -> a ``Qwen35VisionAdapter`` producing per-video visual
                              tokens ``[B, n_tokens, out_hidden_size]`` from the
                              framework's ``[B, n_img, H, W, 3]`` rgb tensor. The
                              ``n_img`` dimension is treated as video time, not as
                              independent images.
* ``get_mm_projector()``    -> projects visual tokens into the LLM embedding space
                              (Identity when ``out_hidden_size == hidden_size``).
* ``get_language_model()``  -> the ``Qwen3_5TextModel`` (exposes ``embed_tokens``,
                              ``config.hidden_size`` and an ``inputs_embeds`` forward).

The vision adapter patchifies rgb exactly like the official Qwen video processor
(temporal + spatial-merge block ordering) so the merger / rotary / position-id logic
stays consistent with the ported ``Qwen3_5VisionModel``. Weights are random-init by
default; pass ``pretrained_path`` (a ``.pt``/``.safetensors`` file or an HF sharded
directory) to load an official Qwen3.5 checkpoint into the vision + language towers.
An official release ``config.json`` loads verbatim through ``Qwen3_5Config.from_dict``.
"""

import json
import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from vla.omni.ported.projects.vln.model.qwen3_5.configuration_qwen3_5 import (
    Qwen3_5Config,
    Qwen3_5TextConfig,
    Qwen3_5VisionConfig,
)
from vla.omni.ported.projects.vln.model.qwen3_5.modeling_text import Qwen3_5TextModel
from vla.omni.ported.projects.vln.model.qwen3_5.modeling_vision import Qwen3_5VisionModel
from vla.omni.ported.projects.vln.model.vlm.mock_vlm import BaseVLMForPlanner


def _load_config_dict_from_pretrained(path: str) -> Dict[str, Any]:
    """Load the HuggingFace ``config.json`` next to a checkpoint path.

    ``path`` may be either the checkpoint directory or one weight shard inside
    it.  Keeping this lookup separate from weight loading lets a single DBN
    config select Qwen3.5-0.8B/2B/4B by changing only ``pretrained_path``.
    """
    checkpoint_dir = path if os.path.isdir(path) else os.path.dirname(os.path.abspath(path))
    config_path = os.path.join(checkpoint_dir, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"Cannot infer Qwen3.5 architecture: {config_path} does not exist. "
            "Point pretrained_path at a HuggingFace checkpoint directory (or a "
            "weight file next to config.json), or provide text_config and "
            "vision_config explicitly."
        )
    with open(config_path, encoding="utf-8") as config_file:
        config = json.load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a JSON object in {config_path}, got {type(config).__name__}")
    if not isinstance(config.get("text_config"), dict) or not isinstance(
        config.get("vision_config"), dict
    ):
        raise ValueError(f"Qwen3.5 config {config_path} must define text_config and vision_config")
    return config


class Qwen35VisionAdapter(nn.Module):
    """Bridges ``[B, n_img, H, W, 3]`` rgb to the official Qwen3.5 ViT contract.

    The planner dataset already returns a sequence of sampled frames. For Qwen3.5
    video parity that sequence must be patchified as one video clip per sample:
    ``grid_thw=[ceil(n_img / temporal_patch_size), H/patch, W/patch]``. Treating
    the same tensor as ``B*n_img`` independent images would double the visual-token
    count for the released ``temporal_patch_size=2`` config and would remove the
    temporal axis from the vision tower.
    """

    def __init__(
        self,
        vision_model: Qwen3_5VisionModel,
        config: Qwen3_5VisionConfig,
        treat_image_sequence_as_video: bool = False,
    ) -> None:
        super().__init__()
        self.visual = vision_model
        self.treat_image_sequence_as_video = treat_image_sequence_as_video
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.merge_size = config.spatial_merge_size
        self.in_channels = config.in_channels
        # advertised token dim consumed by the planner / mm_projector
        self.vis_dim = config.out_hidden_size

    def get_grid_thw(self, rgb: torch.Tensor) -> torch.Tensor:
        """Return Qwen video ``grid_thw`` for ``rgb`` shaped ``[B, T, H, W, C]``."""
        B, T, H, W, _ = rgb.shape
        p, tps = self.patch_size, self.temporal_patch_size
        gh, gw = H // p, W // p
        grid_t = (T + tps - 1) // tps
        return torch.tensor([[grid_t, gh, gw]] * B, device=rgb.device, dtype=torch.long)

    def num_visual_tokens(self, rgb: torch.Tensor) -> int:
        """Number of merged visual tokens produced per sample for ``rgb``."""
        if not self.treat_image_sequence_as_video:
            _, n_img, H, W, _ = rgb.shape
            gh, gw = H // self.patch_size, W // self.patch_size
            return int(n_img * (gh // self.merge_size) * (gw // self.merge_size))
        grid_t, gh, gw = self.get_grid_thw(rgb)[0].tolist()
        return int(grid_t * (gh // self.merge_size) * (gw // self.merge_size))

    def _patchify_images(self, rgb: torch.Tensor):
        """Legacy image mode: ``[N, H, W, C]`` -> independent-image patches."""
        N, H, W, C = rgb.shape
        p, tps, msz = self.patch_size, self.temporal_patch_size, self.merge_size
        gh, gw = H // p, W // p
        x = rgb.permute(0, 3, 1, 2)
        x = x.unsqueeze(1).repeat(1, tps, 1, 1, 1)
        x = x.reshape(N, 1, tps, C, gh // msz, msz, p, gw // msz, msz, p)
        x = x.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9).contiguous()
        flat = x.reshape(N * gh * gw, C * tps * p * p)
        grid_thw = torch.tensor([[1, gh, gw]] * N, device=rgb.device, dtype=torch.long)
        return flat, grid_thw

    def _patchify_video(self, rgb: torch.Tensor):
        """rgb ``[B, T, H, W, C]`` -> (flat_patches, grid_thw) matching Qwen video processing."""
        B, T, H, W, C = rgb.shape
        p, tps, msz = self.patch_size, self.temporal_patch_size, self.merge_size
        if C != self.in_channels:
            raise ValueError(f"expected {self.in_channels} RGB channels, got {C}")
        if H % p != 0 or W % p != 0:
            raise ValueError(f"H/W must be divisible by patch_size={p}, got {(H, W)}")
        gh, gw = H // p, W // p
        if gh % msz != 0 or gw % msz != 0:
            raise ValueError(
                f"grid H/W must be divisible by spatial_merge_size={msz}, got {(gh, gw)}"
            )

        # Official Qwen video processors pad the temporal axis to a whole temporal
        # patch. We repeat the last sampled frame, keeping the input as images while
        # presenting the vision tower with one video clip per batch item.
        pad_t = (-T) % tps
        if pad_t:
            rgb = torch.cat([rgb, rgb[:, -1:].expand(-1, pad_t, -1, -1, -1)], dim=1)
        grid_t = rgb.shape[1] // tps

        x = rgb.permute(0, 1, 4, 2, 3).contiguous()      # [B, T_pad, C, H, W]
        x = x.reshape(B, grid_t, tps, C, gh // msz, msz, p, gw // msz, msz, p)
        # -> [B, grid_t, gh//msz, gw//msz, msz, msz, C, tps, p, p]
        x = x.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9).contiguous()
        flat = x.reshape(B * grid_t * gh * gw, C * tps * p * p)
        grid_thw = torch.tensor([[grid_t, gh, gw]] * B, device=rgb.device, dtype=torch.long)
        return flat, grid_thw

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        B = rgb.shape[0]
        if self.treat_image_sequence_as_video:
            flat, grid_thw = self._patchify_video(rgb)
            tokens = self.visual(flat, grid_thw)                # [B*grid_t*gh*gw/msz^2, out_hidden]
        else:
            flat, grid_thw = self._patchify_images(rgb.flatten(0, 1))
            tokens = self.visual(flat, grid_thw)                # [B*n_img*gh*gw/msz^2, out_hidden]

        # DBN's local port returns the merged visual tokens directly. The official
        # Transformers implementation returns BaseModelOutputWithPooling, where
        # pooler_output is the merger output consumed by Qwen's language tower.
        if hasattr(tokens, "pooler_output") and tokens.pooler_output is not None:
            tokens = tokens.pooler_output
        elif hasattr(tokens, "last_hidden_state"):
            tokens = tokens.last_hidden_state
        elif isinstance(tokens, (tuple, list)):
            tokens = tokens[0]

        tokens = tokens.reshape(B, -1, tokens.shape[-1])        # [B, visual_tokens, out_hidden]
        return tokens


class Qwen35VLM(BaseVLMForPlanner):
    """Real Qwen3.5 VLM: vision ViT + mm projector + hybrid linear/full LLM."""

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
            self.config = Qwen3_5Config(
                text_config=Qwen3_5TextConfig.from_dict(text_config) if text_config else None,
                vision_config=Qwen3_5VisionConfig.from_dict(vision_config) if vision_config else None,
            )

        self.vision_model = Qwen3_5VisionModel(self.config.vision_config)
        self.vision_adapter = Qwen35VisionAdapter(
            self.vision_model,
            self.config.vision_config,
            treat_image_sequence_as_video=treat_image_sequence_as_video,
        )
        self.language_model = Qwen3_5TextModel(self.config.text_config)

        out_hidden = self.config.vision_config.out_hidden_size
        token_dim = self.config.text_config.hidden_size
        if mm_projector is not None:
            self.mm_projector: nn.Module = mm_projector
        elif out_hidden == token_dim:
            self.mm_projector = nn.Identity()
        else:
            self.mm_projector = nn.Linear(out_hidden, token_dim)

        # optional: load official Qwen3.5 pretrained weights (default: random init).
        if pretrained_path:
            self.load_pretrained(pretrained_path, strict=pretrained_strict)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        if self.gradient_checkpointing:
            checkpoint_kwargs = {"use_reentrant": False}
            self.language_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=checkpoint_kwargs
            )
            self.vision_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=checkpoint_kwargs
            )

    # ------------------------------------------------------------------
    # pretrained weight loading
    # ------------------------------------------------------------------
    def load_pretrained(self, path: str, strict: bool = False) -> None:
        """Load official Qwen3.5 weights into the vision + language sub-models.

        Accepts any of:
          * a single ``*.safetensors`` file,
          * a single ``*.pt`` / ``*.pth`` file (a state-dict, or ``{"state_dict": ...}``),
          * a directory holding an HF checkpoint (``model.safetensors.index.json`` +
            shards, or a bare ``*.safetensors`` / ``*.pt``).

        Keys are routed by the official prefixes ``model.language_model.*`` ->
        ``language_model`` and ``model.visual.*`` -> ``vision_model``; ``mtp.*`` is
        ignored (multi-token-prediction head, not part of the backbone), matching the
        official ``_keys_to_ignore_on_load_unexpected``.
        """
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

        lm_res = self.language_model.load_state_dict(lm_sd, strict=strict)
        vis_res = self.vision_model.load_state_dict(vis_sd, strict=strict)

        # tie the (missing) text embedding if the checkpoint only stored the tied copy
        if getattr(self.config.text_config, "tie_word_embeddings", False):
            pass  # TextModel exposes only embed_tokens (already loaded); nothing to tie.

        print(
            f"[Qwen35VLM] loaded pretrained from {path}: "
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


# ---------------------------------------------------------------------------
# weight file readers (.pt / .safetensors / HF sharded dir)
# ---------------------------------------------------------------------------
def _load_state_dict_any(path: str) -> Dict[str, "torch.Tensor"]:
    import os

    if os.path.isdir(path):
        return _load_from_dir(path)
    if path.endswith(".safetensors"):
        return _load_safetensors_file(path)
    # .pt / .pth / generic torch checkpoint
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        raise ValueError(f"Unsupported checkpoint object in {path}: {type(obj)}")
    return obj


def _load_safetensors_file(path: str) -> Dict[str, "torch.Tensor"]:
    try:
        from safetensors.torch import load_file
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Loading .safetensors requires the 'safetensors' package (pip install safetensors)."
        ) from e
    return load_file(path)


def _load_from_dir(path: str) -> Dict[str, "torch.Tensor"]:
    import glob
    import json
    import os

    index = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index) as f:
            weight_map = json.load(f)["weight_map"]
        shards = sorted(set(weight_map.values()))
        merged: Dict[str, Any] = {}
        for shard in shards:
            merged.update(_load_safetensors_file(os.path.join(path, shard)))
        return merged
    # fall back to any single safetensors / pt file in the dir
    for pattern in ("*.safetensors", "*.pt", "*.pth"):
        hits = sorted(glob.glob(os.path.join(path, pattern)))
        if hits:
            return _load_state_dict_any(hits[0])
    raise FileNotFoundError(f"No loadable weight file found under directory: {path}")
