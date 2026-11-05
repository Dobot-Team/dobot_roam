"""VLNPlanner - e2e-style planner base framework (ai_foundation ``E2EPlannerV1`` parity).

Organizes the whole model from a config-built ``vision_language_model`` (VLM) +
``trajectory_pred_head`` + ``losses``, decomposed into the same components as
ai_foundation: VisionEncoder + (optional) TokenCompressor + LLMDecoder +
PlannerHeads.

Forward flow (identical in structure to E2EPlannerV1.forward):
    prompt_ids, attn_mask = _get_prompt_ids(input_dict)
    inputs_embeds, attn_mask, merged_ids = _prepare_embeds(...)     # vision + text + merge
    llm_output = llm_decoder({inputs_embeds, attn_mask})
    planner_input = build_planner_input(llm_output, input_dict)
    llm_output.update(planner_heads(planner_input))
"""

from typing import Any, Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn

from vla.omni.ported.projects.vln.constant.keys import PlannerDataKeys, PlannerModelKeys
from vla.omni.ported.projects.vln.model.arch.components import (
    LLMDecoder,
    PlannerHeads,
    TokenCompressor,
    VisionEncoder,
    build_planner_input,
)
from vla.omni.ported.projects.vln.model.arch.components.image_normalize_and_cast import ImageNormalizeAndCast
from vla.omni.ported.projects.vln.model.arch.components.llm_decoder import ATTN_MASK, INPUTS_EMBEDS, POSITION_IDS
from vla.omni.ported.projects.vln.model.arch.meta_arch import VLNMetaArch
from vla.omni.ported.projects.vln.model.loss.base_loss import BaseLoss

IMAGE_TOKEN_ID_DEFAULT = 3


class VLNPlanner(VLNMetaArch):
    def __init__(self, **kwargs: Any) -> None:
        losses = kwargs.pop("losses", None)
        super().__init__(losses=losses)
        self._build_modules(**kwargs)

    # ---- component class hooks (override in subclasses) ----
    def _get_vision_encoder_cls(self) -> Type[nn.Module]:
        return VisionEncoder

    def _get_llm_decoder_cls(self) -> Type[nn.Module]:
        return LLMDecoder

    # ---- module construction ----
    def _build_modules(
        self,
        vision_language_model: nn.Module,
        trajectory_pred_head: nn.Module,
        tokenizer: Optional[Any] = None,
        vision_token_compressor: Optional[nn.Module] = None,
        use_uint8_input_images: bool = False,
        normalized_data_type: str = "fp32",
        normalize_images: bool = True,
        image_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        image_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        image_token_id: int = IMAGE_TOKEN_ID_DEFAULT,
        video_token_id: Optional[int] = None,
        freeze_patterns: Optional[List[str]] = None,
        alignment_pretrained_weights: Optional[str] = None,
        ckpt_path: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.tokenizer = tokenizer
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.freeze_patterns = freeze_patterns

        vision_encoder_cls = self._get_vision_encoder_cls()
        llm_decoder_cls = self._get_llm_decoder_cls()

        # decompose the VLM (mirror TuringVL.get_vision_model / get_language_model)
        vision_model = vision_language_model.get_vision_model()
        self.vision_encoder = vision_encoder_cls(
            vision_model=vision_model,
            mm_projector=vision_language_model.get_mm_projector(),
            image_normalize_and_cast=ImageNormalizeAndCast(
                use_uint8_input_images=use_uint8_input_images,
                normalized_data_type=normalized_data_type,
                normalize=normalize_images,
                image_mean=image_mean,
                image_std=image_std,
            ),
        )
        if vision_token_compressor is not None:
            self.token_compressor: Optional[TokenCompressor] = TokenCompressor(
                vision_token_compressor=vision_token_compressor
            )
        else:
            self.token_compressor = None

        language_model = vision_language_model.get_language_model()
        self.llm_decoder = llm_decoder_cls(language_model)

        # token_dim inferred from the LLM config (mirror e2e)
        self.token_dim = self.llm_decoder.language_model.config.hidden_size

        resolve_input_channels = getattr(trajectory_pred_head, "resolve_input_channels", None)
        if callable(resolve_input_channels):
            resolve_input_channels(self.token_dim)
        self.planner_heads = PlannerHeads(trajectory_pred_head=trajectory_pred_head)
        self.num_waypoints = getattr(trajectory_pred_head, "num_waypoints", None)

        self._build_special_token_ids()

        if alignment_pretrained_weights:
            self.load_checkpoint(alignment_pretrained_weights, strict=False)
        if ckpt_path:
            self.load_checkpoint(ckpt_path, strict=False)
        if freeze_patterns:
            self._freeze_modules(freeze_patterns)

    def _build_special_token_ids(self) -> None:
        """Hook for subclasses to resolve special token ids from the tokenizer."""
        return None

    def _freeze_modules(self, patterns: List[str]) -> None:
        for name, param in self.named_parameters():
            if any(p in name for p in patterns):
                param.requires_grad = False

    # ---- embedding / merge ----
    def _forward_llm_embedding(self, prompt_ids: torch.Tensor) -> torch.Tensor:
        return self.llm_decoder.language_model.embed_tokens(prompt_ids)

    def _get_prompt_ids(self, input_dict: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        prompt_ids = input_dict[PlannerDataKeys.PROMPT_IDS]
        attn_mask = input_dict.get(PlannerDataKeys.ATTENTION_MASK)
        if attn_mask is None:
            attn_mask = torch.ones_like(prompt_ids)
        return prompt_ids, attn_mask

    def _merge_mm_embeddings(
        self,
        prompt_embeds: torch.Tensor,
        attn_mask: torch.Tensor,
        prompt_ids: torch.Tensor,
        image_embeds: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Scatter visual tokens into IMAGE-token positions with strict shape checks."""
        B = prompt_embeds.size(0)
        expected = image_embeds.size(1)
        vision_mask = prompt_ids == self.image_token_id  # [B, L]
        if self.video_token_id is not None:
            vision_mask = vision_mask | (prompt_ids == self.video_token_id)
        for b in range(B):
            pos = vision_mask[b].nonzero(as_tuple=True)[0]
            actual = pos.numel()
            if actual != expected:
                raise ValueError(
                    f"vision token / visual token count mismatch for sample {b}: "
                    f"prompt has {actual} visual-token positions "
                    f"(image_token_id={self.image_token_id}, video_token_id={self.video_token_id}), "
                    f"but vision encoder produced {expected} tokens. Check img_size/patch_size/"
                    f"spatial_merge_size/temporal_patch_size and the prompt processor's "
                    f"num_vision_tokens."
                )
            if expected > 0:
                prompt_embeds[b, pos] = image_embeds[b].to(prompt_embeds.dtype)
        return prompt_embeds, attn_mask, prompt_ids

    @staticmethod
    def _build_padding_aware_position_ids(attn_mask: torch.Tensor) -> torch.Tensor:
        """Build position ids where real tokens start at 0 even under left padding."""
        position_ids = attn_mask.long().cumsum(dim=1) - 1
        return position_ids.clamp_(min=0)

    def _video_llm_grid(self, input_dict: Dict, batch_idx: int) -> Tuple[int, int, int]:
        """Return the merged Qwen video grid used by text-side MRoPE.

        The vision tower consumes pre-merge ``grid_thw=[grid_t, gh, gw]``. The
        language tower receives one placeholder per merged visual token, so its
        H/W axes are ``gh/spatial_merge_size`` and ``gw/spatial_merge_size``.
        """
        rgb = input_dict[PlannerDataKeys.RGB]
        _, T, H, W, _ = rgb.shape
        adapter = self.vision_encoder.vision_model
        patch_size = int(getattr(adapter, "patch_size", 16))
        merge_size = int(getattr(adapter, "merge_size", 2))
        temporal_patch_size = int(getattr(adapter, "temporal_patch_size", 2))
        grid_t = (int(T) + temporal_patch_size - 1) // temporal_patch_size
        gh = int(H) // patch_size
        gw = int(W) // patch_size
        return grid_t, gh // merge_size, gw // merge_size

    def _build_qwen_video_position_ids(
        self,
        attn_mask: torch.Tensor,
        prompt_ids: torch.Tensor,
        input_dict: Dict,
    ) -> torch.Tensor:
        """Build Qwen-style T/H/W MRoPE ids for prompts containing video pads.

        Text-like tokens advance a scalar cursor on all three axes. A contiguous
        video-pad span advances on three axes using the video grid, matching the
        official Qwen VL convention that ``<|video_pad|>`` positions represent
        merged video tokens rather than ordinary text positions.
        """
        base = self._build_padding_aware_position_ids(attn_mask)
        if self.video_token_id is None or not (prompt_ids == self.video_token_id).any():
            return base

        B, L = prompt_ids.shape
        pos_ids = torch.zeros((3, B, L), dtype=torch.long, device=prompt_ids.device)
        for b in range(B):
            real_pos = attn_mask[b].nonzero(as_tuple=True)[0]
            cursor = 0
            i = 0
            while i < real_pos.numel():
                abs_pos = real_pos[i]
                token_id = int(prompt_ids[b, abs_pos].item())
                if token_id == self.video_token_id:
                    start = i
                    while i < real_pos.numel() and int(prompt_ids[b, real_pos[i]].item()) == self.video_token_id:
                        i += 1
                    span = real_pos[start:i]
                    full_grid_t, grid_h, grid_w = self._video_llm_grid(input_dict, b)
                    frame_expected = grid_h * grid_w
                    full_expected = full_grid_t * frame_expected
                    actual = int(span.numel())
                    if actual == frame_expected:
                        # Official Qwen3.5 video prompts are timestamp-separated:
                        # video_grid_thw=[T,H,W] is repeated into T independent
                        # [1,H,W] grids for language-side MRoPE.
                        grid_t = 1
                    elif actual == full_expected:
                        # Backward-compatible path for a single contiguous video block.
                        grid_t = full_grid_t
                    else:
                        raise ValueError(
                            f"video token / MRoPE grid mismatch for sample {b}: "
                            f"prompt has {actual} video-token positions, but expected either "
                            f"one frame-grid {frame_expected} or full video-grid {full_expected} "
                            f"(grid_t={full_grid_t}, grid_h={grid_h}, grid_w={grid_w})."
                        )
                    t_idx = torch.arange(grid_t, device=prompt_ids.device).view(grid_t, 1, 1)
                    h_idx = torch.arange(grid_h, device=prompt_ids.device).view(1, grid_h, 1)
                    w_idx = torch.arange(grid_w, device=prompt_ids.device).view(1, 1, grid_w)
                    pos_ids[0, b, span] = (t_idx.expand(grid_t, grid_h, grid_w).reshape(-1) + cursor)
                    pos_ids[1, b, span] = (h_idx.expand(grid_t, grid_h, grid_w).reshape(-1) + cursor)
                    pos_ids[2, b, span] = (w_idx.expand(grid_t, grid_h, grid_w).reshape(-1) + cursor)
                    cursor += max(grid_t, grid_h, grid_w)
                else:
                    pos_ids[:, b, abs_pos] = cursor
                    cursor += 1
                    i += 1
        return pos_ids

    def _build_position_ids(
        self,
        attn_mask: torch.Tensor,
        prompt_ids: torch.Tensor,
        input_dict: Dict,
    ) -> torch.Tensor:
        return self._build_qwen_video_position_ids(attn_mask, prompt_ids, input_dict)

    def _prepare_embeds(
        self,
        input_dict: Dict,
        prompt_ids: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_embeds = self.vision_encoder({PlannerDataKeys.RGB: input_dict[PlannerDataKeys.RGB]})
        if self.token_compressor is not None:
            image_embeds = self.token_compressor(image_embeds)
        prompt_embeds = self._forward_llm_embedding(prompt_ids)
        return self._merge_mm_embeddings(prompt_embeds, attn_mask, prompt_ids, image_embeds)

    def _select_trajectory_hidden(
        self, hidden_states: torch.Tensor, mm_merged_prompt_ids: torch.Tensor
    ) -> torch.Tensor:
        """Base planner has no traj tokens -> use the last ``num_waypoints`` states."""
        w = self.num_waypoints or hidden_states.size(1)
        return hidden_states[:, -w:]

    def _trajectory_token_mask(self, mm_merged_prompt_ids: torch.Tensor) -> torch.Tensor:
        """Boolean mask marking trajectory-token positions in the merged prompt.

        Base planner tags the trailing ``num_waypoints`` positions; subclasses that
        append real trajectory tokens override this (e.g. match a special id).
        Mirrors ai_foundation's ``TRAJECTORY_PROMPT_MATCH_MASK`` contract.
        """
        mask = torch.zeros_like(mm_merged_prompt_ids, dtype=torch.bool)
        w = self.num_waypoints or mm_merged_prompt_ids.size(1)
        mask[:, -w:] = True
        return mask

    def forward(self, input_dict: Dict, with_head: bool = True) -> Dict:
        prompt_ids, attn_mask = self._get_prompt_ids(input_dict)
        inputs_embeds, attn_mask, mm_merged_prompt_ids = self._prepare_embeds(
            input_dict, prompt_ids, attn_mask
        )
        position_ids = self._build_position_ids(attn_mask, mm_merged_prompt_ids, input_dict)
        llm_output: Dict[str, Any] = self.llm_decoder(
            {INPUTS_EMBEDS: inputs_embeds, ATTN_MASK: attn_mask, POSITION_IDS: position_ids}
        )
        hidden_states = llm_output[PlannerModelKeys.LLM_HIDDEN_STATES]
        llm_output[PlannerModelKeys.LLM_LAST_HIDDEN_STATE] = self._select_trajectory_hidden(
            hidden_states, mm_merged_prompt_ids
        )
        llm_output[PlannerModelKeys.MM_MERGED_PROMPT_IDS] = mm_merged_prompt_ids
        llm_output[PlannerModelKeys.TRAJECTORY_PROMPT_MATCH_MASK] = self._trajectory_token_mask(
            mm_merged_prompt_ids
        )
        if not with_head:
            return llm_output
        planner_input = build_planner_input(llm_output, input_dict)
        llm_output.update(self.planner_heads(planner_input))
        return llm_output

    @torch.no_grad()
    def generate_trajectories(self, input_dict: Dict) -> torch.Tensor:
        """Inference: run the planner and return predicted future waypoints.

        Mirrors the role of ai_foundation ``E2EPlannerV1.generate_trajectories``.
        """
        self.eval()
        out = self.forward(input_dict, with_head=True)
        return out[PlannerModelKeys.PRED_FUTURE_TRAJECTORY]
