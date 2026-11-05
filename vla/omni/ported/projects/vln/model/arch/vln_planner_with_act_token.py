"""ACT-query planner with configurable learnable queries and output scaling.

The default single query preserves the OmTrack MLP/checkpoint contract. Multiple
queries are appended after the multimodal prompt and their final LLM states are
passed to the trajectory head without pooling. Queries retain the VLM's causal
attention: each sees the full prompt and preceding queries. ``freeze_patterns``
controls backbone training; the query parameters are always created trainable.
"""

from typing import Any, Dict, Sequence, Tuple

import torch
import torch.nn as nn

from vla.omni.ported.projects.vln.constant.keys import PlannerModelKeys
from vla.omni.ported.projects.vln.model.arch.vln_planner import VLNPlanner
from vla.omni.ported.projects.vln.model.arch.vln_planner_with_trajectory_token import (
    TRAJECTORY_QUERY_TOKEN_MARKER_ID,
)

# Reuse the trajectory-query marker id to tag the appended ACT-token position in
# mm_merged_prompt_ids. It never goes through the token embedding table.
ACT_TOKEN_MARKER_ID = TRAJECTORY_QUERY_TOKEN_MARKER_ID


class VLNPlannerWithActToken(VLNPlanner):
    def __init__(
        self,
        output_scale: Sequence[float] = (2.0, 2.0, 1.0),
        **kwargs: Any,
    ) -> None:
        super().__init__(output_scale=output_scale, **kwargs)

    def _build_modules(
        self,
        output_scale: Sequence[float] = (2.0, 2.0, 1.0),
        num_action_queries: int = 1,
        **kwargs: Any,
    ) -> None:
        if not isinstance(num_action_queries, int) or num_action_queries < 1:
            raise ValueError("num_action_queries must be a positive integer")
        self.num_action_queries = num_action_queries
        # super() builds components, loads checkpoints and applies
        # freeze_patterns; the ACT token created afterwards stays trainable.
        super()._build_modules(**kwargs)
        self.act_token = nn.Parameter(
            torch.randn(1, self.num_action_queries, self.token_dim, dtype=torch.float32) * 0.02
        )
        scale = torch.tensor(list(output_scale), dtype=torch.float32).view(1, 1, -1)
        coord_dim = getattr(
            self.planner_heads.trajectory_pred_head, "coord_dim", scale.numel()
        )
        if scale.numel() != coord_dim:
            raise ValueError(
                f"output_scale length ({scale.numel()}) must match "
                f"trajectory_pred_head.coord_dim ({coord_dim})"
            )
        head = self.planner_heads.trajectory_pred_head
        expected_queries = getattr(head, "num_condition_tokens", self.num_action_queries)
        if expected_queries != self.num_action_queries:
            raise ValueError("num_action_queries must match head.num_condition_tokens")
        target_scale = getattr(head, "target_scale", None)
        if target_scale is not None and not torch.equal(target_scale.cpu(), scale):
            raise ValueError("head.target_scale must match planner.output_scale")
        self.register_buffer("output_scale", scale)

    def _build_special_token_ids(self) -> None:
        super()._build_special_token_ids()
        self.act_token_marker_id = ACT_TOKEN_MARKER_ID

    def _merge_mm_embeddings(
        self,
        prompt_embeds: torch.Tensor,
        attn_mask: torch.Tensor,
        prompt_ids: torch.Tensor,
        image_embeds: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prompt_embeds, attn_mask, prompt_ids = super()._merge_mm_embeddings(
            prompt_embeds, attn_mask, prompt_ids, image_embeds
        )
        batch_size = prompt_embeds.size(0)
        act = self.act_token.expand(batch_size, -1, -1).to(prompt_embeds.dtype)
        prompt_embeds = torch.cat([prompt_embeds, act], dim=1)
        prompt_ids = torch.cat(
            [
                prompt_ids,
                torch.full(
                    act.shape[:2],
                    self.act_token_marker_id,
                    dtype=prompt_ids.dtype,
                    device=prompt_ids.device,
                ),
            ],
            dim=1,
        )
        attn_mask = torch.cat(
            [
                attn_mask,
                torch.ones(act.shape[:2], dtype=attn_mask.dtype, device=attn_mask.device),
            ],
            dim=1,
        )
        return prompt_embeds, attn_mask, prompt_ids

    def _select_trajectory_hidden(
        self, hidden_states: torch.Tensor, mm_merged_prompt_ids: torch.Tensor
    ) -> torch.Tensor:
        # Keep every appended query as a separate condition token.
        return hidden_states[:, -self.num_action_queries:]

    def _trajectory_token_mask(self, mm_merged_prompt_ids: torch.Tensor) -> torch.Tensor:
        # the ACT-token position was tagged on append with an internal marker id
        return mm_merged_prompt_ids == self.act_token_marker_id

    def forward(self, input_dict: Dict, with_head: bool = True) -> Dict:
        output = super().forward(input_dict, with_head=with_head)
        if PlannerModelKeys.PRED_FUTURE_TRAJECTORY in output:
            output[PlannerModelKeys.PRED_FUTURE_TRAJECTORY] = (
                output[PlannerModelKeys.PRED_FUTURE_TRAJECTORY] * self.output_scale
            )
        return output
