"""VLNPlannerWithTrajectoryToken (ai_foundation ``E2EPlannerV1WithTrajectoryToken`` parity).

Adds learnable *trajectory query tokens* to the LLM prompt: a
``[1, num_waypoints, token_dim]`` learnable parameter is appended to the merged
multimodal embeddings, and the LLM hidden states at those appended positions are
handed to the trajectory head. This is the mechanism the e2e planner uses to turn
an autoregressive VLM into a trajectory predictor.
"""

from typing import Any, Tuple

import torch
import torch.nn as nn

from vla.omni.ported.projects.vln.constant.keys import PlannerModelKeys
from vla.omni.ported.projects.vln.model.arch.vln_planner import VLNPlanner

# Internal marker used only in mm_merged_prompt_ids to tag appended trajectory-query
# positions. It is never passed through the token embedding table, so keep it out
# of the normal tokenizer vocabulary to avoid confusing it with a real token id.
TRAJECTORY_QUERY_TOKEN_MARKER_ID = -10001


class VLNPlannerWithTrajectoryToken(VLNPlanner):
    def __init__(self, num_waypoints: int, **kwargs: Any) -> None:
        super().__init__(num_waypoints=num_waypoints, **kwargs)

    def _build_modules(self, num_waypoints: int, **kwargs: Any) -> None:
        super()._build_modules(**kwargs)
        self.num_traj_waypoints = num_waypoints
        self.trajectory_embed = nn.Parameter(
            torch.randn(1, num_waypoints, self.token_dim, dtype=torch.float32) * 0.02
        )
        head_wp = getattr(self.planner_heads.trajectory_pred_head, "num_waypoints", None)
        if head_wp is not None:
            assert head_wp == num_waypoints, (
                f"num_waypoints mismatch: arch appends {num_waypoints} trajectory tokens but "
                f"trajectory_pred_head expects {head_wp}. Keep MODEL.num_waypoints == "
                f"trajectory_pred_head.num_waypoints."
            )

    def _build_special_token_ids(self) -> None:
        super()._build_special_token_ids()
        self.trajectory_query_token_marker_id = TRAJECTORY_QUERY_TOKEN_MARKER_ID

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
        B = prompt_embeds.size(0)
        traj = self.trajectory_embed.expand(B, -1, -1).to(prompt_embeds.dtype)
        prompt_embeds = torch.cat([prompt_embeds, traj], dim=1)
        prompt_ids = torch.cat(
            [
                prompt_ids,
                torch.full(
                    traj.shape[:2], self.trajectory_query_token_marker_id,
                    dtype=prompt_ids.dtype, device=prompt_ids.device,
                ),
            ],
            dim=1,
        )
        attn_mask = torch.cat(
            [attn_mask, torch.ones(traj.shape[:2], dtype=attn_mask.dtype, device=attn_mask.device)],
            dim=1,
        )
        return prompt_embeds, attn_mask, prompt_ids

    def _select_trajectory_hidden(
        self, hidden_states: torch.Tensor, mm_merged_prompt_ids: torch.Tensor
    ) -> torch.Tensor:
        # the appended trajectory tokens are the trailing ``num_traj_waypoints`` states
        return hidden_states[:, -self.num_traj_waypoints :]

    def _trajectory_token_mask(self, mm_merged_prompt_ids: torch.Tensor) -> torch.Tensor:
        # trajectory query positions were tagged on append with an internal marker id
        return mm_merged_prompt_ids == self.trajectory_query_token_marker_id
