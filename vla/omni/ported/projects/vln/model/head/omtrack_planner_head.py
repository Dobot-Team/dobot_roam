"""ACT-token trajectory head used by OmTrackVLA."""

from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn

from vla.omni.ported.projects.vln.constant.keys import PlannerModelKeys
from vla.omni.ported.projects.vln.model.head.base_head import BaseHead


class OmTrackPlannerHead(BaseHead):
    def __init__(
        self,
        input_channels: Optional[int] = 1024,
        hidden_channels: Sequence[int] = (2048, 2048),
        num_waypoints: int = 8,
        coord_dim: int = 3,
        use_tanh: bool = False,
        hidden_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.input_channels: Optional[int] = None
        self.hidden_channels = tuple(int(channel) for channel in hidden_channels)
        self.mlp: Optional[nn.Sequential] = None
        self.num_waypoints = int(num_waypoints)
        self.coord_dim = int(coord_dim)
        self.use_tanh = bool(use_tanh)
        self.hidden_layer_norm = bool(hidden_layer_norm)
        if input_channels is not None:
            self.resolve_input_channels(input_channels)

        # Keep the original Linear indices stable while allowing checkpoints
        # trained before the optional hidden LayerNorms to initialize them.
        self.checkpoint_optional_prefixes = (
            tuple(f"mlp.{2 * index + 2}.0" for index in range(len(self.hidden_channels)))
            if self.hidden_layer_norm
            else ()
        )

    def resolve_input_channels(self, input_channels: int) -> None:
        """Build a deferred head from the VLM hidden size.

        Explicit historical ``input_channels`` values still build immediately.
        Passing ``null`` in a new config defers construction until
        :class:`VLNPlanner` has discovered the selected language model width.
        """
        if self.mlp is not None:
            return
        input_channels = int(input_channels)
        if input_channels <= 0:
            raise ValueError(f"input_channels must be positive, got {input_channels}")
        self.input_channels = input_channels
        dims = [input_channels, *self.hidden_channels]
        layers = [nn.LayerNorm(input_channels)]
        for input_dim, output_dim in zip(dims[:-1], dims[1:]):
            activation = nn.GELU()
            if self.hidden_layer_norm:
                activation = nn.Sequential(nn.LayerNorm(output_dim), activation)
            layers.extend((nn.Linear(input_dim, output_dim), activation))
        layers.append(nn.Linear(dims[-1], self.num_waypoints * self.coord_dim))
        self.mlp = nn.Sequential(*layers)

    def select_input(self, planner_input: Dict[str, Any]) -> torch.Tensor:
        hidden = planner_input[PlannerModelKeys.LLM_LAST_HIDDEN_STATE]
        return hidden[:, -1] if hidden.ndim == 3 else hidden

    def forward(self, planner_input: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        if self.mlp is None:
            raise RuntimeError(
                "OmTrackPlannerHead input_channels is unresolved; build it through "
                "VLNPlanner or call resolve_input_channels() first"
            )
        trajectory = self.mlp(self.select_input(planner_input).float())
        if self.use_tanh:
            trajectory = torch.tanh(trajectory)
        trajectory = trajectory.view(-1, self.num_waypoints, self.coord_dim)
        return {PlannerModelKeys.PRED_FUTURE_TRAJECTORY: trajectory}
