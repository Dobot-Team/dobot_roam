"""PlannerHeads + build_planner_input (ai_foundation parity).

Mirrors ``model/arch/components/planner_heads.py``: ``build_planner_input``
merges the LLM output dict and the dataset input dict down to the keys the
trajectory head consumes (LLM output wins on conflicts); ``PlannerHeads`` is a
thin wrapper that runs the injected ``trajectory_pred_head``.
"""

from typing import Any, Dict

import torch
from torch import nn

from vla.omni.ported.projects.vln.constant.keys import PlannerDataKeys, PlannerModelKeys

# Keys the trajectory head reads (mirror PLANNER_HEADS_ALL_KEYS).
PLANNER_HEADS_ALL_KEYS = [
    PlannerModelKeys.LLM_LAST_HIDDEN_STATE,
    PlannerModelKeys.LLM_HIDDEN_STATES,
    PlannerModelKeys.TRAJECTORY_PROMPT_MATCH_MASK,
    PlannerDataKeys.FUTURE_TRAJECTORY,
    PlannerDataKeys.FUTURE_TRAJECTORY_MASK,
]


def build_planner_input(
    llm_output_dict: Dict[str, Any],
    input_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge LLM output + dataset input into the dict planner heads consume.

    Filters to PLANNER_HEADS_ALL_KEYS; on conflict the LLM output wins.
    """
    return {
        k: llm_output_dict[k] if k in llm_output_dict else input_dict[k]
        for k in PLANNER_HEADS_ALL_KEYS
        if k in llm_output_dict or k in input_dict
    }


class PlannerHeads(nn.Module):
    def __init__(self, trajectory_pred_head: nn.Module) -> None:
        super().__init__()
        self.trajectory_pred_head = trajectory_pred_head

    def forward(self, planner_input: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        return self.trajectory_pred_head(planner_input)
