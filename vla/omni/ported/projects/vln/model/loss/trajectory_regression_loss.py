"""TrajectoryRegressionLoss - config-driven waypoint regression with horizon decay.

Dependency-light analogue of ai_foundation's ``PytorchLossWrapperWithDecay``:
reads GT from ``gt_dict[gt_key]`` and prediction from ``pred_dict[pred_key]``,
applies a configurable point loss (smooth_l1 / mse / l1) and an optional per-
waypoint horizon decay that down-weights farther-future waypoints
(``weight = decay_factor ** max(0, t - decay_start_step)``).
"""

from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F

from vla.omni.ported.projects.vln.constant.keys import PlannerDataKeys, PlannerModelKeys
from vla.omni.ported.projects.vln.model.loss.base_loss import BaseLoss

_LOSS_FNS = {
    "smooth_l1": lambda p, t: F.smooth_l1_loss(p, t, reduction="none"),
    "mse": lambda p, t: F.mse_loss(p, t, reduction="none"),
    "l1": lambda p, t: F.l1_loss(p, t, reduction="none"),
}


class TrajectoryRegressionLoss(BaseLoss):
    def __init__(
        self,
        log_prefix: str = "Trajectory",
        weight: float = 1.0,
        gt_key: str = PlannerDataKeys.FUTURE_TRAJECTORY,
        pred_key: str = PlannerModelKeys.PRED_FUTURE_TRAJECTORY,
        mask_key: str = PlannerDataKeys.FUTURE_TRAJECTORY_MASK,
        loss_fn: str = "smooth_l1",
        decay_start_step: int = 0,
        decay_factor: float = 1.0,
        coord_scale: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__(log_prefix=log_prefix, weight=weight)
        self.gt_key = gt_key
        self.pred_key = pred_key
        self.mask_key = mask_key
        assert loss_fn in _LOSS_FNS, f"unknown loss_fn '{loss_fn}'"
        self.loss_fn = _LOSS_FNS[loss_fn]
        self.decay_start_step = decay_start_step
        self.decay_factor = decay_factor
        scale = None if coord_scale is None else torch.tensor(coord_scale, dtype=torch.float32)
        self.register_buffer("coord_scale", scale)

    def _horizon_weight(self, num_waypoints: int, device: torch.device) -> torch.Tensor:
        t = torch.arange(num_waypoints, device=device)
        exp = torch.clamp(t - self.decay_start_step, min=0).float()
        return self.decay_factor ** exp  # [num_waypoints]

    def _forward_loss(self, gt_dict: Dict, pred_dict: Dict) -> Dict[str, torch.Tensor]:
        gt = gt_dict[self.gt_key]                     # [B, W, C]
        pred = pred_dict[self.pred_key]               # [B, W, C]
        pred_for_loss = pred.float()
        gt_for_loss = gt.float()
        if self.coord_scale is not None:
            if self.coord_scale.numel() != pred.size(-1):
                raise ValueError(
                    f"coord_scale has {self.coord_scale.numel()} values for {pred.size(-1)} coordinates"
                )
            scale = self.coord_scale.to(device=pred.device).clamp_min(1e-6)
            pred_for_loss = pred_for_loss / scale
            gt_for_loss = gt_for_loss / scale
        per_elem = self.loss_fn(pred_for_loss, gt_for_loss)  # [B, W, C]
        per_wp = per_elem.mean(dim=-1)                # [B, W]

        hw = self._horizon_weight(per_wp.size(1), per_wp.device).unsqueeze(0)  # [1, W]
        weights = hw
        mask: Optional[torch.Tensor] = gt_dict.get(self.mask_key)
        if mask is not None:
            weights = weights * mask.float()

        denom = weights.sum().clamp(min=1e-6)
        loss = (per_wp * weights).sum() / denom
        return {"waypoint": loss}
