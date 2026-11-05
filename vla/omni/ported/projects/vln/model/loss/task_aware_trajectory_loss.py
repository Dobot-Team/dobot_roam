"""Unweighted per-task metrics and separately weighted optimization."""
import torch
import torch.distributed as dist
from vla.omni.ported.projects.vln.constant.omni import TASK_NAMES
from vla.omni.ported.projects.vln.model.loss.trajectory_regression_loss import TrajectoryRegressionLoss


class TaskAwareTrajectoryRegressionLoss(TrajectoryRegressionLoss):
    def __init__(self, loss_weights=None, global_loss_scale=10.0, **kwargs):
        kwargs.pop("weight", None)
        super().__init__(weight=1.0, **kwargs)
        values = dict(loss_weights or {})
        if set(values) - set(TASK_NAMES):
            raise ValueError("unknown task loss weight")
        weights = torch.tensor([values.get(t, 1.0) for t in TASK_NAMES], dtype=torch.float32)
        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("task loss weights must be finite and nonnegative")
        if not torch.isfinite(torch.tensor(global_loss_scale)) or global_loss_scale <= 0:
            raise ValueError("global_loss_scale must be positive and finite")
        self.register_buffer("task_weights", weights)
        self.global_loss_scale = float(global_loss_scale)

    def denominators(self, batch):
        mask = batch[self.mask_key].float()
        task = batch["task_id"].long()
        if ((task < 0) | (task >= len(TASK_NAMES))).any():
            raise ValueError("invalid task_id")
        weights = mask * self._horizon_weight(mask.shape[1], mask.device)
        return torch.stack([weights[task == i].sum() for i in range(len(TASK_NAMES))])

    def forward(self, gt_dict, pred_dict):
        pred = pred_dict[self.pred_key].float()
        gt = gt_dict[self.gt_key].float()
        mask = gt_dict[self.mask_key].bool()
        # Avoid NaN gradients even when invalid padded targets are nonfinite.
        if not torch.isfinite(gt[mask]).all():
            raise ValueError("nonfinite valid trajectory target")
        gt = torch.where(mask.unsqueeze(-1), gt, torch.zeros_like(gt))
        scale = 1.0 if self.coord_scale is None else self.coord_scale.to(pred.device)
        if self.coord_scale is not None and scale.numel() != pred.shape[-1]:
            raise ValueError("coord_scale dimension mismatch")
        elem = self.loss_fn(pred / scale, gt / scale).mean(dim=-1)
        weights = mask.float() * self._horizon_weight(pred.shape[1], pred.device)
        task = gt_dict["task_id"].long()
        denominator = self.denominators(gt_dict)
        numerator = torch.stack([(elem * weights * (task == i).unsqueeze(-1)).sum()
                                 for i in range(len(TASK_NAMES))])
        counts = torch.stack([((task == i) & mask.any(dim=1)).sum()
                              for i in range(len(TASK_NAMES))]).to(pred.dtype)
        supplied = gt_dict.get("_task_loss_denominators")
        global_denominator = denominator.detach().clone() if supplied is None else supplied.to(pred.device)
        world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        if supplied is None and self.training and world > 1:
            dist.all_reduce(global_denominator)
        if self.training and global_denominator.sum() <= 0:
            raise ValueError("global batch has no valid trajectory supervision")
        contributions = torch.where(global_denominator > 0,
                                    numerator / global_denominator.clamp_min(1e-12),
                                    numerator * 0)
        optimization = self.global_loss_scale * (contributions * self.task_weights).sum()
        if self.training:
            optimization = optimization * world
        local_raw = numerator.detach() / denominator.clamp_min(1e-12)
        result = {"optimization_loss": optimization, "total_loss": local_raw.sum()}
        result["_loss_statistics"] = {
            f"loss/{name}/trajectory": torch.stack((numerator[i].detach(), denominator[i], counts[i])).detach()
            for i, name in enumerate(TASK_NAMES)
        }
        result.update({f"loss/{name}/trajectory": local_raw[i] for i, name in enumerate(TASK_NAMES)})
        return result
