"""build_lr_scheduler: config-driven LR scheduler factory (ai_foundation style)."""

import math
from typing import Any, Dict, List, Optional, Union

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class OneCycleLREarlyStop(torch.optim.lr_scheduler.OneCycleLR):
    """OneCycleLR with a separate annealing horizon and actual stop step.

    ``annealing_effective_total_steps`` controls the full OneCycle curve shape.
    ``total_steps`` controls how many scheduler steps are actually executed.
    Once the actual step budget is reached, additional ``step()`` calls become
    no-ops, so training can consume only the early portion of a longer LR curve.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        max_lr: Union[float, List[float]],
        total_steps: Optional[int] = None,
        annealing_effective_total_steps: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        effective_total_steps = annealing_effective_total_steps or total_steps
        if effective_total_steps is None:
            raise ValueError(
                "OneCycleLREarlyStop requires total_steps or annealing_effective_total_steps."
            )
        self.annealing_effective_total_steps = int(effective_total_steps)
        self.actual_total_steps = None if total_steps is None else int(total_steps)
        super().__init__(
            optimizer=optimizer,
            max_lr=max_lr,
            total_steps=self.annealing_effective_total_steps,
            **kwargs,
        )

    def step(self, epoch: Optional[int] = None) -> None:
        if self.actual_total_steps is not None and self.last_epoch >= self.actual_total_steps:
            return
        super().step(epoch)


def build_lr_scheduler(cfg: Dict[str, Any], optimizer: Optimizer) -> LRScheduler:
    cfg = dict(cfg)
    sched_type = cfg.pop("type")
    if sched_type == "CosineLR":
        t_max = cfg.pop("total_steps", cfg.pop("t_max", 1000))
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, **cfg)
    if sched_type == "CosineWithMinLR":
        t_max = cfg.pop("total_steps", cfg.pop("t_max", 1000))
        min_lr_rate = float(cfg.pop("min_lr_rate", 0.0))
        if min_lr_rate < 0.0:
            raise ValueError("CosineWithMinLR min_lr_rate must be non-negative")
        base_lrs = {float(group["lr"]) for group in optimizer.param_groups}
        if len(base_lrs) != 1:
            raise ValueError(
                "CosineWithMinLR requires all optimizer groups to share one base lr"
            )
        eta_min = next(iter(base_lrs)) * min_lr_rate
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=eta_min,
            **cfg,
        )
    if sched_type == "WarmupCosineWithMinLR":
        total_steps = int(cfg.pop("total_steps"))
        warmup_steps = cfg.pop("warmup_steps", None)
        warmup_ratio = float(cfg.pop("warmup_ratio", 0.0))
        min_lr_rate = float(cfg.pop("min_lr_rate", 0.0))
        if cfg:
            raise TypeError(
                f"Unexpected WarmupCosineWithMinLR arguments: {sorted(cfg)}"
            )
        if total_steps <= 0:
            raise ValueError("WarmupCosineWithMinLR total_steps must be positive")
        if warmup_steps is None:
            warmup_steps = math.ceil(total_steps * warmup_ratio)
        warmup_steps = int(warmup_steps)
        if not 0 <= warmup_steps <= total_steps:
            raise ValueError("warmup_steps must be in [0, total_steps]")
        if not 0.0 <= min_lr_rate <= 1.0:
            raise ValueError("min_lr_rate must be in [0, 1]")

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            progress = min(progress, 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_rate + (1.0 - min_lr_rate) * cosine

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    if sched_type == "OneCycleLREarlyStop":
        return OneCycleLREarlyStop(optimizer, **cfg)
    if sched_type == "OneCycleLR":
        return torch.optim.lr_scheduler.OneCycleLR(optimizer, **cfg)
    if sched_type == "LinearLR":
        return torch.optim.lr_scheduler.LinearLR(optimizer, **cfg)
    if sched_type == "StepLR":
        return torch.optim.lr_scheduler.StepLR(optimizer, **cfg)
    if sched_type == "ConstantLR":
        return torch.optim.lr_scheduler.ConstantLR(optimizer, **cfg)
    raise KeyError(f"Unknown lr_scheduler type '{sched_type}'")
