"""BaseLoss: loss base class (ai_foundation ``model/loss/base_loss.py``).

``forward`` prefixes each loss name with ``log_prefix/`` so the meta-arch can
merge multiple loss modules into a flat loss dict.
"""

from abc import ABC, abstractmethod
from typing import Dict

import torch


class BaseLoss(torch.nn.Module, ABC):
    def __init__(self, log_prefix: str = "loss", weight: float = 1.0) -> None:
        super().__init__()
        self.log_prefix = log_prefix
        self.weight = weight

    @abstractmethod
    def _forward_loss(self, gt_dict: Dict, pred_dict: Dict) -> Dict[str, torch.Tensor]:
        ...

    def forward(self, gt_dict: Dict, pred_dict: Dict) -> Dict[str, torch.Tensor]:
        loss_dict = self._forward_loss(gt_dict, pred_dict)
        return {f"{self.log_prefix}/{name}": self.weight * loss for name, loss in loss_dict.items()}
