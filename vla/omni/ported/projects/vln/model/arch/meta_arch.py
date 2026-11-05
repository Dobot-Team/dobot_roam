"""VLNMetaArch: base class for VLN model architectures (ai_foundation style).

Mirrors ``E2EMetaArch``:
  * subclasses implement ``forward(batch) -> pred_dict``;
  * ``get_loss_dict(gt_dict, pred_dict)`` iterates ``self.losses`` and produces
    a flat loss dict whose ``total_loss`` entry drives back-propagation;
  * ``load_checkpoint`` handles the ``{'state_dict': ...}`` wrapper.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from vla.omni.ported.projects.vln.constant.keys import ModelKeys
from vla.omni.ported.projects.vln.model.loss.base_loss import BaseLoss

logger = logging.getLogger(__name__)


class VLNMetaArch(nn.Module, ABC):
    def __init__(self, losses: Optional[List[BaseLoss]] = None) -> None:
        super().__init__()
        self.losses = nn.ModuleList(losses) if losses else None

    @abstractmethod
    def forward(self, batch: Dict) -> Dict:
        raise NotImplementedError

    def get_loss_dict(self, gt_dict: Dict, pred_dict: Dict) -> Dict[str, torch.Tensor]:
        if self.losses is None:
            raise ValueError("No loss is defined for the model.")
        loss_dict: Dict[str, torch.Tensor] = {}
        for loss_module in self.losses:
            loss_dict.update(loss_module(gt_dict, pred_dict))
        total = sum(v for v in loss_dict.values() if isinstance(v, torch.Tensor))
        loss_dict[ModelKeys.TOTAL_LOSS] = total
        return loss_dict

    def load_checkpoint(self, ckpt_path: str, strict: bool = False):
        # Full trainer checkpoints also contain optimizer/RNG state (including
        # NumPy objects), so PyTorch >= 2.6's weights_only=True default cannot
        # deserialize them. Checkpoints passed here are explicit, trusted local
        # training artifacts; match the trainer's full-checkpoint load path.
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        result = self.load_state_dict(state, strict=strict)
        logger.info("load_checkpoint missing=%s unexpected=%s",
                    result.missing_keys, result.unexpected_keys)
        return result

    @staticmethod
    def freeze(module: nn.Module) -> None:
        for param in module.parameters():
            param.requires_grad = False
