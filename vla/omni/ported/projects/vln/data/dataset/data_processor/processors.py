"""Example data processors (registry-configurable, ordered per dataset).

Processors run on a single sample's *observation dict* (keys: rgb/depth/
instruction/progress) and mutate it in place.
"""

from typing import Any, Dict

import torch

from vla.omni.ported.projects.vln.constant.keys import DataKeys
from vla.omni.ported.projects.vln.data.dataset.data_processor.base_data_processor import BaseDataProcessor


class RGBNormalizeProcessor(BaseDataProcessor):
    """Scale RGB from [0,255] to [0,1] (guarded by a max check)."""

    def __init__(self, scale: float = 255.0) -> None:
        self.scale = scale

    def __call__(self, data_dict: Dict[str, Any], mode: str = "train") -> None:
        rgb = data_dict[DataKeys.RGB]
        if torch.is_tensor(rgb) and rgb.max() > 1.5:
            data_dict[DataKeys.RGB] = rgb / self.scale


class RGBNoiseProcessor(BaseDataProcessor):
    """Train-only Gaussian noise augmentation on RGB."""

    def __init__(self, std: float = 0.02) -> None:
        self.std = std

    def __call__(self, data_dict: Dict[str, Any], mode: str = "train") -> None:
        if mode != "train":
            return
        rgb = data_dict[DataKeys.RGB]
        if torch.is_tensor(rgb):
            data_dict[DataKeys.RGB] = rgb + torch.randn_like(rgb) * self.std


class PlannerRGBNoiseProcessor(BaseDataProcessor):
    """Train-only Gaussian noise on the planner RGB tensor (PlannerDataKeys.RGB).

    Operates on the planner sample dict (rgb is ``[n_img, H, W, 3]`` in [0,255]),
    demonstrating the configurable DataProcessor layer (dbn.md section 5) on the
    TrackVLA planner path. A no-op outside training.
    """

    def __init__(self, std: float = 3.0) -> None:
        self.std = std

    def __call__(self, data_dict: Dict[str, Any], mode: str = "train") -> None:
        if mode != "train":
            return
        from vla.omni.ported.projects.vln.constant.keys import PlannerDataKeys

        rgb = data_dict.get(PlannerDataKeys.RGB)
        if torch.is_tensor(rgb):
            data_dict[PlannerDataKeys.RGB] = (rgb + torch.randn_like(rgb) * self.std).clamp_(0, 255)
