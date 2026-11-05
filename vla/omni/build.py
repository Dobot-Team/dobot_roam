"""Factories for the preserved Omni components, without the legacy trainer."""
from .ported.util.build import build_from_cfg
from .ported.util.registry import Registry
from .ported.projects.vln.data.dataset.dobot_omni_rgb_dataset import DobotOmniRGBDataset
from .ported.projects.vln.data.dataset.data_processor.omni_prompt_processor import OmniVLNPromptProcessor, OmniRGBNoiseProcessor
from .ported.projects.vln.data.dataset.data_processor.omni_tasks import DEFAULT_PROCESSORS
from .ported.projects.vln.data.collate_fn.omni_planner_collate import omni_planner_collate_fn
from .ported.projects.vln.model.arch.omni_vln_planner import OmniVLNPlannerWithActToken
from .ported.projects.vln.model.head.omtrack_planner_head import OmTrackPlannerHead
from .ported.projects.vln.model.loss.task_aware_trajectory_loss import TaskAwareTrajectoryRegressionLoss
from .ported.projects.vln.model.qwen3_5.official_vlm import OfficialQwen35VLM
from .ported.projects.vln.model.qwen3_5.vlm import Qwen35VLM

DATA = Registry('OMNI_DATA')
MODEL = Registry('OMNI_MODEL')
for cls in (DobotOmniRGBDataset, OmniVLNPromptProcessor, OmniRGBNoiseProcessor, *DEFAULT_PROCESSORS.values()):
    DATA.register_module(cls)
for cls in (OmniVLNPlannerWithActToken, OmTrackPlannerHead, TaskAwareTrajectoryRegressionLoss, OfficialQwen35VLM, Qwen35VLM):
    MODEL.register_module(cls)


def build_dataset(config):
    return build_from_cfg(config, DATA)


def build_model(config):
    return build_from_cfg(config, MODEL)
