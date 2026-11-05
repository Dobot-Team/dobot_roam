"""Per-task instruction and target construction; no dataset clock changes."""
import torch

from vla.omni.ported.projects.vln.constant.omni import TASK_NAMES
from vla.omni.ported.projects.vln.data.dataset.data_processor.base_data_processor import BaseDataProcessor
from vla.omni.ported.projects.vln.data.dataset.dobot_omtrack_dataset import classify_roaming_motion_intent

ROAM_PROMPT = "Roam slowly while prioritizing safety."
INTENT_PROMPTS = {
    "left_turn": "Turn left.", "forward_left": "Move forward-left.",
    "straight": "Go straight.", "forward_right": "Move forward-right.",
    "right_turn": "Turn right.", "stop": "Stop.",
}


def set_instruction(sample, text):
    sample["instruction_text"] = sample["instruction"] = text
    sample["instruction_parts"] = [{"text": text}]


class TrackingTaskProcessor(BaseDataProcessor):
    def __call__(self, sample, mode="train"):
        # Preserve source instruction and already-retargeted trajectory exactly.
        set_instruction(sample, sample["instruction_text"])


class RoamingTaskProcessor(BaseDataProcessor):
    def __init__(self, instruction=ROAM_PROMPT):
        self.instruction = instruction

    def __call__(self, sample, mode="train"):
        set_instruction(sample, self.instruction)


class IntentTaskProcessor(BaseDataProcessor):
    def __init__(self, prompts=None, stop_max_translation_m=0.2,
                 stop_max_abs_yaw_rad=0.15, straight_max_abs_yaw_rad=0.15,
                 straight_max_abs_path_bearing_rad=0.15,
                 turn_min_abs_deviation_rad=0.45):
        self.prompts = dict(INTENT_PROMPTS if prompts is None else prompts)
        if set(self.prompts) != set(INTENT_PROMPTS):
            raise ValueError("intent prompts must contain the existing six classes")
        self.thresholds = dict(
            stop_max_translation_m=stop_max_translation_m,
            stop_max_abs_yaw_rad=stop_max_abs_yaw_rad,
            straight_max_abs_yaw_rad=straight_max_abs_yaw_rad,
            straight_max_abs_path_bearing_rad=straight_max_abs_path_bearing_rad,
            turn_min_abs_deviation_rad=turn_min_abs_deviation_rad,
        )

    def __call__(self, sample, mode="train"):
        intent = classify_roaming_motion_intent(
            sample["future_trajectory"].numpy(),
            sample["future_trajectory_mask"].numpy(), **self.thresholds)
        if intent is None:
            raise ValueError("intent sample has no valid trajectory")
        set_instruction(sample, self.prompts[intent])
        sample["intent_class"] = intent


def stop_at_goal(sample):
    """Retain the existing (possibly time-resampled) prefix, hold after arrival."""
    goal = sample["_future_goal"]
    target = sample["future_trajectory"].clone()
    mask = sample["future_trajectory_mask"].clone()
    times = torch.arange(1, len(target) + 1, dtype=torch.float64) * sample["_waypoint_dt"]
    reached = times >= float(goal["time_s"]) - 1e-7
    pose = torch.as_tensor(goal["pose"], dtype=target.dtype)
    if pose.shape != target.shape[1:] or not torch.isfinite(pose).all():
        raise ValueError("invalid goal pose")
    target[reached] = pose
    mask[reached] = True
    sample["future_trajectory"] = target
    sample["future_trajectory_mask"] = mask
    sample["goal_metadata"] = {k: v for k, v in goal.items() if k != "rgb"}


class PointNavigationTaskProcessor(BaseDataProcessor):
    def __init__(self, instruction_template="Move to the point({x:.3f}, {y:.3f})."):
        self.template = instruction_template

    def __call__(self, sample, mode="train"):
        goal = sample["_future_goal"]
        set_instruction(sample, self.template.format(x=goal["pose"][0], y=goal["pose"][1]))
        stop_at_goal(sample)


class ImageNavigationTaskProcessor(BaseDataProcessor):
    def __init__(self, instruction_template="Move to the place in the {image}."):
        if instruction_template.count("{image}") != 1:
            raise ValueError("image instruction needs exactly one {image} block")
        self.before, self.after = instruction_template.split("{image}")

    def __call__(self, sample, mode="train"):
        goal = sample["_future_goal"]
        sample["goal_rgb"] = goal["rgb"]
        set_instruction(sample, self.before + "{image}" + self.after)
        sample["instruction_parts"] = [
            {"text": self.before}, {"image": "goal_rgb"}, {"text": self.after}]
        stop_at_goal(sample)


DEFAULT_PROCESSORS = dict(zip(TASK_NAMES, (
    TrackingTaskProcessor, RoamingTaskProcessor, IntentTaskProcessor,
    PointNavigationTaskProcessor, ImageNavigationTaskProcessor,
)))
