"""Dobot OmTrack dataset backed by precomputed vision tokens.

The source JSON/JSONL records remain lazy and the source cache is read-only.  A
sample contains historical coarse-token frames, one current fine-token frame,
an instruction, and an 8x3 local-frame waypoint target.  When
``num_input_frames`` and ``input_hz`` are configured, the total visual input
contains ``num_input_frames`` frames sampled on that temporal grid (the last
frame is the current fine-token frame).  An optional online vision encoder can
fill cache misses without mutating the shared source cache.
"""

import hashlib
import json
import logging
import math
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from vla.omni.ported.constant.data import DataMetricKeys
from vla.omni.ported.projects.vln.constant.keys import OmTrackDataKeys, PlannerDataKeys
from vla.omni.ported.projects.vln.data.dataset.base_dataset import BaseVLNDataset, StatefulDataset
from vla.omni.ported.projects.vln.data.dataset.data_processor.base_data_processor import BaseDataProcessor

logger = logging.getLogger(__name__)


ROAMING_MOTION_INTENTS = (
    "left_turn",
    "forward_left",
    "straight",
    "forward_right",
    "right_turn",
    "stop",
)
DEFAULT_ROAMING_INTENT_PROMPTS = {
    "left_turn": "Turn left.",
    "forward_left": "Move forward-left.",
    "straight": "Go straight.",
    "forward_right": "Move forward-right.",
    "right_turn": "Turn right.",
    "stop": "Stop.",
}

SOURCE_SAMPLING_GROUPS = (
    "dobot_tracking",
    "dobot_roaming",
    "trackvla_at",
    "trackvla_dt",
    "trackvla_stt",
    "other",
)


def _stable_sample_fraction(identity: str, seed: int) -> float:
    """Map a sample identity to a reproducible value in ``[0, 1)``."""
    digest = hashlib.blake2b(digest_size=8, person=b"dbn-roam")
    digest.update(str(int(seed)).encode("utf-8"))
    digest.update(b"\0")
    digest.update(identity.encode("utf-8"))
    return int.from_bytes(digest.digest(), byteorder="big") / float(1 << 64)


def classify_roaming_motion_intent(
    waypoints: np.ndarray,
    mask: np.ndarray,
    *,
    stop_max_translation_m: float,
    stop_max_abs_yaw_rad: float,
    straight_max_abs_yaw_rad: float,
    straight_max_abs_path_bearing_rad: float,
    turn_min_abs_deviation_rad: float,
) -> Optional[str]:
    """Classify valid local-frame ``[x, y, yaw]`` GT into six commands.

    Positive yaw/path bearing is treated as a left turn and negative values as
    a right turn. Stop is evaluated first over the complete valid horizon. A
    trajectory is straight only when both its heading and its displacement
    bearing stay within their configured bounds for the complete valid future.
    This deliberately rejects paths that bend and later return to a near-zero
    terminal yaw. ``None`` means that the record has no usable GT and its
    prompt should stay unchanged.
    """
    trajectory = np.asarray(waypoints, dtype=np.float64)
    valid_mask = np.asarray(mask, dtype=np.bool_).copy()
    if trajectory.ndim != 2 or trajectory.shape[1] < 3:
        raise ValueError(
            "roaming intent waypoints must have shape [steps, coord_dim>=3]"
        )
    if valid_mask.shape != (trajectory.shape[0],):
        raise ValueError(
            "roaming intent mask must have one value per waypoint"
        )
    valid_mask &= np.isfinite(trajectory[:, :3]).all(axis=1)
    if not valid_mask.any():
        return None

    valid_trajectory = trajectory[valid_mask, :3]
    translation = np.linalg.norm(valid_trajectory[:, :2], axis=1)
    yaw = np.unwrap(valid_trajectory[:, 2])
    if (
        float(translation.max()) <= stop_max_translation_m
        and float(np.abs(yaw).max()) <= stop_max_abs_yaw_rad
    ):
        return "stop"

    # Ignore near-origin bearings: their direction is unstable and stop has
    # already been decided from the full trajectory above.
    moving = translation > stop_max_translation_m
    path_bearing = np.arctan2(
        valid_trajectory[moving, 1], valid_trajectory[moving, 0]
    )
    max_abs_path_bearing = (
        float(np.abs(path_bearing).max()) if path_bearing.size else 0.0
    )
    if (
        float(np.abs(yaw).max()) <= straight_max_abs_yaw_rad
        and max_abs_path_bearing <= straight_max_abs_path_bearing_rad
    ):
        return "straight"

    # Whichever of heading or path bearing deviates most determines the turn
    # side. This also handles turning-in-place trajectories with no moving
    # waypoint beyond the stop translation threshold.
    dominant_turn_signal = float(yaw[np.abs(yaw).argmax()])
    if path_bearing.size:
        dominant_path_bearing = float(
            path_bearing[np.abs(path_bearing).argmax()]
        )
        if abs(dominant_path_bearing) > abs(dominant_turn_signal):
            dominant_turn_signal = dominant_path_bearing
    if abs(dominant_turn_signal) >= turn_min_abs_deviation_rad:
        return "left_turn" if dominant_turn_signal > 0.0 else "right_turn"
    return "forward_left" if dominant_turn_signal > 0.0 else "forward_right"


def _load_tokens_file(path: Path) -> torch.Tensor:
    try:
        obj = torch.load(str(path), map_location="cpu")
    except Exception:
        obj = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(obj, torch.Tensor):
        return obj.float()
    if isinstance(obj, dict):
        for key in ("V", "Vfine", "Vcoarse", "tokens", "feat", "features"):
            value = obj.get(key)
            if isinstance(value, torch.Tensor):
                if value.ndim == 3 and value.size(0) == 1:
                    value = value[0]
                return value.float()
    raise ValueError(f"unrecognized vision-token cache file: {path}")


def _integrate_actions(actions: np.ndarray, num_waypoints: int, dt: float) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim == 1:
        actions = actions[None, :]
    steps, dims = actions.shape
    vx = actions[:, 0]
    vy = actions[:, 1] if dims > 1 else np.zeros_like(vx)
    wz = actions[:, 2] if dims > 2 else np.zeros_like(vx)
    x = np.zeros(steps, dtype=np.float32)
    y = np.zeros(steps, dtype=np.float32)
    yaw = np.zeros(steps, dtype=np.float32)
    for index in range(1, steps):
        yaw[index] = yaw[index - 1] + wz[index - 1] * dt
        cosine, sine = np.cos(yaw[index - 1]), np.sin(yaw[index - 1])
        x[index] = x[index - 1] + (cosine * vx[index - 1] - sine * vy[index - 1]) * dt
        y[index] = y[index - 1] + (sine * vx[index - 1] + cosine * vy[index - 1]) * dt
    trajectory = np.stack((x, y, yaw), axis=-1)
    indices = np.linspace(0, steps - 1, num_waypoints).round().astype(np.int64)
    return trajectory[indices]


def _trackvla_time_retargeted_waypoints(
    actions: np.ndarray,
    *,
    num_waypoints: int,
    waypoint_dt: float,
    record_dt: float,
    action_integration_dt: float,
    speed_coefficients: Sequence[float],
    source_to_target_time_scale: float,
    padding_value: float,
    preserve_original_horizon_mask: bool,
    yaw_source_to_target_time_scale: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build simulator-scaled TrackVLA targets on a retargeted time grid.

    Raw TrackVLA actions are normalized controller commands.  Each component
    is clipped, converted with its configured forward/lateral/yaw speed
    coefficient, and integrated for ``action_integration_dt``.  The resulting
    pose after each action remains associated with the source record timeline
    for interpolation.  By default yaw uses the same retargeted clock as x/y.
    ``yaw_source_to_target_time_scale`` can use a faster yaw clock while x/y
    retain the source-to-target translation speed mapping.  Once that yaw
    clock reaches the source horizon, the final heading is held rather than
    extrapolated.

    This reconstructs an ideal command rollout only.  Collision filtering,
    NavMesh clamping, and dynamic sliding cannot be recovered from the stored
    actions.
    """
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim == 1:
        actions = actions[None, :]
    if actions.ndim != 2 or actions.shape[1] < 3 or actions.shape[0] == 0:
        raise ValueError("TrackVLA actions must have shape [steps, >=3]")
    actions = actions[:, :3]
    if not np.isfinite(actions).all():
        raise ValueError("TrackVLA actions contain non-finite values")

    coefficients = np.asarray(speed_coefficients, dtype=np.float64)
    if coefficients.shape != (3,):
        raise ValueError("TrackVLA speed_coefficients must contain 3 values")

    scaled_actions = np.clip(actions, -1.0, 1.0) * coefficients[None, :]
    trajectory = np.empty((actions.shape[0], 3), dtype=np.float64)
    x = y = yaw = 0.0
    for index, (vx, vy, wz) in enumerate(scaled_actions):
        cosine, sine = np.cos(yaw), np.sin(yaw)
        x += (cosine * vx - sine * vy) * action_integration_dt
        y += (sine * vx + cosine * vy) * action_integration_dt
        yaw += wz * action_integration_dt
        trajectory[index] = (x, y, yaw)

    # Include the current origin so target times between 0 and record_dt can be
    # interpolated instead of rounded to the first recorded action.
    source_poses = np.concatenate(
        (np.zeros((1, 3), dtype=np.float64), trajectory), axis=0
    )
    source_poses[:, 2] = np.unwrap(source_poses[:, 2])
    source_times = np.arange(source_poses.shape[0], dtype=np.float64) * record_dt
    # A faster source trajectory is stretched onto the target clock. With
    # TrackVLA speed v2 and Dobot speed v1 this is t_dobot=(v2/v1)*t_trackvla.
    mapped_source_times = source_times * source_to_target_time_scale
    target_times = (
        np.arange(1, num_waypoints + 1, dtype=np.float64) * waypoint_dt
    )
    tolerance = max(record_dt, waypoint_dt) * 1.0e-9
    available = target_times <= mapped_source_times[-1] + tolerance
    if preserve_original_horizon_mask:
        # Keep the pre-retargeting supervision horizon.  Nine 10 Hz actions
        # therefore continue to supervise only 0.3, 0.6, and 0.9 seconds.
        available &= target_times <= source_times[-1] + tolerance

    waypoints = np.full(
        (num_waypoints, 3), padding_value, dtype=np.float32
    )
    for coord_index in range(2):
        waypoints[available, coord_index] = np.interp(
            target_times[available],
            mapped_source_times,
            source_poses[:, coord_index],
        ).astype(np.float32)
    if yaw_source_to_target_time_scale is None:
        # Preserve the legacy interpolation path exactly for existing configs.
        yaw_mapped_source_times = mapped_source_times
        yaw_target_times = target_times[available]
    else:
        yaw_scale = float(yaw_source_to_target_time_scale)
        if not np.isfinite(yaw_scale) or yaw_scale <= 0.0:
            raise ValueError(
                "yaw_source_to_target_time_scale must be finite and positive"
            )
        yaw_mapped_source_times = source_times * yaw_scale
        yaw_target_times = np.minimum(
            target_times[available], yaw_mapped_source_times[-1]
        )
    waypoints[available, 2] = np.interp(
        yaw_target_times,
        yaw_mapped_source_times,
        source_poses[:, 2],
    ).astype(np.float32)
    return waypoints, available


class DobotOmTrackDataset(BaseVLNDataset, StatefulDataset):
    """Lazy, stateful reader for Dobot OmTrack JSONL and vision-token caches."""

    def __init__(
        self,
        json_path: str,
        cache_root: Optional[str] = None,
        history: int = 31,
        num_input_frames: Optional[int] = None,
        input_hz: Optional[float] = None,
        frame_index_hz: float = 10.0,
        num_waypoints: int = 8,
        coord_dim: int = 3,
        coarse_tokens_per_frame: int = 4,
        fine_tokens_per_frame: int = 64,
        vision_feature_dim: int = 1536,
        cached_feature_slice: Optional[Sequence[int]] = None,
        default_dt: float = 0.1,
        trajectory_dt: float = 0.1,
        waypoint_dt: float = 0.3,
        trajectory_padding_value: float = -1.0,
        base_root: Optional[str] = None,
        exclude_jsonl_path_parts: Optional[List[str]] = None,
        follow_jsonl_directory_symlinks: bool = False,
        length: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 42,
        mode: str = "train",
        strict_cache_shapes: bool = True,
        online_encode_on_cache_miss: bool = False,
        online_vision_encoder_config: Optional[Mapping[str, Any]] = None,
        online_token_cache_size: int = 64,
        history_padding: str = "current",
        worker_shard_batch_size: Optional[int] = None,
        shuffle_block_size: Optional[int] = None,
        resample_on_error: bool = False,
        max_resample_times: int = 100,
        data_processor_list: Optional[List[BaseDataProcessor]] = None,
        preload_size: int = 1,
        additional_json_paths: Optional[Sequence[str]] = None,
        trackvla_time_retargeting: Optional[Mapping[str, Any]] = None,
        roaming_instruction_relabeling: Optional[Mapping[str, Any]] = None,
        source_sampling: Optional[Mapping[str, Any]] = None,
        sample_selector: Optional[Mapping[str, Any]] = None,
        record_filter: Optional[Mapping[str, Any]] = None,
        instruction_override: Optional[str] = None,
        target_mode: str = "recorded",
    ) -> None:
        del preload_size  # ordering is maintained explicitly for resumability
        self.json_path = Path(json_path)
        additional_paths = tuple(additional_json_paths or ())
        if any(not str(path).strip() for path in additional_paths):
            raise ValueError("additional_json_paths cannot contain an empty path")
        self.additional_json_paths = tuple(Path(path) for path in additional_paths)
        self.json_paths = (self.json_path, *self.additional_json_paths)
        if len(set(self.json_paths)) != len(self.json_paths):
            raise ValueError("json_path and additional_json_paths must be unique")
        self.cache_root = Path(cache_root) if cache_root else None
        self.history = int(history)
        if (num_input_frames is None) != (input_hz is None):
            raise ValueError("num_input_frames and input_hz must be configured together")
        self._temporal_sampling_enabled = num_input_frames is not None
        self.num_input_frames = (
            self.history + 1 if num_input_frames is None else int(num_input_frames)
        )
        self.input_hz = None if input_hz is None else float(input_hz)
        self.frame_index_hz = float(frame_index_hz)
        self._timestamp_timeline_cache: Dict[str, np.ndarray] = {}
        self.num_waypoints = int(num_waypoints)
        self.coord_dim = int(coord_dim)
        self.coarse_tokens_per_frame = int(coarse_tokens_per_frame)
        self.fine_tokens_per_frame = int(fine_tokens_per_frame)
        self.vision_feature_dim = int(vision_feature_dim)
        self.cached_feature_slice: Optional[Tuple[int, int]] = None
        if cached_feature_slice is not None:
            if len(cached_feature_slice) != 2:
                raise ValueError("cached_feature_slice must contain [start, stop]")
            start, stop = (int(value) for value in cached_feature_slice)
            if start < 0 or stop <= start:
                raise ValueError(
                    "cached_feature_slice must satisfy 0 <= start < stop"
                )
            if stop - start != self.vision_feature_dim:
                raise ValueError(
                    "cached_feature_slice width must equal vision_feature_dim"
                )
            self.cached_feature_slice = (start, stop)
        self.default_dt = float(default_dt)
        # Opt in per dataset: preserve the existing mixed-data target path by default.
        self.target_mode = str(target_mode)
        if self.target_mode not in {"recorded", "omtrack_official"}:
            raise ValueError("target_mode must be 'recorded' or 'omtrack_official'")
        if self.target_mode == "omtrack_official" and self.coord_dim != 3:
            raise ValueError("OmTrack official action integration requires coord_dim=3")
        self.trajectory_dt = float(trajectory_dt)
        self.waypoint_dt = float(waypoint_dt)
        self.trajectory_padding_value = float(trajectory_padding_value)
        if instruction_override is None:
            self.instruction_override: Optional[str] = None
        else:
            self.instruction_override = str(instruction_override).strip()
            if not self.instruction_override:
                raise ValueError("instruction_override cannot be empty")
        if sample_selector is not None and not isinstance(sample_selector, Mapping):
            raise ValueError("sample_selector must be a mapping")
        self.sample_selector = (
            None if sample_selector is None else dict(sample_selector)
        )
        if record_filter is not None and not isinstance(record_filter, Mapping):
            raise ValueError("record_filter must be a mapping")
        self.record_filter: Dict[str, Tuple[Any, ...]] = {}
        for raw_field, raw_allowed_values in dict(record_filter or {}).items():
            field = str(raw_field).strip()
            if not field:
                raise ValueError("record_filter field names cannot be empty")
            if isinstance(raw_allowed_values, (str, bytes)):
                allowed_values = (raw_allowed_values,)
            elif isinstance(raw_allowed_values, Sequence):
                allowed_values = tuple(raw_allowed_values)
            else:
                allowed_values = (raw_allowed_values,)
            if not allowed_values:
                raise ValueError(
                    f"record_filter values for {field!r} cannot be empty"
                )
            self.record_filter[field] = allowed_values
        if record_filter is not None and not self.record_filter:
            raise ValueError("record_filter cannot be empty")
        if source_sampling is not None and not isinstance(source_sampling, Mapping):
            raise ValueError("source_sampling must be a mapping")
        source_sampling_config = dict(source_sampling or {})
        allowed_source_sampling_keys = {
            "enabled",
            "seed",
            "dobot_data_source",
            "ratios",
        }
        unknown_source_sampling_keys = (
            set(source_sampling_config) - allowed_source_sampling_keys
        )
        if unknown_source_sampling_keys:
            raise ValueError(
                "unknown source_sampling keys: "
                f"{sorted(unknown_source_sampling_keys)}"
            )
        self.source_sampling_enabled = bool(
            source_sampling_config.get("enabled", False)
        )
        self.source_sampling_seed = int(source_sampling_config.get("seed", seed))
        raw_dobot_data_source = str(
            source_sampling_config.get("dobot_data_source", "all")
        ).strip().lower()
        dobot_data_source_aliases = {
            "dobot": "dobot",
            "dobot_v2.0": "dobot_v2.0",
            "dobot2.0": "dobot_v2.0",
            "all": "all",
        }
        if raw_dobot_data_source not in dobot_data_source_aliases:
            raise ValueError(
                "source_sampling dobot_data_source must be one of "
                "{'dobot', 'dobot_v2.0', 'dobot2.0', 'all'}"
            )
        self.dobot_data_source = dobot_data_source_aliases[raw_dobot_data_source]
        raw_source_sampling_ratios = source_sampling_config.get("ratios", {})
        if not isinstance(raw_source_sampling_ratios, Mapping):
            raise ValueError("source_sampling ratios must be a mapping")
        unknown_source_sampling_groups = (
            set(raw_source_sampling_ratios) - set(SOURCE_SAMPLING_GROUPS)
        )
        if unknown_source_sampling_groups:
            raise ValueError(
                "unknown source_sampling ratio groups: "
                f"{sorted(unknown_source_sampling_groups)}"
            )
        self.source_sampling_ratios = {
            group: float(raw_source_sampling_ratios.get(group, 1.0))
            for group in SOURCE_SAMPLING_GROUPS
        }
        for group, ratio in self.source_sampling_ratios.items():
            if not math.isfinite(ratio) or ratio < 0.0:
                raise ValueError(
                    f"source_sampling ratio for {group!r} must be finite "
                    "and non-negative"
                )
        self.source_record_counts: Dict[str, int] = {}
        self.sampled_source_record_counts: Dict[str, int] = {}
        retargeting = dict(trackvla_time_retargeting or {})
        allowed_retargeting_keys = {
            "enabled",
            "path_component",
            "simulator_ctrl_freq_hz",
            "record_frequency_hz",
            "action_integration_dt",
            "dobot_speed_v1_mps",
            "trackvla_speed_v2_mps",
            "speed_coefficients",
            "speed_coefficients_by_source",
            "time_scale",
            "yaw_source_to_target_time_scale",
            "preserve_original_horizon_mask",
        }
        unknown_retargeting_keys = set(retargeting) - allowed_retargeting_keys
        if unknown_retargeting_keys:
            raise ValueError(
                "unknown trackvla_time_retargeting keys: "
                f"{sorted(unknown_retargeting_keys)}"
            )
        self.trackvla_time_retarget_enabled = bool(
            retargeting.get("enabled", False)
        )
        if self.target_mode == "omtrack_official" and self.trackvla_time_retarget_enabled:
            raise ValueError("OmTrack official targets cannot use TrackVLA time retargeting")
        self.trackvla_path_component = str(
            retargeting.get("path_component", "trackvla")
        )
        self.trackvla_simulator_ctrl_freq_hz = float(
            retargeting.get("simulator_ctrl_freq_hz", 40.0)
        )
        self.trackvla_record_frequency_hz = float(
            retargeting.get("record_frequency_hz", 10.0)
        )
        raw_action_integration_dt = retargeting.get("action_integration_dt")
        self.trackvla_action_integration_dt = (
            float(raw_action_integration_dt)
            if raw_action_integration_dt is not None
            else (
                1.0 / self.trackvla_simulator_ctrl_freq_hz
                if self.trackvla_simulator_ctrl_freq_hz > 0.0
                else float("nan")
            )
        )
        raw_speed_coefficients = retargeting.get("speed_coefficients")
        raw_speed_coefficients_by_source = retargeting.get(
            "speed_coefficients_by_source"
        )
        if (
            raw_speed_coefficients is not None
            and raw_speed_coefficients_by_source is not None
        ):
            raise ValueError(
                "configure either TrackVLA speed_coefficients or "
                "speed_coefficients_by_source, not both"
            )
        self.trackvla_speed_coefficients: Optional[Tuple[float, ...]] = None
        self.trackvla_speed_coefficients_by_source: Dict[
            str, Tuple[float, ...]
        ] = {}
        if raw_speed_coefficients_by_source is not None:
            if not isinstance(raw_speed_coefficients_by_source, Mapping):
                raise ValueError(
                    "TrackVLA speed_coefficients_by_source must be a mapping"
                )
            self.trackvla_speed_coefficients_by_source = {
                str(source): tuple(float(value) for value in coefficients)
                for source, coefficients in raw_speed_coefficients_by_source.items()
            }
        else:
            self.trackvla_speed_coefficients = tuple(
                float(value)
                for value in (
                    raw_speed_coefficients
                    if raw_speed_coefficients is not None
                    else (15.0, 5.0, 6.28)
                )
            )
        raw_dobot_speed_v1_mps = retargeting.get("dobot_speed_v1_mps")
        raw_trackvla_speed_v2_mps = retargeting.get("trackvla_speed_v2_mps")
        if (raw_dobot_speed_v1_mps is None) != (
            raw_trackvla_speed_v2_mps is None
        ):
            raise ValueError(
                "configure both dobot_speed_v1_mps and "
                "trackvla_speed_v2_mps, or neither"
            )
        if raw_dobot_speed_v1_mps is not None and "time_scale" in retargeting:
            raise ValueError(
                "configure either the Dobot/TrackVLA speed pair or time_scale, "
                "not both"
            )
        self.trackvla_dobot_speed_v1_mps: Optional[float] = None
        self.trackvla_speed_v2_mps: Optional[float] = None
        if raw_dobot_speed_v1_mps is not None:
            self.trackvla_dobot_speed_v1_mps = float(raw_dobot_speed_v1_mps)
            self.trackvla_speed_v2_mps = float(raw_trackvla_speed_v2_mps)
            self.trackvla_time_scale = (
                self.trackvla_dobot_speed_v1_mps / self.trackvla_speed_v2_mps
                if self.trackvla_speed_v2_mps != 0.0
                else float("nan")
            )
            self.trackvla_source_to_dobot_time_scale = (
                self.trackvla_speed_v2_mps / self.trackvla_dobot_speed_v1_mps
                if self.trackvla_dobot_speed_v1_mps != 0.0
                else float("nan")
            )
        else:
            # Legacy configs express the inverse lookup directly: a Dobot
            # target time is multiplied by time_scale to query source time.
            self.trackvla_time_scale = float(retargeting.get("time_scale", 1.0))
            self.trackvla_source_to_dobot_time_scale = (
                1.0 / self.trackvla_time_scale
                if self.trackvla_time_scale > 0.0
                else float("nan")
            )
        raw_yaw_source_to_target_time_scale = retargeting.get(
            "yaw_source_to_target_time_scale"
        )
        self.trackvla_yaw_source_to_target_time_scale: Optional[float] = (
            None
            if raw_yaw_source_to_target_time_scale is None
            else float(raw_yaw_source_to_target_time_scale)
        )
        self.trackvla_preserve_original_horizon_mask = bool(
            retargeting.get("preserve_original_horizon_mask", True)
        )
        relabeling = dict(roaming_instruction_relabeling or {})
        allowed_relabeling_keys = {
            "enabled",
            "ratio",
            "seed",
            "source_path_component",
            "prompts",
            "stop_max_translation_m",
            "stop_max_abs_yaw_rad",
            "straight_max_abs_yaw_rad",
            "straight_max_abs_path_bearing_rad",
            "turn_min_abs_deviation_rad",
        }
        unknown_relabeling_keys = set(relabeling) - allowed_relabeling_keys
        if unknown_relabeling_keys:
            raise ValueError(
                "unknown roaming_instruction_relabeling keys: "
                f"{sorted(unknown_relabeling_keys)}"
            )
        self.roaming_instruction_relabeling_enabled = bool(
            relabeling.get("enabled", False)
        )
        self.roaming_instruction_relabeling_ratio = float(
            relabeling.get("ratio", 0.0)
        )
        self.roaming_instruction_relabeling_seed = int(
            relabeling.get("seed", seed)
        )
        self.roaming_instruction_source_path_component = str(
            relabeling.get("source_path_component", "jsonl_train_manyou")
        )
        raw_prompts = relabeling.get("prompts", DEFAULT_ROAMING_INTENT_PROMPTS)
        if not isinstance(raw_prompts, Mapping):
            raise ValueError("roaming instruction prompts must be a mapping")
        missing_prompts = set(ROAMING_MOTION_INTENTS) - set(raw_prompts)
        unknown_prompts = set(raw_prompts) - set(ROAMING_MOTION_INTENTS)
        if missing_prompts or unknown_prompts:
            raise ValueError(
                "roaming instruction prompts require exactly "
                f"{list(ROAMING_MOTION_INTENTS)}; missing={sorted(missing_prompts)}, "
                f"unknown={sorted(unknown_prompts)}"
            )
        self.roaming_instruction_prompts = {
            intent: str(raw_prompts[intent]).strip()
            for intent in ROAMING_MOTION_INTENTS
        }
        if any(not prompt for prompt in self.roaming_instruction_prompts.values()):
            raise ValueError("roaming instruction prompts cannot be empty")
        self.roaming_stop_max_translation_m = float(
            relabeling.get("stop_max_translation_m", 0.2)
        )
        self.roaming_stop_max_abs_yaw_rad = float(
            relabeling.get("stop_max_abs_yaw_rad", 0.15)
        )
        self.roaming_straight_max_abs_yaw_rad = float(
            relabeling.get("straight_max_abs_yaw_rad", 0.15)
        )
        self.roaming_straight_max_abs_path_bearing_rad = float(
            relabeling.get("straight_max_abs_path_bearing_rad", 0.15)
        )
        self.roaming_turn_min_abs_deviation_rad = float(
            relabeling.get("turn_min_abs_deviation_rad", 0.45)
        )
        self.exclude_jsonl_path_parts = frozenset(
            str(part) for part in (exclude_jsonl_path_parts or [])
        )
        self.follow_jsonl_directory_symlinks = bool(
            follow_jsonl_directory_symlinks
        )
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.mode = mode
        self.strict_cache_shapes = bool(strict_cache_shapes)
        self.online_encode_on_cache_miss = bool(online_encode_on_cache_miss)
        self.online_vision_encoder_config = (
            None
            if online_vision_encoder_config is None
            else dict(online_vision_encoder_config)
        )
        self.online_token_cache_size = int(online_token_cache_size)
        self._online_vision_encoder: Optional[Any] = None
        self._online_token_cache: OrderedDict[
            str, Tuple[torch.Tensor, torch.Tensor]
        ] = OrderedDict()
        self.history_padding = history_padding
        self.resample_on_error = bool(resample_on_error)
        self.max_resample_times = int(max_resample_times)
        self.worker_shard_batch_size = (
            None if worker_shard_batch_size is None else int(worker_shard_batch_size)
        )
        self.shuffle_block_size = (
            None if shuffle_block_size is None else int(shuffle_block_size)
        )
        self.data_processor_list = data_processor_list or []

        if self.history <= 0 or self.num_waypoints <= 0:
            raise ValueError("history and num_waypoints must be positive")
        if self.num_input_frames < 2:
            raise ValueError("num_input_frames must be at least 2")
        if self.input_hz is not None and self.input_hz <= 0:
            raise ValueError("input_hz must be positive")
        if self.frame_index_hz <= 0:
            raise ValueError("frame_index_hz must be positive")
        if self.trajectory_dt <= 0 or self.waypoint_dt <= 0:
            raise ValueError("trajectory_dt and waypoint_dt must be positive")
        if (
            not np.isfinite(self.roaming_instruction_relabeling_ratio)
            or not 0.0 <= self.roaming_instruction_relabeling_ratio <= 1.0
        ):
            raise ValueError("roaming instruction relabeling ratio must be in [0, 1]")
        if (
            self.roaming_instruction_relabeling_enabled
            and not self.roaming_instruction_source_path_component
        ):
            raise ValueError(
                "roaming instruction source_path_component cannot be empty"
            )
        if self.roaming_instruction_relabeling_enabled and self.coord_dim < 3:
            raise ValueError(
                "roaming instruction relabeling requires coord_dim>=3"
            )
        for name, value in (
            (
                "stop_max_translation_m",
                self.roaming_stop_max_translation_m,
            ),
            ("stop_max_abs_yaw_rad", self.roaming_stop_max_abs_yaw_rad),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"roaming instruction {name} must be finite and non-negative"
                )
        if (
            not np.isfinite(self.roaming_turn_min_abs_deviation_rad)
            or self.roaming_turn_min_abs_deviation_rad
            <= max(
                self.roaming_straight_max_abs_yaw_rad,
                self.roaming_straight_max_abs_path_bearing_rad,
            )
        ):
            raise ValueError(
                "roaming instruction turn_min_abs_deviation_rad must be finite "
                "and greater than both straight thresholds"
            )
        for name, value in (
            (
                "straight_max_abs_yaw_rad",
                self.roaming_straight_max_abs_yaw_rad,
            ),
            (
                "straight_max_abs_path_bearing_rad",
                self.roaming_straight_max_abs_path_bearing_rad,
            ),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"roaming instruction {name} must be finite and positive"
                )
        if self.trackvla_time_retarget_enabled:
            if self.coord_dim != 3:
                raise ValueError(
                    "TrackVLA time retargeting requires coord_dim=3"
                )
            if not self.trackvla_path_component:
                raise ValueError(
                    "TrackVLA time retargeting path_component cannot be empty"
                )
            if (
                not np.isfinite(self.trackvla_simulator_ctrl_freq_hz)
                or self.trackvla_simulator_ctrl_freq_hz <= 0
            ):
                raise ValueError(
                    "TrackVLA simulator_ctrl_freq_hz must be finite and positive"
                )
            if (
                not np.isfinite(self.trackvla_record_frequency_hz)
                or self.trackvla_record_frequency_hz <= 0
            ):
                raise ValueError(
                    "TrackVLA record_frequency_hz must be finite and positive"
                )
            if (
                not np.isfinite(self.trackvla_action_integration_dt)
                or self.trackvla_action_integration_dt <= 0
            ):
                raise ValueError(
                    "TrackVLA action_integration_dt must be finite and positive"
                )
            if self.trackvla_speed_coefficients_by_source:
                for source, coefficients in (
                    self.trackvla_speed_coefficients_by_source.items()
                ):
                    if not source:
                        raise ValueError(
                            "TrackVLA speed_coefficients_by_source keys cannot be empty"
                        )
                    if (
                        len(coefficients) != 3
                        or not np.isfinite(coefficients).all()
                        or any(value <= 0 for value in coefficients)
                    ):
                        raise ValueError(
                            "TrackVLA speed coefficients for source "
                            f"{source!r} must contain 3 finite positive values"
                        )
            elif self.trackvla_speed_coefficients is None or (
                len(self.trackvla_speed_coefficients) != 3
                or not np.isfinite(self.trackvla_speed_coefficients).all()
                or any(value <= 0 for value in self.trackvla_speed_coefficients)
            ):
                raise ValueError(
                    "TrackVLA speed_coefficients must contain 3 finite positive values"
                )
            if not np.isfinite(self.trackvla_time_scale) or self.trackvla_time_scale <= 0:
                raise ValueError("TrackVLA time_scale must be finite and positive")
            if self.trackvla_dobot_speed_v1_mps is not None and (
                not np.isfinite(self.trackvla_dobot_speed_v1_mps)
                or self.trackvla_dobot_speed_v1_mps <= 0
                or self.trackvla_speed_v2_mps is None
                or not np.isfinite(self.trackvla_speed_v2_mps)
                or self.trackvla_speed_v2_mps <= 0
            ):
                raise ValueError(
                    "TrackVLA dobot_speed_v1_mps and trackvla_speed_v2_mps "
                    "must be finite and positive"
                )
            if (
                not np.isfinite(self.trackvla_source_to_dobot_time_scale)
                or self.trackvla_source_to_dobot_time_scale <= 0
            ):
                raise ValueError(
                    "TrackVLA source-to-Dobot time scale must be finite and positive"
                )
            if (
                self.trackvla_yaw_source_to_target_time_scale is not None
                and (
                    not np.isfinite(
                        self.trackvla_yaw_source_to_target_time_scale
                    )
                    or self.trackvla_yaw_source_to_target_time_scale <= 0.0
                )
            ):
                raise ValueError(
                    "TrackVLA yaw_source_to_target_time_scale must be finite "
                    "and positive"
                )
        trajectory_stride = self.waypoint_dt / self.trajectory_dt
        self.trajectory_stride = int(round(trajectory_stride))
        if self.trajectory_stride <= 0 or not np.isclose(
            trajectory_stride, self.trajectory_stride, rtol=0.0, atol=1.0e-6
        ):
            raise ValueError("waypoint_dt must be an integer multiple of trajectory_dt")
        if self.history_padding not in {"current", "earliest"}:
            raise ValueError("history_padding must be 'current' or 'earliest'")
        if self.worker_shard_batch_size is not None and self.worker_shard_batch_size <= 0:
            raise ValueError("worker_shard_batch_size must be positive")
        if self.shuffle_block_size is not None and self.shuffle_block_size <= 0:
            raise ValueError("shuffle_block_size must be positive")
        if self.max_resample_times <= 0:
            raise ValueError("max_resample_times must be positive")
        if "" in self.exclude_jsonl_path_parts:
            raise ValueError("exclude_jsonl_path_parts cannot contain an empty value")
        if self.online_token_cache_size <= 0:
            raise ValueError("online_token_cache_size must be positive")
        if self.cache_root is not None and not self.cache_root.is_dir():
            raise FileNotFoundError(f"vision cache root does not exist: {self.cache_root}")
        if (
            self.online_encode_on_cache_miss
            and self.online_vision_encoder_config is None
        ):
            raise ValueError(
                "online_vision_encoder_config is required when "
                "online_encode_on_cache_miss=True"
            )

        total_slots = self.history + 1
        self._fallback_frame_slot_indices = np.linspace(
            0, total_slots - 1, self.num_input_frames
        ).round().astype(np.int64).tolist()
        if self.input_hz is None:
            self._desired_time_offsets = None
            self.input_frame_times = None
        else:
            self._desired_time_offsets = np.arange(
                self.num_input_frames - 1, -1, -1, dtype=np.float64
            ) / self.input_hz
            self.input_frame_times = (-self._desired_time_offsets).tolist()

        self.base_root = Path(base_root) if base_root else self._infer_base_root()
        self._json_records: Optional[List[Dict[str, Any]]] = None
        self._jsonl_index: Optional[List[Tuple[str, int]]] = None
        total_records = self._build_index()
        if self.record_filter:
            total_records = self._apply_record_filter(total_records)
        if self.sample_selector is not None:
            total_records = self._select_single_sample(
                self.sample_selector, total_records
            )
        requested_length = total_records if length is None else min(int(length), total_records)
        if requested_length <= 0:
            raise ValueError(f"dataset is empty: {json_path}")
        super().__init__(length=requested_length, preload_size=1)

        self._epoch = 0
        self._served = 0
        self._order: Optional[np.ndarray] = None
        self._cursor = 0
        self._resuming = False
        self.worker_id = 0
        self.num_workers = 1
        self._last_attempted_record_index = -1

    def _infer_base_root(self) -> Path:
        candidate = self.json_path if self.json_path.is_dir() else self.json_path.parent
        for _ in range(6):
            if (candidate / "frames").exists():
                return candidate
            if candidate.parent == candidate:
                break
            candidate = candidate.parent
        return self.json_path if self.json_path.is_dir() else self.json_path.parent

    def _find_jsonl_files(self, json_root: Path) -> List[Path]:
        if not self.follow_jsonl_directory_symlinks:
            return sorted(json_root.rglob("*.jsonl"))

        files: List[Path] = []
        pending_directories = [json_root]
        visited_directories = set()
        while pending_directories:
            directory = pending_directories.pop()
            resolved_directory = directory.resolve(strict=True)
            if resolved_directory in visited_directories:
                continue
            visited_directories.add(resolved_directory)
            for path in directory.iterdir():
                if path.is_dir():
                    pending_directories.append(path)
                elif path.is_file() and path.suffix.lower() == ".jsonl":
                    files.append(path)
        return sorted(files)

    def _build_index(self) -> int:
        if (
            len(self.json_paths) == 1
            and self.json_path.is_file()
            and self.json_path.suffix.lower() == ".json"
        ):
            if self.source_sampling_enabled:
                raise ValueError("source_sampling requires JSONL input")
            with self.json_path.open("r", encoding="utf-8") as file_obj:
                records = json.load(file_obj)
            if not isinstance(records, list):
                raise ValueError(f"JSON top level must be a list: {self.json_path}")
            self._json_records = records
            return len(records)

        files: List[Path] = []
        file_source_groups: Dict[Path, str] = {}
        for json_path_index, json_path in enumerate(self.json_paths):
            if json_path.is_file() and json_path.suffix.lower() == ".jsonl":
                source_files = [json_path]
            elif json_path.is_dir():
                source_files = self._find_jsonl_files(json_path)
                if self.exclude_jsonl_path_parts:
                    source_files = [
                        path
                        for path in source_files
                        if self.exclude_jsonl_path_parts.isdisjoint(
                            path.relative_to(json_path).parent.parts
                        )
                    ]
            elif json_path.is_file() and json_path.suffix.lower() == ".json":
                raise ValueError(
                    "additional_json_paths can only be combined with JSONL "
                    f"files or directories, not JSON files: {json_path}"
                )
            else:
                raise FileNotFoundError(
                    f"JSON/JSONL path does not exist: {json_path}"
                )
            if not source_files:
                exclusion_note = (
                    " after excluding path parts "
                    f"{sorted(self.exclude_jsonl_path_parts)}"
                    if self.exclude_jsonl_path_parts
                    else ""
                )
                raise FileNotFoundError(
                    f"no JSONL files found under: {json_path}{exclusion_note}"
                )
            for source_file in source_files:
                source_group = self._source_group_for_jsonl(
                    json_path_index, json_path, source_file
                )
                if source_group is None:
                    continue
                if source_file not in file_source_groups:
                    files.append(source_file)
                    file_source_groups[source_file] = source_group

        # Overlapping directory roots must not insert the same record twice.
        self._jsonl_index = []
        index_source_groups: List[str] = []
        for jsonl_path in files:
            with jsonl_path.open("rb") as file_obj:
                while True:
                    offset = file_obj.tell()
                    line = file_obj.readline()
                    if not line:
                        break
                    if line.strip():
                        self._jsonl_index.append((str(jsonl_path), offset))
                        index_source_groups.append(file_source_groups[jsonl_path])
        if self.source_sampling_enabled:
            self._apply_source_sampling(index_source_groups)
        return len(self._jsonl_index)

    def _apply_record_filter(self, total_records: int) -> int:
        """Keep records whose top-level fields match every configured value set."""

        def matches(record: Mapping[str, Any]) -> bool:
            return all(
                record.get(field) in allowed_values
                for field, allowed_values in self.record_filter.items()
            )

        if self._json_records is not None:
            self._json_records = [
                record for record in self._json_records if matches(record)
            ]
            selected_records = len(self._json_records)
        else:
            assert self._jsonl_index is not None
            selected_index: List[Tuple[str, int]] = []
            active_path: Optional[str] = None
            file_obj = None
            try:
                for file_path, offset in self._jsonl_index:
                    if file_path != active_path:
                        if file_obj is not None:
                            file_obj.close()
                        file_obj = open(file_path, "rb")
                        active_path = file_path
                    file_obj.seek(offset)
                    record = json.loads(file_obj.readline().decode("utf-8"))
                    if matches(record):
                        selected_index.append((file_path, offset))
            finally:
                if file_obj is not None:
                    file_obj.close()
            self._jsonl_index = selected_index
            selected_records = len(selected_index)

        if selected_records <= 0:
            raise ValueError(
                f"record_filter selected no records from {total_records}: "
                f"{self.record_filter}"
            )
        logger.info(
            "Filtered Dobot OmTrack records: selected=%d total=%d filter=%s",
            selected_records,
            total_records,
            self.record_filter,
        )
        return selected_records

    def _source_group_for_jsonl(
        self, json_path_index: int, json_root: Path, jsonl_path: Path
    ) -> Optional[str]:
        """Classify one JSONL file and apply the configured Dobot version filter."""
        if not self.source_sampling_enabled:
            return "other"

        relative_parent_parts = tuple(
            part.lower() for part in jsonl_path.relative_to(json_root).parent.parts
        )
        # New raw-RGB manifests keep Dobot and TrackVLA in independent roots,
        # so an additional root can no longer be assumed to mean roaming.  Keep
        # the legacy jsonl_train_manyou convention while classifying all other
        # roots from explicit path components.
        root_parts = tuple(part.lower() for part in json_root.parts)
        classification_parts = set(relative_parent_parts) | set(root_parts[-2:])

        if {
            "dobot_roaming",
            "roaming",
            "jsonl_train_manyou",
            "jsonl_val_manyou",
        } & classification_parts:
            return "dobot_roaming"

        subset_names = {
            part.split("_", 1)[0]
            for part in classification_parts
            if part.split("_", 1)[0] in {"at", "dt", "stt"}
        }
        if len(subset_names) > 1:
            raise ValueError(f"ambiguous TrackVLA subset path: {jsonl_path}")
        if subset_names:
            return f"trackvla_{next(iter(subset_names))}"

        # Preserve the legacy contract: an otherwise unclassified additional
        # root is a Dobot roaming source. Explicit TrackVLA subsets above take
        # precedence for the new raw-source indexes.
        if json_path_index > 0:
            return "dobot_roaming"

        dobot_components = {
            part for part in relative_parent_parts if part in {"dobot", "dobot_v2.0"}
        }
        if dobot_components:
            if len(dobot_components) != 1:
                raise ValueError(
                    f"ambiguous Dobot source path with multiple versions: {jsonl_path}"
                )
            dobot_component = next(iter(dobot_components))
            if (
                self.dobot_data_source != "all"
                and dobot_component != self.dobot_data_source
            ):
                return None
            return "dobot_tracking"

        if "tracking" in classification_parts or "dobot_tracking" in classification_parts:
            return "dobot_tracking"
        return "other"

    def _apply_source_sampling(self, index_source_groups: Sequence[str]) -> None:
        """Apply deterministic per-source record multipliers to the JSONL index."""
        assert self._jsonl_index is not None
        if len(index_source_groups) != len(self._jsonl_index):
            raise RuntimeError("source group index is inconsistent with JSONL index")

        grouped_indices: Dict[str, List[int]] = {
            group: [] for group in SOURCE_SAMPLING_GROUPS
        }
        for index, group in enumerate(index_source_groups):
            grouped_indices[group].append(index)

        self.source_record_counts = {
            group: len(indices) for group, indices in grouped_indices.items()
        }
        if (
            self.dobot_data_source != "all"
            and self.source_record_counts["dobot_tracking"] == 0
        ):
            logger.warning(
                "source_sampling selected Dobot source %r, but no matching "
                "records were found under the primary json_path %s",
                self.dobot_data_source,
                self.json_path,
            )
        base_indices: List[int] = []
        extra_indices: List[int] = []
        self.sampled_source_record_counts = {}
        for group_index, group in enumerate(SOURCE_SAMPLING_GROUPS):
            indices = np.asarray(grouped_indices[group], dtype=np.int64)
            ratio = self.source_sampling_ratios[group]
            logical_count = int(math.ceil(len(indices) * ratio))
            self.sampled_source_record_counts[group] = logical_count
            if logical_count == 0 or len(indices) == 0:
                self.sampled_source_record_counts[group] = 0
                continue

            rng = np.random.default_rng(
                self.source_sampling_seed + 1_000_003 * group_index
            )
            if logical_count < len(indices):
                selected = rng.choice(
                    indices, size=logical_count, replace=False
                ).astype(np.int64)
                base_indices.extend(int(index) for index in selected)
                continue

            base_indices.extend(int(index) for index in indices)
            remaining = logical_count - len(indices)
            while remaining > 0:
                cycle = rng.permutation(indices)
                take = min(remaining, len(cycle))
                extra_indices.extend(int(index) for index in cycle[:take])
                remaining -= take

        # Retain original global ordering for the once-or-less portion. Extra
        # upsampled records are shuffled with the full pool at epoch start.
        selected_indices = sorted(base_indices) + extra_indices
        original_index = self._jsonl_index
        self._jsonl_index = [original_index[index] for index in selected_indices]
        logger.info(
            "Dobot OmTrack source sampling: dobot_data_source=%s raw=%s "
            "ratios=%s sampled=%s",
            self.dobot_data_source,
            self.source_record_counts,
            self.source_sampling_ratios,
            self.sampled_source_record_counts,
        )

    def _select_single_sample(
        self, selector: Mapping[str, Any], total_records: int
    ) -> int:
        """Reduce the backing index to one exact record for deterministic infer."""
        allowed_keys = {"record_index", "current_path"}
        unknown_keys = set(selector) - allowed_keys
        if unknown_keys:
            raise ValueError(
                f"unknown sample_selector keys: {sorted(unknown_keys)}"
            )

        raw_record_index = selector.get("record_index")
        raw_current_path = selector.get("current_path")
        current_path = (
            "" if raw_current_path is None else str(raw_current_path).strip()
        )
        has_record_index = raw_record_index is not None
        has_current_path = bool(current_path)
        if has_record_index == has_current_path:
            raise ValueError(
                "sample_selector requires exactly one of record_index or "
                "current_path"
            )

        if has_record_index:
            if isinstance(raw_record_index, bool):
                raise ValueError("sample_selector record_index must be an integer")
            record_index = int(raw_record_index)
            if record_index != raw_record_index:
                raise ValueError("sample_selector record_index must be an integer")
            if not 0 <= record_index < total_records:
                raise IndexError(
                    f"sample_selector record_index={record_index} is outside "
                    f"[0, {total_records})"
                )
        else:
            matches: List[int] = []
            if self._json_records is not None:
                matches = [
                    index
                    for index, record in enumerate(self._json_records)
                    if str(record.get("current", "")) == current_path
                ]
            else:
                assert self._jsonl_index is not None
                active_path: Optional[str] = None
                file_obj = None
                try:
                    for index, (file_path, offset) in enumerate(self._jsonl_index):
                        if file_path != active_path:
                            if file_obj is not None:
                                file_obj.close()
                            file_obj = open(file_path, "rb")
                            active_path = file_path
                        file_obj.seek(offset)
                        record = json.loads(
                            file_obj.readline().decode("utf-8")
                        )
                        if str(record.get("current", "")) == current_path:
                            matches.append(index)
                            if len(matches) > 1:
                                break
                finally:
                    if file_obj is not None:
                        file_obj.close()
            if not matches:
                raise KeyError(
                    f"sample_selector current_path was not found: {current_path}"
                )
            if len(matches) > 1:
                raise ValueError(
                    f"sample_selector current_path is not unique: {current_path}"
                )
            record_index = matches[0]

        self.selected_original_record_index = record_index
        if self._json_records is not None:
            self._json_records = [self._json_records[record_index]]
        else:
            assert self._jsonl_index is not None
            self._jsonl_index = [self._jsonl_index[record_index]]
        logger.info(
            "Selected one dataset record for inference: original_index=%d, "
            "current_path=%s",
            record_index,
            self._record(0).get("current", ""),
        )
        return 1

    def _record(self, index: int) -> Dict[str, Any]:
        if self._json_records is not None:
            return self._json_records[index]
        assert self._jsonl_index is not None
        file_path, offset = self._jsonl_index[index]
        with open(file_path, "rb") as file_obj:
            file_obj.seek(offset)
            return json.loads(file_obj.readline().decode("utf-8"))

    def _record_source_path(self, index: int) -> Path:
        if self._json_records is not None:
            return self.json_path
        assert self._jsonl_index is not None
        return Path(self._jsonl_index[index][0])

    def _is_trackvla_source(self, source_path: Optional[Path]) -> bool:
        return (
            self.trackvla_time_retarget_enabled
            and source_path is not None
            and self.trackvla_path_component in source_path.parts
        )

    def _trackvla_speed_coefficients_for_source(
        self, source_path: Optional[Path]
    ) -> Tuple[float, ...]:
        if not self.trackvla_speed_coefficients_by_source:
            assert self.trackvla_speed_coefficients is not None
            return self.trackvla_speed_coefficients
        if source_path is None:
            raise ValueError(
                "TrackVLA source path is required for source-specific speed coefficients"
            )
        matches = [
            source
            for source in self.trackvla_speed_coefficients_by_source
            if source in source_path.parts
        ]
        if len(matches) != 1:
            raise ValueError(
                "expected exactly one configured TrackVLA speed source in "
                f"{source_path}; found {matches}"
            )
        return self.trackvla_speed_coefficients_by_source[matches[0]]

    def state_dict(self) -> Dict[str, Any]:
        return {
            "epoch": self._epoch,
            "served": self._served,
            "order": None if self._order is None else self._order.tolist(),
            "cursor": self._cursor,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self._epoch = int(state.get("epoch", 0))
        self._served = int(state.get("served", 0))
        order = state.get("order")
        self._order = None if order is None else np.asarray(order, dtype=np.int64)
        self._cursor = int(state.get("cursor", 0))
        self._resuming = True

    def _reset_order(self) -> None:
        order = np.arange(self.length, dtype=np.int64)
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self._epoch)
            if self.shuffle_block_size is None:
                order = rng.permutation(order)
            else:
                block_size = self.shuffle_block_size
                blocks = [
                    order[start : start + block_size]
                    for start in range(0, len(order), block_size)
                ]
                block_order = rng.permutation(len(blocks))
                order = np.concatenate(
                    [rng.permutation(blocks[index]) for index in block_order]
                )
        rank = int(os.environ.get("RANK", "0"))
        world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
        if rank < 0 or rank >= world_size:
            raise ValueError(f"invalid distributed rank/world size: {rank}/{world_size}")
        total_workers = self.num_workers * world_size
        global_worker_id = rank * self.num_workers + self.worker_id
        if total_workers == 1 or self.worker_shard_batch_size is None:
            self._order = order[global_worker_id::total_workers]
        else:
            batch_size = self.worker_shard_batch_size
            full_batch_count = len(order) // batch_size
            full_end = full_batch_count * batch_size
            full_batches = order[:full_end].reshape(full_batch_count, batch_size)
            worker_batches = full_batches[global_worker_id::total_workers].reshape(-1)
            remainder = order[full_end:]
            remainder_worker = full_batch_count % total_workers
            if global_worker_id == remainder_worker and remainder.size:
                worker_batches = np.concatenate((worker_batches, remainder))
            self._order = worker_batches
        self._cursor = 0

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self.worker_id, self.num_workers = 0, 1
        else:
            self.worker_id, self.num_workers = worker_info.id, worker_info.num_workers
        if self._resuming:
            self._resuming = False
        else:
            self._epoch += 1
            self._served = 0
            self._reset_order()
        return self

    def _resolve_image_path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.base_root / path

    @staticmethod
    def _numeric_stem(path: Path) -> Optional[int]:
        try:
            return int(path.stem)
        except ValueError:
            return None

    @staticmethod
    def _frame_number(path: Path) -> Optional[int]:
        digits = "".join(ch for ch in path.stem if ch.isdigit())
        return int(digits) if digits else None

    @staticmethod
    def _nearest_indices(
        values: np.ndarray,
        targets: np.ndarray,
        max_index: int,
        pad_index: int,
    ) -> List[int]:
        """Return nearest source indices without selecting a future frame."""
        result: List[int] = []
        for target in targets:
            if target < values[0]:
                result.append(pad_index)
                continue
            right = min(int(np.searchsorted(values, target, side="left")), max_index)
            left = max(0, min(right - 1, max_index))
            if abs(float(values[right]) - float(target)) < abs(
                float(values[left]) - float(target)
            ):
                result.append(right)
            else:
                result.append(left)
        return result

    def _timestamp_timeline(self, directory: Path) -> np.ndarray:
        cache_key = str(directory)
        cached = self._timestamp_timeline_cache.get(cache_key)
        if cached is not None:
            return cached
        timestamps = sorted(
            timestamp
            for path in directory.iterdir()
            if path.is_file()
            and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            and (timestamp := self._numeric_stem(path)) is not None
        )
        timeline = np.asarray(timestamps, dtype=np.int64)
        self._timestamp_timeline_cache[cache_key] = timeline
        return timeline

    def _select_timestamped_paths(
        self, current_path: Path, current_timestamp: int
    ) -> Tuple[List[Path], List[float]]:
        assert self._desired_time_offsets is not None
        timeline = self._timestamp_timeline(current_path.parent)
        if timeline.size == 0:
            raise FileNotFoundError(
                f"no timestamp-named images found under {current_path.parent}"
            )
        current_index = int(
            np.searchsorted(timeline, current_timestamp, side="right") - 1
        )
        if current_index < 0:
            raise ValueError(
                f"current timestamp {current_timestamp} predates {current_path.parent}"
            )
        targets = current_timestamp - np.rint(
            self._desired_time_offsets * 1.0e9
        ).astype(np.int64)
        pad_index = current_index if self.history_padding == "current" else 0
        selected_indices = self._nearest_indices(
            timeline, targets, current_index, pad_index
        )
        selected_timestamps = timeline[selected_indices]
        paths = [
            current_path.parent / f"{int(timestamp)}{current_path.suffix}"
            for timestamp in selected_timestamps
        ]
        frame_times = (selected_timestamps - selected_timestamps[0]) / 1.0e9
        return paths, frame_times.astype(np.float64).tolist()

    def _select_indexed_paths(
        self, record: Dict[str, Any], current_path: Path
    ) -> Tuple[List[Path], List[float]]:
        assert self._desired_time_offsets is not None
        history_paths = [
            self._resolve_image_path(value)
            for value in record.get("images", [])[-self.history :]
        ]
        available_paths = history_paths + [current_path]
        frame_numbers = [self._frame_number(path) for path in available_paths]
        if all(number is not None for number in frame_numbers):
            values = np.asarray(frame_numbers, dtype=np.float64)
            targets = values[-1] - self._desired_time_offsets * self.frame_index_hz
            pad_index = len(values) - 1 if self.history_padding == "current" else 0
            selected_indices = self._nearest_indices(
                values, targets, len(values) - 1, pad_index
            )
            selected_values = values[selected_indices]
            frame_times = (selected_values - selected_values[0]) / self.frame_index_hz
            return (
                [available_paths[index] for index in selected_indices],
                frame_times.astype(np.float64).tolist(),
            )

        missing = self.history - len(history_paths)
        if self.history_padding == "current" or not history_paths:
            pad_path = current_path
        else:
            pad_path = history_paths[0]
        slot_paths = [pad_path] * missing + history_paths + [current_path]
        paths = [slot_paths[index] for index in self._fallback_frame_slot_indices]
        assert self.input_hz is not None
        frame_times = (np.arange(self.num_input_frames) / self.input_hz).tolist()
        return paths, frame_times

    def _select_frame_paths(
        self, record: Dict[str, Any]
    ) -> Tuple[List[Path], List[float]]:
        if not self._temporal_sampling_enabled:
            raise RuntimeError(
                "time-aware selection requires num_input_frames and input_hz"
            )
        current_path = self._resolve_image_path(record["current"])
        current_timestamp = self._numeric_stem(current_path)
        if current_timestamp is not None:
            return self._select_timestamped_paths(current_path, current_timestamp)
        return self._select_indexed_paths(record, current_path)

    def _cache_paths(self, image_path: Path, suffix: str) -> List[Path]:
        absolute = (
            image_path if image_path.is_absolute() else self.base_root / image_path
        )
        relative_paths: List[Path] = []
        try:
            relative_paths.append(absolute.relative_to(self.base_root))
        except ValueError:
            pass
        absolute_relative = Path(*absolute.parts[1:]) if absolute.is_absolute() else absolute
        if absolute_relative not in relative_paths:
            relative_paths.append(absolute_relative)
        return [
            self.cache_root / relative.parent / f"{relative.stem}_{suffix}.pt"
            for relative in relative_paths
        ]

    def _get_online_vision_encoder(self) -> Any:
        if self._online_vision_encoder is not None:
            return self._online_vision_encoder
        if not self.online_encode_on_cache_miss:
            raise RuntimeError("online cache-miss encoding is disabled")
        assert self.online_vision_encoder_config is not None
        config = dict(self.online_vision_encoder_config)
        config.pop("type", None)
        worker_info = torch.utils.data.get_worker_info()
        device = str(config.get("device", "cuda"))
        if worker_info is not None and device.startswith("cuda"):
            raise RuntimeError(
                "CUDA online vision encoding cannot run safely inside DataLoader "
                "workers; configure num_workers=0 for this dataloader"
            )
        from vla.omni.ported.projects.vln.model.encoder.omtrack_online_vision_encoder import (
            OmTrackOnlineVisionEncoder,
        )

        logger.info("Initializing online OmTrack vision encoder on %s", device)
        self._online_vision_encoder = OmTrackOnlineVisionEncoder(**config).eval()
        return self._online_vision_encoder

    def _load_online_tokens(
        self, image_path: Path
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        absolute = (
            image_path if image_path.is_absolute() else self.base_root / image_path
        )
        key = str(absolute)
        cached = self._online_token_cache.pop(key, None)
        if cached is not None:
            self._online_token_cache[key] = cached
            return cached
        if not absolute.is_file():
            raise FileNotFoundError(
                f"cannot encode missing source image for cache fallback: {absolute}"
            )

        from PIL import Image

        try:
            with Image.open(absolute) as image:
                rgb = np.asarray(image.convert("RGB"))
        except (OSError, ValueError) as exc:
            raise OSError(
                f"failed to read source image for online cache fallback: {absolute}"
            ) from exc
        encoder = self._get_online_vision_encoder()
        coarse, fine = encoder(rgb)
        coarse = coarse.detach().float().cpu()
        fine = fine.detach().float().cpu()
        expected_coarse = (self.coarse_tokens_per_frame, self.vision_feature_dim)
        expected_fine = (self.fine_tokens_per_frame, self.vision_feature_dim)
        if tuple(coarse.shape) != expected_coarse or tuple(fine.shape) != expected_fine:
            raise ValueError(
                "unexpected online vision-token shapes for "
                f"{absolute}: coarse={tuple(coarse.shape)} != {expected_coarse}, "
                f"fine={tuple(fine.shape)} != {expected_fine}"
            )
        result = (coarse, fine)
        self._online_token_cache[key] = result
        while len(self._online_token_cache) > self.online_token_cache_size:
            self._online_token_cache.popitem(last=False)
        logger.warning(
            "Vision cache miss for %s; generated coarse/fine tokens online "
            "without writing the shared cache",
            absolute,
        )
        return result

    def _load_cached(self, image_path: Path, suffix: str, rows: int) -> torch.Tensor:
        cache_paths = (
            [] if self.cache_root is None else self._cache_paths(image_path, suffix)
        )
        cache_path = next((path for path in cache_paths if path.is_file()), None)
        if cache_path is None:
            if self.online_encode_on_cache_miss:
                coarse, fine = self._load_online_tokens(image_path)
                tokens = coarse if suffix == "vcoarse" else fine
            else:
                if self.cache_root is None:
                    raise RuntimeError(
                        "vision-token cache requested but cache_root is None; "
                        "configure cache_root or enable online cache-miss encoding"
                    )
                raise FileNotFoundError(
                    f"missing {suffix} cache for {image_path}; tried {cache_paths}; "
                    "set online_encode_on_cache_miss=true to generate features "
                    "from the source image"
                )
        else:
            tokens = _load_tokens_file(cache_path)
            if self.cached_feature_slice is not None:
                start, stop = self.cached_feature_slice
                if tokens.ndim != 2 or tokens.size(-1) < stop:
                    raise ValueError(
                        f"cannot apply cached feature slice [{start}:{stop}] "
                        f"to token shape {tuple(tokens.shape)} in {cache_path}"
                    )
                tokens = tokens[:, start:stop]
        expected = (rows, self.vision_feature_dim)
        if self.strict_cache_shapes and tuple(tokens.shape) != expected:
            source = (
                cache_path
                if cache_path is not None
                else f"online encoder for {image_path}"
            )
            raise ValueError(
                f"unexpected token shape in {source}: "
                f"{tuple(tokens.shape)} != {expected}"
            )
        return tokens

    def _load_vision(self, record: Dict[str, Any]) -> Tuple[torch.Tensor, ...]:
        if self._temporal_sampling_enabled:
            selected_paths, _ = self._select_frame_paths(record)
            history_paths = selected_paths[:-1]
            current_path = selected_paths[-1]
            history_frame_count = len(history_paths)
        else:
            current_path = self._resolve_image_path(record["current"])
            history_paths = [
                self._resolve_image_path(value)
                for value in record.get("images", [])[-self.history :]
            ]
            history_frame_count = self.history
        fine = self._load_cached(current_path, "vfine", self.fine_tokens_per_frame)
        fine_tidx = torch.full(
            (fine.size(0),), history_frame_count, dtype=torch.long
        )

        history_tokens = [
            self._load_cached(path, "vcoarse", self.coarse_tokens_per_frame)
            for path in history_paths
        ]
        missing = history_frame_count - len(history_tokens)
        if missing and self.history_padding == "current":
            pad_token = self._load_cached(current_path, "vcoarse", self.coarse_tokens_per_frame)
        elif history_tokens:
            pad_token = history_tokens[0]
        else:
            pad_token = self._load_cached(current_path, "vcoarse", self.coarse_tokens_per_frame)
        frame_tokens = [pad_token] * missing + history_tokens
        coarse = torch.cat(frame_tokens, dim=0)
        coarse_tidx = torch.cat(
            [torch.full((tokens.size(0),), frame, dtype=torch.long) for frame, tokens in enumerate(frame_tokens)],
            dim=0,
        )
        return coarse, coarse_tidx, fine, fine_tidx, current_path

    def _sample_recorded_trajectory(
        self, record: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample recorded t=trajectory_dt... poses on the configured waypoint grid."""
        trajectory = np.asarray(record["trajectory"], dtype=np.float32)
        if trajectory.ndim != 2 or trajectory.shape[1] != self.coord_dim:
            raise ValueError(
                f"unexpected trajectory shape {tuple(trajectory.shape)}; "
                f"expected [steps, {self.coord_dim}]"
            )

        # Source point zero represents t=trajectory_dt rather than the current origin.
        source_indices = (
            np.arange(1, self.num_waypoints + 1, dtype=np.int64) * self.trajectory_stride - 1
        )
        available = source_indices < trajectory.shape[0]
        waypoints = np.full(
            (self.num_waypoints, self.coord_dim),
            self.trajectory_padding_value,
            dtype=np.float32,
        )
        waypoints[available] = trajectory[source_indices[available]]
        finite = np.isfinite(waypoints).all(axis=1)
        available &= finite
        waypoints[~available] = self.trajectory_padding_value
        return waypoints, available

    def _load_target(
        self, record: Dict[str, Any], source_path: Optional[Path] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        available = np.ones(self.num_waypoints, dtype=np.bool_)
        if self.target_mode == "omtrack_official":
            # Upstream train.py / JsonTrackingDataset: explicit waypoints win;
            # otherwise integrate raw actions. A coexisting `trajectory` is ignored.
            # Keep the origin, integrate at record dt (default 0.1), then use
            # round(linspace(0, T - 1, M)). No simulator speed/clock conversion.
            if "waypoints" in record:
                waypoints = np.asarray(record["waypoints"], dtype=np.float32)
            else:
                if "actions" not in record:
                    raise KeyError("OmTrack official targets require 'waypoints' or 'actions'")
                actions = np.asarray(record["actions"], dtype=np.float32)
                if actions.ndim == 1:
                    actions = actions[None, :]
                if actions.ndim != 2 or min(actions.shape) == 0 or not np.isfinite(actions).all():
                    raise ValueError("OmTrack official actions must be a non-empty finite [T, D] array")
                dt = float(record.get("dt", self.default_dt))
                if not np.isfinite(dt) or dt <= 0.0:
                    raise ValueError("OmTrack official action dt must be finite and positive")
                waypoints = _integrate_actions(actions, max(2, self.num_waypoints), dt)
                if self.num_waypoints == 1:
                    # Upstream returns the endpoint for the one-waypoint case.
                    waypoints = waypoints[-1:]
        elif self._is_trackvla_source(source_path):
            if "actions" not in record:
                raise KeyError(
                    "TrackVLA time retargeting requires raw actions in every record"
                )
            waypoints, available = _trackvla_time_retargeted_waypoints(
                np.asarray(record["actions"], dtype=np.float64),
                num_waypoints=self.num_waypoints,
                waypoint_dt=self.waypoint_dt,
                record_dt=1.0 / self.trackvla_record_frequency_hz,
                action_integration_dt=self.trackvla_action_integration_dt,
                speed_coefficients=(
                    self._trackvla_speed_coefficients_for_source(source_path)
                ),
                source_to_target_time_scale=(
                    self.trackvla_source_to_dobot_time_scale
                ),
                padding_value=self.trajectory_padding_value,
                preserve_original_horizon_mask=(
                    self.trackvla_preserve_original_horizon_mask
                ),
                yaw_source_to_target_time_scale=(
                    self.trackvla_yaw_source_to_target_time_scale
                ),
            )
        elif "waypoints" in record:
            waypoints = np.asarray(record["waypoints"], dtype=np.float32)
        elif "trajectory" in record:
            waypoints, available = self._sample_recorded_trajectory(record)
        elif "actions" in record:
            waypoints = _integrate_actions(
                np.asarray(record["actions"], dtype=np.float32),
                self.num_waypoints,
                float(record.get("dt", self.default_dt)),
            )
        else:
            raise KeyError("OmTrack record requires 'waypoints', 'trajectory', or 'actions'")
        expected = (self.num_waypoints, self.coord_dim)
        if tuple(waypoints.shape) != expected:
            raise ValueError(f"unexpected waypoint shape {tuple(waypoints.shape)} != {expected}")

        if "valid_mask" in record:
            mask = np.asarray(record["valid_mask"], dtype=np.bool_)
        elif "valid_idx" in record:
            mask = np.zeros(self.num_waypoints, dtype=np.bool_)
            mask[np.asarray(record["valid_idx"], dtype=np.int64)] = True
        else:
            mask = np.ones(self.num_waypoints, dtype=np.bool_)
        if tuple(mask.shape) != (self.num_waypoints,):
            raise ValueError(f"unexpected waypoint mask shape: {tuple(mask.shape)}")
        mask &= available
        return torch.from_numpy(waypoints), torch.from_numpy(mask)

    def _is_roaming_instruction_source(self, source_path: Path) -> bool:
        return (
            self.roaming_instruction_relabeling_enabled
            and self.roaming_instruction_source_path_component in source_path.parts
        )

    def _should_relabel_roaming_instruction(
        self, record: Mapping[str, Any]
    ) -> bool:
        ratio = self.roaming_instruction_relabeling_ratio
        if ratio <= 0.0:
            return False
        if ratio >= 1.0:
            return True
        identity = str(record.get("current", ""))
        if not identity:
            raise ValueError(
                "roaming instruction relabeling requires record['current'] "
                "as a stable sample identity"
            )
        return (
            _stable_sample_fraction(
                identity, self.roaming_instruction_relabeling_seed
            )
            < ratio
        )

    def _instruction_for_record(
        self,
        record: Mapping[str, Any],
        source_path: Path,
        waypoints: torch.Tensor,
        mask: torch.Tensor,
    ) -> str:
        if self.instruction_override is not None:
            return self.instruction_override
        instruction = str(record.get("instruction", "follow the person"))
        if not self._is_roaming_instruction_source(source_path):
            return instruction
        if not self._should_relabel_roaming_instruction(record):
            return instruction
        intent = classify_roaming_motion_intent(
            waypoints.detach().cpu().numpy(),
            mask.detach().cpu().numpy(),
            stop_max_translation_m=self.roaming_stop_max_translation_m,
            stop_max_abs_yaw_rad=self.roaming_stop_max_abs_yaw_rad,
            straight_max_abs_yaw_rad=self.roaming_straight_max_abs_yaw_rad,
            straight_max_abs_path_bearing_rad=(
                self.roaming_straight_max_abs_path_bearing_rad
            ),
            turn_min_abs_deviation_rad=(
                self.roaming_turn_min_abs_deviation_rad
            ),
        )
        if intent is None:
            return instruction
        return self.roaming_instruction_prompts[intent]

    def _load_next_once(self) -> Dict[str, Any]:
        if self._order is None:
            self._reset_order()
        assert self._order is not None
        if self._cursor >= len(self._order):
            raise StopIteration
        record_index = int(self._order[self._cursor])
        self._last_attempted_record_index = record_index
        self._cursor += 1
        self._served += 1
        source_path = self._record_source_path(record_index)
        record = self._record(record_index)

        coarse, coarse_tidx, fine, fine_tidx, current_path = self._load_vision(record)
        waypoints, mask = self._load_target(record, source_path)
        instruction = self._instruction_for_record(
            record, source_path, waypoints, mask
        )
        yaw_frame_count = (
            self.num_input_frames - 1
            if self._temporal_sampling_enabled
            else self.history
        )
        yaw_history = record.get("yaw_hist")
        if self._temporal_sampling_enabled:
            if yaw_history:
                yaw_history = list(yaw_history)[-yaw_frame_count:]
                pad_value = yaw_history[0]
                yaw_history = [pad_value] * (yaw_frame_count - len(yaw_history)) + yaw_history
            else:
                yaw_history = [0.0] * yaw_frame_count
        elif yaw_history is None:
            yaw_history = [0.0] * yaw_frame_count
        sample: Dict[str, Any] = {
            OmTrackDataKeys.COARSE_TOKENS: coarse,
            OmTrackDataKeys.COARSE_TIME_INDEX: coarse_tidx,
            OmTrackDataKeys.FINE_TOKENS: fine,
            OmTrackDataKeys.FINE_TIME_INDEX: fine_tidx,
            OmTrackDataKeys.YAW_HISTORY: torch.tensor(
                yaw_history, dtype=torch.float32
            ),
            OmTrackDataKeys.YAW_CURRENT: torch.tensor(
                record.get("yaw_curr", 0.0), dtype=torch.float32
            ).view(1),
            OmTrackDataKeys.INSTRUCTION: instruction,
            OmTrackDataKeys.CURRENT_PATH: str(current_path),
            PlannerDataKeys.FUTURE_TRAJECTORY: waypoints,
            PlannerDataKeys.FUTURE_TRAJECTORY_MASK: mask,
        }
        for processor in self.data_processor_list:
            processor(sample, self.mode)
        return sample

    def _load_next(self) -> Dict[str, Any]:
        """Skip unreadable cache records while preserving data-health counters."""
        resample_count = 0
        while True:
            try:
                sample = self._load_next_once()
            except StopIteration:
                raise
            except Exception as exc:
                if not self.resample_on_error:
                    raise
                resample_count += 1
                logger.warning(
                    "%s worker=%d failed to read record=%d "
                    "(resample %d/%d): %s: %s",
                    type(self).__name__,
                    self.worker_id,
                    self._last_attempted_record_index,
                    resample_count,
                    self.max_resample_times,
                    type(exc).__name__,
                    exc,
                )
                if resample_count >= self.max_resample_times:
                    raise RuntimeError(
                        f"{type(self).__name__} exceeded max_resample_times="
                        f"{self.max_resample_times} after consecutive read failures"
                    ) from exc
                continue

            sample[DataMetricKeys.RESAMPLE_COUNT] = resample_count
            sample[DataMetricKeys.READ_ATTEMPT_COUNT] = resample_count + 1
            return sample
