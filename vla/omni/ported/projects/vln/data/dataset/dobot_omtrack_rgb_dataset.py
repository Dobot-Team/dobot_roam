"""Dobot OmTrack dataset variant that reads raw RGB frames for online VLM encoding.

Shares the record index, cursor/order statefulness, waypoint target sampling and
``data_processor_list`` mechanism of :class:`DobotOmTrackDataset`, but replaces
the cached DINOv3+SigLIP token loading with raw image loading so the Qwen3.5
vision tower can encode frames online.

``num_input_frames`` and ``input_hz`` define the desired observation times
relative to the current frame:

    [-(num_input_frames - 1) / input_hz, ..., -1 / input_hz, 0]

Dobot frames have nanosecond timestamps in their filenames, so they are selected
from the complete camera directory by nearest timestamp. Frame-index datasets
such as TrackVLA use ``frame_index_hz`` to map the same desired times to frame
numbers. The current frame is always the last input.

The emitted sample dict matches ``TrackVLADataset._load_next`` so
``VLNPromptProcessor`` and ``planner_collate_fn`` can be reused unchanged:

    rgb                    [num_input_frames, img_size, img_size, 3] float32 [0, 255]
    instruction_text       str
    future_trajectory      [num_waypoints, coord_dim]
    future_trajectory_mask [num_waypoints]
    _num_image_tokens / _num_video_tokens / _num_vision_tokens
    _video_grid_t / _tokens_per_video_grid / _input_hz / _temporal_patch_size
"""

import io
import json
import logging
import math
import os
import time
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
import torch
from PIL import Image

from vla.omni.ported.constant.data import DataMetricKeys
from vla.omni.ported.projects.vln.constant.keys import OmTrackDataKeys, PlannerDataKeys
from vla.omni.ported.projects.vln.data.dataset.dobot_omtrack_dataset import DobotOmTrackDataset

DEFAULT_INSTRUCTION = "follow the person"
DEFAULT_ROAMING_INSTRUCTION = "Roam slowly while prioritizing safety."

logger = logging.getLogger(__name__)


class DobotOmTrackRGBDataset(DobotOmTrackDataset):
    """DobotOmTrackDataset that yields raw RGB video instead of cached tokens."""

    def __init__(
        self,
        img_size: int = 256,
        num_input_frames: int = 16,
        input_hz: float = 5.0,
        temporal_patch_size: int = 2,
        frame_index_hz: float = 10.0,
        raw_camera: str = "cam1",
        raw_record_cache_size: int = 16,
        raw_roaming_instruction: str = DEFAULT_ROAMING_INSTRUCTION,
        resample_on_error: bool = False,
        max_resample_times: int = 100,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.img_size = int(img_size)
        self.num_input_frames = int(num_input_frames)
        self.input_hz = float(input_hz)
        self.temporal_patch_size = int(temporal_patch_size)
        self.frame_index_hz = float(frame_index_hz)
        self.raw_camera = str(raw_camera).strip()
        self.raw_record_cache_size = int(raw_record_cache_size)
        self.raw_roaming_instruction = str(raw_roaming_instruction).strip()
        self._raw_dobot_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.resample_on_error = bool(resample_on_error)
        self.max_resample_times = int(max_resample_times)
        self._last_attempted_record_index = -1
        self._timestamp_timeline_cache: Dict[str, np.ndarray] = {}
        self._camera_file_names_cache: Dict[str, frozenset[str]] = {}

        if self.img_size <= 0:
            raise ValueError("img_size must be positive")
        if self.num_input_frames <= 0:
            raise ValueError("num_input_frames must be positive")
        if self.input_hz <= 0 or self.frame_index_hz <= 0:
            raise ValueError("input_hz and frame_index_hz must be positive")
        if self.temporal_patch_size <= 0:
            raise ValueError("temporal_patch_size must be positive")
        if self.max_resample_times <= 0:
            raise ValueError("max_resample_times must be positive")
        if not self.raw_camera or self.raw_record_cache_size <= 0:
            raise ValueError("raw_camera must be non-empty and raw_record_cache_size positive")
        if not self.raw_roaming_instruction:
            raise ValueError("raw_roaming_instruction cannot be empty")
        total_slots = self.history + 1
        self._fallback_frame_slot_indices = np.linspace(
            0, total_slots - 1, self.num_input_frames
        ).round().astype(np.int64).tolist()
        self._desired_time_offsets = np.arange(
            self.num_input_frames - 1, -1, -1, dtype=np.float64
        ) / self.input_hz
        self.input_frame_times = (-self._desired_time_offsets).tolist()
        logger.info(
            "%s sampling %d frames at %.3f Hz; desired times relative to "
            "current frame: %s",
            type(self).__name__,
            self.num_input_frames,
            self.input_hz,
            [round(value, 6) for value in self.input_frame_times],
        )

        # Spatial tokens per Qwen ViT temporal grid after merge (patch=16,
        # merge=2 in the released Qwen3.5-0.8B vision config), mirroring
        # TrackVLADataset's token bookkeeping for VLNPromptProcessor.
        merged = self.img_size // 16 // 2
        self.tokens_per_video_grid = max(1, merged * merged)
        self.video_grid_t = int(math.ceil(self.num_input_frames / self.temporal_patch_size))
        self.num_video_tokens = self.video_grid_t * self.tokens_per_video_grid
        self.num_image_tokens = self.num_input_frames * self.tokens_per_video_grid

    def _read_frame(self, path: Path) -> np.ndarray:
        """Read one frame as float32 [img_size, img_size, 3] in [0, 255]."""
        with Image.open(path) as img:
            img = img.convert("RGB").resize(
                (self.img_size, self.img_size), Image.BILINEAR
            )
            return np.asarray(img, dtype=np.float32)

    @staticmethod
    def _numeric_stem(path: Path) -> Any:
        try:
            return int(path.stem)
        except ValueError:
            return None

    @staticmethod
    def _frame_number(path: Path) -> Any:
        digits = "".join(ch for ch in path.stem if ch.isdigit())
        return int(digits) if digits else None

    @staticmethod
    def _nearest_indices(
        values: np.ndarray,
        targets: np.ndarray,
        max_index: int,
        pad_index: int,
    ) -> List[int]:
        """Return nearest source indices, never looking past the current frame."""
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

    def _camera_file_names(self, directory: Path) -> frozenset[str]:
        key = str(directory)
        cached = self._camera_file_names_cache.get(key)
        if cached is None:
            # Source camera directories are read-only during training. Reuse
            # readdir file types instead of issuing one remote stat per frame.
            with os.scandir(directory) as entries:
                cached = frozenset(entry.name for entry in entries if entry.is_file())
            self._camera_file_names_cache[key] = cached
        return cached

    def _timestamp_timeline(self, directory: Path) -> np.ndarray:
        cache_key = str(directory)
        cached = self._timestamp_timeline_cache.get(cache_key)
        if cached is not None:
            return cached
        timestamps = sorted(
            timestamp
            for name in self._camera_file_names(directory)
            if (path := Path(name)).suffix.lower() in {".jpg", ".jpeg", ".png"}
            and (timestamp := self._numeric_stem(path)) is not None
        )
        timeline = np.asarray(timestamps, dtype=np.int64)
        self._timestamp_timeline_cache[cache_key] = timeline
        return timeline

    def _select_timestamped_paths(
        self, current_path: Path, current_timestamp: int
    ) -> Any:
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
    ) -> Any:
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
        frame_times = (np.arange(self.num_input_frames) / self.input_hz).tolist()
        return paths, frame_times

    def _select_frame_paths(self, record: Dict[str, Any]) -> Any:
        current_path = self._resolve_image_path(record["current"])
        current_timestamp = self._numeric_stem(current_path)
        if current_timestamp is not None:
            return self._select_timestamped_paths(current_path, current_timestamp)
        return self._select_indexed_paths(record, current_path)

    def _load_rgb_frames(self, record: Dict[str, Any]) -> Any:
        """Return sampled RGB video and actual times relative to its first frame."""
        frames_file = str(record.get("frames_file", ""))
        if frames_file.lower().endswith(".jpgs"):
            return self._load_archive_frames(record)
        frame_paths, frame_times = self._select_frame_paths(record)
        frames = [self._read_frame(path) for path in frame_paths]
        return (
            torch.from_numpy(np.stack(frames, axis=0)),
            frame_times,
            str(frame_paths[-1]),
        )

    @staticmethod
    def _world_to_robot(dx: float, dy: float, yaw: float) -> Any:
        return (
            dx * math.cos(yaw) + dy * math.sin(yaw),
            -dx * math.sin(yaw) + dy * math.cos(yaw),
        )

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def _load_raw_dobot_file(self, path: Path) -> Dict[str, Any]:
        key = str(path)
        cached = self._raw_dobot_cache.pop(key, None)
        if cached is not None:
            self._raw_dobot_cache[key] = cached
            return cached
        with path.open("r", encoding="utf-8") as file_obj:
            data = json.load(file_obj)
        self._raw_dobot_cache[key] = data
        while len(self._raw_dobot_cache) > self.raw_record_cache_size:
            self._raw_dobot_cache.popitem(last=False)
        return data

    def _raw_camera_path(
        self, training_json: Path, frame: Mapping[str, Any]
    ) -> Path:
        stored = frame.get(self.raw_camera)
        if not stored:
            raise KeyError(f"raw Dobot frame has no camera {self.raw_camera!r}")
        path = training_json.parent / self.raw_camera / Path(str(stored)).name
        if path.name not in self._camera_file_names(path.parent):
            raise FileNotFoundError(path)
        return path

    def _materialize_dobot_index(self, index_record: Mapping[str, Any]) -> Dict[str, Any]:
        training_json = Path(str(index_record["training_json"]))
        data = self._load_raw_dobot_file(training_json)
        frames = data["frames"]
        labels = data["labels"]
        label_index = int(index_record["label_index"])
        current_frame_index = int(index_record["current_frame_index"])
        current_label = labels[label_index]
        current_pose = current_label["pose"]
        x0 = float(current_pose["x"])
        y0 = float(current_pose["y"])
        yaw0 = float(current_pose["yaw"])

        # Store a dense runtime record only in memory. _load_target then samples
        # it at the config's trajectory_dt/waypoint_dt grid.
        required_steps = self.trajectory_stride * self.num_waypoints
        trajectory = []
        for label in labels[label_index + 1 : label_index + required_steps + 1]:
            pose = label["pose"]
            x, y = self._world_to_robot(
                float(pose["x"]) - x0, float(pose["y"]) - y0, yaw0
            )
            trajectory.append(
                [x, y, self._normalize_angle(float(pose["yaw"]) - yaw0)]
            )

        history_paths = []
        camera_files = self._camera_file_names(training_json.parent / self.raw_camera)
        for index in range(
            max(0, current_frame_index - self.history), current_frame_index
        ):
            stored = frames[index].get(self.raw_camera)
            if not stored:
                continue
            candidate = (
                training_json.parent / self.raw_camera / Path(str(stored)).name
            )
            if candidate.name in camera_files:
                history_paths.append(str(candidate))
        velocity = current_label["velocity"]
        vx, vy = self._world_to_robot(
            float(velocity["vx"]), float(velocity["vy"]), yaw0
        )
        task = str(index_record.get("task", "tracking"))
        instruction = (
            self.raw_roaming_instruction
            if task == "roaming"
            else str(data.get("bag_meta", {}).get("scene_desc", "")).strip()
        )
        return {
            "instruction": instruction or DEFAULT_INSTRUCTION,
            "current": str(
                self._raw_camera_path(training_json, frames[current_frame_index])
            ),
            "images": history_paths,
            "trajectory": trajectory,
            "actions": [vx, vy, float(velocity["omega"])],
            "frames_file": str(training_json.parent),
        }

    @staticmethod
    def _read_indexed_json_object(path: Path, offset: int) -> Dict[str, Any]:
        with path.open("rb") as file_obj:
            file_obj.seek(offset)
            payload = bytearray()
            depth = 0
            in_string = False
            escaped = False
            while True:
                chunk = file_obj.read(1 << 16)
                if not chunk:
                    break
                for byte in chunk:
                    payload.append(byte)
                    char = chr(byte)
                    if in_string:
                        if escaped:
                            escaped = False
                        elif char == "\\":
                            escaped = True
                        elif char == '"':
                            in_string = False
                        continue
                    if char == '"':
                        in_string = True
                    elif char in "[{":
                        depth += 1
                    elif char in "]}":
                        depth -= 1
                        if depth == 0:
                            return json.loads(payload.decode("utf-8"))
            raise ValueError(f"cannot read JSON object at {path}:{offset}")

    def _materialize_trackvla_index(
        self, index_record: Mapping[str, Any]
    ) -> Dict[str, Any]:
        dataset_json = Path(str(index_record["dataset_json"]))
        record = self._read_indexed_json_object(
            dataset_json, int(index_record["byte_offset"])
        )
        frames_file = Path(str(record["frames_file"]))
        if not frames_file.is_absolute():
            frames_file = Path(str(index_record["data_root"])) / frames_file
        record["frames_file"] = str(frames_file)
        return record

    def _materialize_raw_index(self, record: Dict[str, Any]) -> Dict[str, Any]:
        raw_format = record.get("raw_format")
        if raw_format == "dobot":
            return self._materialize_dobot_index(record)
        if raw_format == "trackvla":
            return self._materialize_trackvla_index(record)
        return record

    def _load_archive_frames(self, record: Dict[str, Any]) -> Any:
        """Read TrackVLA history directly from its ``.jpgs`` zip archive."""
        archive_path = Path(str(record["frames_file"]))
        if not archive_path.is_absolute():
            archive_path = self.base_root / archive_path
        current_number = self._frame_number(Path(str(record.get("current", ""))))
        if current_number is None:
            raise ValueError("TrackVLA archive record requires a numbered current frame")

        with zipfile.ZipFile(archive_path) as archive:
            names = sorted(name for name in archive.namelist() if not name.endswith("/"))
            if not names:
                raise ValueError(f"TrackVLA frame archive is empty: {archive_path}")
            frame_numbers = np.asarray(
                [self._frame_number(Path(name)) for name in names], dtype=np.float64
            )
            if not np.isfinite(frame_numbers).all():
                raise ValueError(f"TrackVLA archive has unnumbered frames: {archive_path}")
            current_index = int(
                np.searchsorted(frame_numbers, current_number, side="right") - 1
            )
            current_index = min(max(current_index, 0), len(names) - 1)
            targets = frame_numbers[current_index] - (
                self._desired_time_offsets * self.frame_index_hz
            )
            pad_index = current_index if self.history_padding == "current" else 0
            selected_indices = self._nearest_indices(
                frame_numbers, targets, current_index, pad_index
            )
            selected_numbers = frame_numbers[selected_indices]
            frames = []
            for index in selected_indices:
                with archive.open(names[index]) as file_obj:
                    with Image.open(io.BytesIO(file_obj.read())) as image:
                        image = image.convert("RGB").resize(
                            (self.img_size, self.img_size), Image.BILINEAR
                        )
                        frames.append(np.asarray(image, dtype=np.float32))

        frame_times = (
            (selected_numbers - selected_numbers[0]) / self.frame_index_hz
        ).astype(np.float64).tolist()
        current_ref = f"{archive_path}::{names[current_index]}"
        return torch.from_numpy(np.stack(frames, axis=0)), frame_times, current_ref

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
        load_started = time.perf_counter()
        source_path = self._record_source_path(record_index)
        record = self._materialize_raw_index(self._record(record_index))

        rgb, frame_times, current_path = self._load_rgb_frames(record)
        waypoints, mask = self._load_target(record, source_path)
        instruction = self._instruction_for_record(
            record, source_path, waypoints, mask
        )
        sample: Dict[str, Any] = {
            PlannerDataKeys.RGB: rgb,
            PlannerDataKeys.FUTURE_TRAJECTORY: waypoints,
            PlannerDataKeys.FUTURE_TRAJECTORY_MASK: mask,
            "instruction_text": instruction or DEFAULT_INSTRUCTION,
            OmTrackDataKeys.INSTRUCTION: instruction or DEFAULT_INSTRUCTION,
            OmTrackDataKeys.CURRENT_PATH: current_path,
            "_num_image_tokens": self.num_image_tokens,
            "_num_video_tokens": self.num_video_tokens,
            "_num_vision_tokens": self.num_video_tokens,
            "_video_grid_t": self.video_grid_t,
            "_tokens_per_video_grid": self.tokens_per_video_grid,
            "_input_hz": float(self.input_hz),
            "_temporal_patch_size": int(self.temporal_patch_size),
            "_input_frame_times": torch.as_tensor(frame_times, dtype=torch.float64),
            "_video_grid_timestamps": torch.as_tensor(
                frame_times[:: self.temporal_patch_size],
                dtype=torch.float64,
            ),
        }
        load_time_s = time.perf_counter() - load_started
        process_started = time.perf_counter()
        for processor in self.data_processor_list:
            processor(sample, self.mode)
        sample[DataMetricKeys.LOAD_TIME_S] = load_time_s
        sample[DataMetricKeys.PROCESS_TIME_S] = time.perf_counter() - process_started
        return sample

    def _load_next(self) -> Dict[str, Any]:
        """Load a sample, skipping corrupt records when error resampling is enabled."""
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
                    "DobotOmTrackRGBDataset worker=%d failed to read record=%d "
                    "(resample %d/%d): %s: %s",
                    self.worker_id,
                    self._last_attempted_record_index,
                    resample_count,
                    self.max_resample_times,
                    type(exc).__name__,
                    exc,
                )
                if resample_count >= self.max_resample_times:
                    raise RuntimeError(
                        "DobotOmTrackRGBDataset exceeded max_resample_times="
                        f"{self.max_resample_times} after consecutive read failures"
                    ) from exc
                continue

            sample[DataMetricKeys.RESAMPLE_COUNT] = resample_count
            sample[DataMetricKeys.READ_ATTEMPT_COUNT] = resample_count + 1
            return sample


class DobotOmTrackRawTokenDataset(DobotOmTrackRGBDataset):
    """Materialize canC raw records and encode OmTrack tokens online.

    The raw-index and trajectory code is shared with
    :class:`DobotOmTrackRGBDataset`, while the emitted sample contract matches
    :class:`DobotOmTrackDataset`: the historical frames provide coarse tokens
    and the current frame provides fine tokens. This lets cached-feature
    OmTrack checkpoints run on canC's raw-RGB evaluation index without
    constructing or mutating a persistent vision cache.
    """

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("online_encode_on_cache_miss", True)
        super().__init__(**kwargs)
        if not self.online_encode_on_cache_miss:
            raise ValueError(
                "DobotOmTrackRawTokenDataset requires online_encode_on_cache_miss=True"
            )
        if self.online_vision_encoder_config is None:
            raise ValueError(
                "DobotOmTrackRawTokenDataset requires online_vision_encoder_config"
            )
        # DobotOmTrackRGBDataset owns the explicit raw-video sampling grid. The
        # cached-token parent uses this flag to select that same temporal grid.
        self._temporal_sampling_enabled = True

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

        load_started = time.perf_counter()
        source_path = self._record_source_path(record_index)
        record = self._materialize_raw_index(self._record(record_index))
        coarse, coarse_tidx, fine, fine_tidx, current_path = (
            DobotOmTrackDataset._load_vision(self, record)
        )
        waypoints, mask = self._load_target(record, source_path)
        instruction = self._instruction_for_record(
            record, source_path, waypoints, mask
        )

        history_frame_count = self.num_input_frames - 1
        yaw_history = record.get("yaw_hist")
        if yaw_history:
            yaw_history = list(yaw_history)[-history_frame_count:]
            pad_value = yaw_history[0]
            yaw_history = [pad_value] * (
                history_frame_count - len(yaw_history)
            ) + yaw_history
        else:
            yaw_history = [0.0] * history_frame_count

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
        load_time_s = time.perf_counter() - load_started
        process_started = time.perf_counter()
        for processor in self.data_processor_list:
            processor(sample, self.mode)
        sample[DataMetricKeys.LOAD_TIME_S] = load_time_s
        sample[DataMetricKeys.PROCESS_TIME_S] = (
            time.perf_counter() - process_started
        )
        return sample
