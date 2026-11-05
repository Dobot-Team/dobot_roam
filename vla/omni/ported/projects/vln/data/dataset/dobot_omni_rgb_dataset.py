"""Five-task RGB dataset preserving Dobot/TrackVLA target time retargeting."""
import logging
import math
from pathlib import Path

import numpy as np
import torch

from vla.omni.ported.constant.data import DataMetricKeys
from vla.omni.ported.projects.vln.constant.omni import TASK_NAMES, SOURCE_NAMES
from vla.omni.ported.projects.vln.data.dataset.dobot_omtrack_rgb_dataset import DobotOmTrackRGBDataset
from vla.omni.ported.projects.vln.data.dataset.omni_sampling import OmniTaskSourceSampler, stable_seed
from vla.omni.ported.projects.vln.data.dataset.data_processor.omni_tasks import DEFAULT_PROCESSORS

logger = logging.getLogger(__name__)


class DobotOmniRGBDataset(DobotOmTrackRGBDataset):
    def __init__(self, task_sampling=None, source_sampling=None, task_processors=None,
                 future_goal_sampling=None, data_processor_list=None, **kwargs):
        self._omni_ready = False
        self._retry_record_index = None
        # Task selection replaces the old random roaming relabel switch. The
        # intent processor uses the SAME classification function and thresholds.
        kwargs["roaming_instruction_relabeling"] = None
        super().__init__(source_sampling=None, data_processor_list=[], **kwargs)
        self.goal_config = {
            "min_future_s": 0.3, "near_max_s": 2.4, "max_future_s": 4.8,
            "near_probability": 0.8, "far_probability": 0.2,
            "near_bin_width_s": 0.3,
        }
        unknown = set(future_goal_sampling or {}) - set(self.goal_config)
        if unknown:
            raise ValueError(f"unknown future_goal_sampling keys: {unknown}")
        self.goal_config.update(future_goal_sampling or {})
        c = self.goal_config
        if not all(np.isfinite(float(v)) for v in c.values()):
            raise ValueError("future goal settings must be finite")
        if not 0 < c["min_future_s"] <= c["near_max_s"] < c["max_future_s"]:
            raise ValueError("require 0 < min_future_s <= near_max_s < max_future_s")
        if c["near_bin_width_s"] <= 0 or min(c["near_probability"], c["far_probability"]) < 0:
            raise ValueError("invalid goal probability/bin width")
        if not np.isclose(c["near_probability"] + c["far_probability"], 1):
            raise ValueError("near/far probabilities must sum to one")
        if self.coord_dim != 3:
            raise ValueError("Omni tasks require x/y/yaw targets")
        self.task_processors = {t: cls() for t, cls in DEFAULT_PROCESSORS.items()}
        if set(task_processors or {}) - set(TASK_NAMES):
            raise ValueError("unknown task processor")
        self.task_processors.update(task_processors or {})
        if not all(callable(p) for p in self.task_processors.values()):
            raise TypeError("task processors must be built callables")
        pools = {s: [] for s in SOURCE_NAMES}
        # Classify manifests explicitly; additional roots are not implicitly roam.
        manifest_sources = {}
        for index in range(self.length):
            path = super()._record_source_path(index)
            if str(path) in manifest_sources and self._json_records is None:
                pools[manifest_sources[str(path)]].append(index)
                continue
            parts = {part.lower() for part in path.parts}
            source = None
            for prefix in ("at", "dt", "stt"):
                if any(p.split("_", 1)[0] == prefix for p in parts):
                    source = "trackvla_" + prefix
            if source is None:
                if "roaming" in parts or "dobot_roaming" in parts:
                    source = "dobot_roaming"
                elif any(p.startswith("tracking") for p in parts):
                    source = "dobot_tracking"
                else:
                    record = super()._record(index)
                    if record.get("raw_format") == "dobot":
                        source = "dobot_roaming" if record.get("task") == "roaming" else "dobot_tracking"
            if source is None:
                raise ValueError(f"cannot identify Omni source for {path}")
            pools[source].append(index)
            manifest_sources[str(path)] = source
        self.sampler = OmniTaskSourceSampler(task_sampling, source_sampling, pools, self.length)
        self.length = self.sampler.length
        self.data_processor_list = [self._process_task] + list(data_processor_list or [])
        self._omni_ready = True
        self._order = None
        logger.info("Omni tasks=%s source_conditionals=%s pool_counts=%s",
                    self.sampler.tasks, self.sampler.conditionals,
                    {k: len(v) for k, v in pools.items()})

    def set_data_epoch(self, epoch):
        """Set in the parent before workers start; every rank uses the same epoch."""
        if not self._resuming:
            self._epoch = int(epoch)

    def _reset_order(self):
        if not self._omni_ready:
            return super()._reset_order()
        import os
        epoch = self._epoch if self.mode == "train" else 0
        self._plan = self.sampler.plan(epoch, shuffle=self.shuffle)
        rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
        workers = max(1, self.num_workers) * world
        worker_id = rank * max(1, self.num_workers) + self.worker_id
        width = workers * (self.worker_shard_batch_size or 1)
        usable = len(self._plan) // width * width
        if not usable:
            raise ValueError("samples_per_epoch is smaller than distributed worker/batch count")
        self._order = np.arange(usable, dtype=np.int64)[worker_id::workers]
        self._cursor = 0

    def load_state_dict(self, state):
        super().load_state_dict(state)
        epoch = self._epoch if self.mode == "train" else 0
        self._plan = self.sampler.plan(epoch, shuffle=self.shuffle)
        if state.get("order") is not None:
            # Trainer publishes rank 0's checkpoint. Recreate THIS rank's shard
            # from the common plan instead of replaying rank 0's order everywhere.
            cursor = self._cursor
            self._reset_order()
            if cursor > len(self._order):
                raise ValueError("resume cursor exceeds current Omni shard length")
            self._cursor = cursor

    def _raw_index(self, index):
        if not self._omni_ready:
            return index
        return int(self._plan[index, 2]) if self._retry_record_index is None else self._retry_record_index

    def _record_source_path(self, index):
        return super()._record_source_path(self._raw_index(index))

    def _record(self, index):
        result = super()._record(self._raw_index(index))
        if self._omni_ready:
            self._active_draw = int(index)
            self._active_index_record = result
        return result

    def _choose_goal(self, record, task, rng):
        if record.get("raw_format") != "dobot" or record.get("task") != "roaming":
            raise ValueError("navigation goals require raw Dobot roaming records")
        data = self._load_raw_dobot_file(Path(record["training_json"]))
        labels = data["labels"]
        origin_index = int(record["label_index"])
        origin = labels[origin_index]["pose"]
        end = min(len(labels), origin_index + int(math.ceil(
            self.goal_config["max_future_s"] / self.trajectory_dt)) + 2)
        subset = labels[origin_index:end]
        timestamps = np.asarray([float(p["timestamp"]) for p in subset])
        # Deduplicate equal timestamps only for image/pose association. Keep the
        # original label offset as the retargeted clock: NEVER rewrite the base GT.
        if np.any(np.diff(timestamps) < 0):
            raise ValueError("future pose timestamps run backwards")
        unique = np.r_[True, np.diff(timestamps) > 0]
        times = timestamps[unique]
        offsets = np.arange(len(subset), dtype=np.float64)[unique]
        poses = np.asarray([[p["pose"][k] for k in ("x", "y", "yaw")] for p in subset], dtype=np.float64)[unique]
        candidates = []
        if task == "point_nav":
            for offset, label in enumerate(subset[1:], 1):
                p = label["pose"]
                candidates.append((offset * self.trajectory_dt,
                                   np.asarray([p["x"], p["y"], p["yaw"]], dtype=np.float64),
                                   None, float(label["timestamp"])-timestamps[0]))
        else:
            if len(times) < 2:
                raise ValueError("image goal requires a future pose interval")
            yaw = np.unwrap(poses[:, 2])
            base = Path(record["training_json"]).parent / self.raw_camera
            # Use each camera's actual timestamp, not the multi-camera frame clock.
            for frame in data["frames"]:
                stored = frame.get(self.raw_camera)
                if not stored:
                    continue
                path = base / Path(str(stored)).name
                try:
                    actual_time = int(path.stem) * 1e-9
                except ValueError:
                    actual_time = float(frame["timestamp"]) * 1e-9
                if not times[0] <= actual_time <= times[-1]:
                    continue
                target_time = float(np.interp(actual_time, times, offsets)) * self.trajectory_dt
                if not self.goal_config["min_future_s"]-1e-7 <= target_time <= self.goal_config["max_future_s"]+1e-7:
                    continue
                if not path.is_file():
                    continue
                pose = np.array([np.interp(actual_time, times, poses[:, 0]),
                                 np.interp(actual_time, times, poses[:, 1]),
                                 np.interp(actual_time, times, yaw)])
                candidates.append((target_time, pose, path, actual_time-timestamps[0]))
        c = self.goal_config
        candidates = [v for v in candidates if c["min_future_s"]-1e-7 <= v[0] <= c["max_future_s"]+1e-7
                      and np.isfinite(v[1]).all()]
        near = [v for v in candidates if v[0] <= c["near_max_s"]+1e-7]
        far = [v for v in candidates if v[0] > c["near_max_s"]+1e-7]
        requested_far = rng.random() < c["far_probability"]
        choices = far if requested_far else near
        fallback = not bool(choices)
        if not choices:
            choices = near if requested_far else far
        if not choices:
            raise ValueError("no valid future navigation goal")
        # Equalize valid near-time bins, then select a real candidate within a bin.
        if choices is near:
            bins = {}
            for value in choices:
                key = int(round((value[0]-c["min_future_s"]) / c["near_bin_width_s"], 7))
                bins.setdefault(key, []).append(value)
            keys = sorted(bins)
            choices = bins[keys[int(rng.integers(len(keys)))]]
        goal_time, world_pose, image_path, source_elapsed = choices[int(rng.integers(len(choices)))]
        x, y = self._world_to_robot(world_pose[0]-origin["x"], world_pose[1]-origin["y"], origin["yaw"])
        goal = dict(time_s=float(goal_time), pose=[x, y, self._normalize_angle(world_pose[2]-origin["yaw"])],
                    source_elapsed_s=float(source_elapsed), interval_fallback=fallback)
        if image_path is not None:
            goal["rgb"] = torch.from_numpy(self._read_frame(image_path))
            goal["image_path"] = str(image_path)
        return goal

    def _process_task(self, sample, mode):
        task_index, source_index, _ = self._plan[self._active_draw]
        task, source = TASK_NAMES[task_index], SOURCE_NAMES[source_index]
        raw_index = self._raw_index(self._active_draw)
        identity = self._jsonl_index[raw_index] if self._jsonl_index is not None else raw_index
        epoch = self._epoch if mode == "train" else 0
        rng = np.random.default_rng(stable_seed(self.sampler.seed, epoch, identity, self._active_draw))
        sample.update(task_id=int(task_index), task_name=task, source_id=int(source_index),
                      source_name=source, sample_id=str(identity), _waypoint_dt=self.waypoint_dt)
        if task in ("point_nav", "image_nav"):
            sample["_future_goal"] = self._choose_goal(self._active_index_record, task, rng)
        self.task_processors[task](sample, mode)
        sample.pop("_future_goal", None)

    def _load_next(self):
        # Retrying keeps the selected task/source, unlike skipping the next draw.
        if self._order is None:
            self._reset_order()
        if self._cursor >= len(self._order):
            raise StopIteration
        cursor, served = self._cursor, self._served
        draw = int(self._order[cursor])
        source = SOURCE_NAMES[self._plan[draw, 1]]
        epoch = self._epoch if self.mode == "train" else 0
        rng = np.random.default_rng(stable_seed(self.sampler.seed, epoch, draw, "retry"))
        try:
            for attempt in range(self.max_resample_times):
                try:
                    sample = super()._load_next_once()
                    sample[DataMetricKeys.RESAMPLE_COUNT] = attempt
                    sample[DataMetricKeys.READ_ATTEMPT_COUNT] = attempt + 1
                    return sample
                except Exception:
                    if not self.resample_on_error or attempt + 1 == self.max_resample_times:
                        raise
                    logger.warning("Omni retry task=%s source=%s draw=%s",
                                   TASK_NAMES[self._plan[draw, 0]], source, draw, exc_info=True)
                    self._cursor, self._served = cursor, served
                    pool = self.sampler.pools[source]
                    self._retry_record_index = int(pool[rng.integers(len(pool))])
        finally:
            self._retry_record_index = None
