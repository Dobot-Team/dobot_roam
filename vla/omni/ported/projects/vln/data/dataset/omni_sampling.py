"""Deterministic task/source quotas, independent of raw source sizes."""
import hashlib
import numpy as np
from vla.omni.ported.projects.vln.constant.omni import TASK_NAMES, SOURCE_NAMES, TASK_SOURCES


def stable_seed(*values):
    payload = "\0".join(map(str, values)).encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def normalize(weights, allowed, label):
    unknown = set(weights) - set(allowed)
    if unknown:
        raise ValueError(f"unknown {label}: {sorted(unknown)}")
    result = {key: float(weights.get(key, 0)) for key in allowed}
    if not all(np.isfinite(v) and v >= 0 for v in result.values()) or sum(result.values()) <= 0:
        raise ValueError(f"{label} weights must be finite, nonnegative and have a positive sum")
    total = sum(result.values())
    return {k: v / total for k, v in result.items()}


def allocate(total, probabilities):
    expected = np.asarray(probabilities, dtype=np.float64) * total
    counts = np.floor(expected).astype(np.int64)
    for i in np.argsort(-(expected - counts), kind="stable")[:total-int(counts.sum())]:
        counts[i] += 1
    return counts


class OmniTaskSourceSampler:
    def __init__(self, task_sampling, source_sampling, pools, default_length):
        task_cfg = dict(task_sampling or {})
        self.seed = int(task_cfg.get("seed", 42))
        self.tasks = normalize(task_cfg.get("ratios", {t: 1 for t in TASK_NAMES}),
                               TASK_NAMES, "task")
        requested = task_cfg.get("samples_per_epoch", "auto")
        self.length = default_length if requested in (None, "auto") else int(requested)
        if self.length <= 0:
            raise ValueError("samples_per_epoch must be positive")
        source_cfg = dict(source_sampling or {})
        self.conditionals = {}
        mode = source_cfg.get("mode", "conditional")
        if mode == "conditional":
            by_task = source_cfg.get("weights_by_task", {})
            if set(by_task) - set(TASK_NAMES):
                raise ValueError("unknown task in weights_by_task")
            for task, probability in self.tasks.items():
                if probability:
                    self.conditionals[task] = normalize(
                        by_task.get(task, {s: 1 for s in TASK_SOURCES[task]}),
                        TASK_SOURCES[task], f"{task} source")
        elif mode == "global_marginals":
            if "weights_by_task" in source_cfg:
                raise ValueError("global_marginals cannot combine weights_by_task")
            sources = normalize(source_cfg.get("ratios", {}), SOURCE_NAMES, "source")
            if not np.isclose(sources["dobot_roaming"], 1-self.tasks["tracking"], atol=1e-9):
                raise ValueError("incompatible marginals: dobot_roaming must equal 1 - tracking")
            for task, probability in self.tasks.items():
                if probability:
                    self.conditionals[task] = (
                        {s: sources[s] / probability for s in TASK_SOURCES[task]}
                        if task == "tracking" else {"dobot_roaming": 1.0})
        else:
            raise ValueError("source_sampling mode must be conditional or global_marginals")
        self.pools = {s: np.asarray(v, dtype=np.int64) for s, v in pools.items()}
        for task, sources in self.conditionals.items():
            for source, weight in sources.items():
                if weight and not len(self.pools.get(source, [])):
                    raise ValueError(f"empty data pool for {task}/{source}")

    def plan(self, epoch, shuffle=True):
        rng = np.random.default_rng(stable_seed(self.seed, epoch))
        # Array rows: task index, source index, raw record index. Draw identity is
        # its global row index AFTER the final shuffle, before rank/worker slicing.
        result = np.empty((self.length, 3), dtype=np.int64)
        cursor = 0
        task_counts = allocate(self.length, list(self.tasks.values()))
        for ti, (task, count) in enumerate(zip(TASK_NAMES, task_counts)):
            if not count:
                continue
            sources = self.conditionals[task]
            source_counts = allocate(int(count), list(sources.values()))
            for source, n in zip(sources, source_counts):
                if not n:
                    continue
                pool = self.pools[source]
                # Sample without replacement per pass, cycle only when needed.
                target = result[cursor:cursor+n]
                target[:, 0] = ti
                target[:, 1] = SOURCE_NAMES.index(source)
                filled = 0
                while filled < n:
                    cycle = rng.permutation(pool) if shuffle else pool
                    take = min(n-filled, len(cycle))
                    target[filled:filled+take, 2] = cycle[:take]
                    filled += take
                cursor += n
        # Interleave tasks even for validation, deterministically. Validation has
        # a fixed epoch so repeated evaluation uses identical goals and order.
        return result[rng.permutation(len(result))]
