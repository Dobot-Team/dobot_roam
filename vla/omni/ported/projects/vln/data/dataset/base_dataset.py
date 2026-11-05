"""Dataset base classes for the VLN project.

* ``BaseVLNDataset``  - streaming trajectory dataset (InternNav ``BaseDataset`` role).
* ``StatefulDataset`` - mixin adding ``state_dict``/``load_state_dict`` so the
  trainer can checkpoint and resume the exact data-reading progress (feature 1).

Concrete datasets implement ``_load_next`` returning a 4-tuple
``(obs_dict, prev_actions, gt_actions, inflection_weights)``.
"""

import random
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import IterableDataset


class ObservationsDict(dict):
    def pin_memory(self):
        for k, v in self.items():
            self[k] = v.pin_memory()
        return self


def block_shuffle(lst: List, block_size: int) -> List:
    blocks = [lst[i : i + block_size] for i in range(0, len(lst), block_size)]
    random.shuffle(blocks)
    return [ele for block in blocks for ele in block]


class StatefulDataset:
    """Mixin: datasets that can report / restore their reading progress.

    ``is_stateful(ds)`` lets the fetcher decide whether an *exact* resume is
    possible (single-worker) or a fast-forward replay is needed.
    """

    def state_dict(self) -> Dict[str, Any]:
        raise NotImplementedError

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        raise NotImplementedError


def is_stateful(obj: Any) -> bool:
    return isinstance(obj, StatefulDataset)


class BaseVLNDataset(IterableDataset):
    def __init__(self, length: int, preload_size: int = 100) -> None:
        super().__init__()
        self.length = length
        self.preload_size = preload_size

    def _load_next(self):
        raise NotImplementedError

    def __next__(self):
        return self._load_next()

    def __len__(self) -> int:
        return self.length

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            start, end = 0, self.length
        else:
            per_worker = int(np.ceil(self.length / worker_info.num_workers))
            start = per_worker * worker_info.id
            end = min(start + per_worker, self.length)
        self.load_ordering = list(reversed(block_shuffle(list(range(start, end)), self.preload_size)))
        return self
