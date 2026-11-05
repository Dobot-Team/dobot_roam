"""BaseDataProcessor: per-sample transform (ai_foundation ``BaseDataProcessor`` role).

A data processor is a callable that mutates a sample ``data_dict`` in place
(augmentation, target generation, normalization, tokenization...). Datasets run
an ordered ``data_processor_list`` on every sample, so preprocessing is fully
described in the config and reusable across datasets (feature 2).

``mode`` is one of "train" / "val" / "test" so processors can behave differently
(e.g. augmentation only during training).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseDataProcessor(ABC):
    @abstractmethod
    def __call__(self, data_dict: Dict[str, Any], mode: str = "train") -> None:
        """Mutate ``data_dict`` in place."""
        raise NotImplementedError
