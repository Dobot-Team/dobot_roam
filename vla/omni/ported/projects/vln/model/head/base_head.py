"""BaseHead: task head base class (ai_foundation ``model/head/base_head.py``)."""

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch.nn as nn


class BaseHead(nn.Module, ABC):
    @abstractmethod
    def select_input(self, features: Any, input_dict: Dict[str, Any]) -> Any:
        """Select the tensor(s) this head consumes from the model features."""
        raise NotImplementedError

    @abstractmethod
    def forward(self, features: Any, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError
