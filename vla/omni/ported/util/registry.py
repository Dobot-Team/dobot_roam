"""Registry: name -> class/function mapping (mmcv / ai_foundation style).

This is the backbone of the config-driven architecture. Every pluggable
component (dataset, processor, model arch, head, loss, evaluator, ...) is
registered into a ``Registry`` and later instantiated from a YAML config by
``build_from_cfg``.
"""

from functools import partial
from typing import Any, Callable, Dict, Optional


class Registry:
    def __init__(self, name: str):
        self._name = name
        self._module_dict: Dict[str, object] = {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self._name}, items={list(self._module_dict.keys())})"

    @property
    def name(self) -> str:
        return self._name

    @property
    def module_dict(self) -> Dict[str, object]:
        return self._module_dict

    def __getitem__(self, name: str) -> Any:
        if name not in self._module_dict:
            raise KeyError(f"'{name}' is not registered in registry '{self._name}'. "
                           f"Available: {list(self._module_dict.keys())}")
        return self._module_dict[name]

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._module_dict

    def _register_module(
        self, module: Any, override: bool = False, module_name: Optional[str] = None
    ) -> None:
        if module_name is None:
            module_name = module.__name__
        if not override and module_name in self._module_dict:
            raise KeyError(f"{module_name} is already registered in {self.name}")
        self._module_dict[module_name] = module

    def register_module(
        self,
        module: Optional[Any] = None,
        override: bool = False,
        module_name: Optional[str] = None,
    ) -> Optional[Callable]:
        """Usable both as ``REG.register_module(MyClass)`` and as a decorator
        ``@REG.register_module()``."""
        if module is None:
            return partial(self.register_module, override=override, module_name=module_name)
        self._register_module(module, override=override, module_name=module_name)
        return module

    def merge_with(self, other: "Registry", override: bool = False) -> None:
        for k, v in other.module_dict.items():
            if k in self._module_dict and not override:
                continue
            self._module_dict[k] = v
