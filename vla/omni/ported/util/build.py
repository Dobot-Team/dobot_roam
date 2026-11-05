"""build_from_cfg: recursively instantiate an object tree from a nested config.

Rules (identical in spirit to ai_foundation ``util/build.py``):
  * dict containing ``type``           -> recursively build kwargs, then
                                          ``registry[type](**kwargs)``.
  * dict with ``mark_: BOTTOM``        -> leaf, returned as a plain dict.
  * dict without ``type``              -> recurse into values, return plain dict.
  * list                               -> build each element.
  * scalar                            -> returned as-is.
  * a registered *function* is turned into ``partial(fn, **kwargs)`` when it is
    not directly callable with the given kwargs.

Keys listed in a class attribute ``_opaque_kwargs`` are passed through raw
(not recursively built), matching ai_foundation semantics.
"""

from functools import partial
from types import FunctionType
from typing import Any

from vla.omni.ported.util.registry import Registry

TYPE_KEY = "type"
MARK_KEY = "mark_"
MARK_BOTTOM = "BOTTOM"


def _build_helper(cfg: Any, registry: Registry) -> Any:
    if isinstance(cfg, dict):
        if cfg.get(MARK_KEY, None) == MARK_BOTTOM:
            cfg = dict(cfg)
            cfg.pop(MARK_KEY)
            return cfg

        opaque_keys: set = set()
        peek_type = cfg.get(TYPE_KEY)
        if peek_type is not None and peek_type in registry:
            opaque_keys = set(getattr(registry[peek_type], "_opaque_kwargs", ()))

        arg_dict = {}
        for key, value in cfg.items():
            if key in opaque_keys:
                arg_dict[key] = value
            else:
                arg_dict[key] = _build_helper(value, registry)

        if TYPE_KEY in arg_dict:
            type_name = arg_dict.pop(TYPE_KEY)
            cls_or_fn = registry[type_name]
            try:
                return cls_or_fn(**arg_dict)
            except TypeError:
                if isinstance(cls_or_fn, FunctionType):
                    return partial(cls_or_fn, **arg_dict)
                raise
        return arg_dict

    if isinstance(cfg, list):
        return [_build_helper(element, registry) for element in cfg]

    return cfg


def build_from_cfg(cfg: dict, registry: Registry) -> Any:
    """Build a module (tree) from a nested config using ``registry``."""
    return _build_helper(cfg, registry)
