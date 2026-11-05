"""Config loading / saving utilities (ai_foundation ``config/util.py`` style).

Supports a top-level ``_VARS`` section for defining source parameters and
computed values.  Any string value in the config matching ``${expr}`` is
resolved against ``_VARS`` (with simple arithmetic / int / round / float).
Resolution is recursive: a ``_VARS`` entry may reference earlier entries.
"""

import copy
import logging
import math
import os
import re
from typing import Any, Dict, List

import yaml

logger = logging.getLogger(__name__)

_VAR_PATTERN = re.compile(r"^\$\{(.+)\}$")

def first_existing_path(*paths: str) -> str:
    """Select a model asset mount explicitly listed by a portable config.

    Keep the first candidate when none exists, so offline config inspection still
    works and the model/tokenizer loader retains its normal missing-file error.
    Configurations without this expression retain their original paths.
    """
    if not paths or any(not isinstance(path, str) or not os.path.isabs(path) for path in paths):
        raise ValueError("first_existing_path requires one or more absolute paths")
    return next((path for path in paths if os.path.exists(path)), paths[0])


_SAFE_BUILTINS = {"int": int, "float": float, "round": round, "abs": abs,
                  "min": min, "max": max, "ceil": math.ceil, "floor": math.floor,
                  "first_existing_path": first_existing_path}


def _eval_expr(expr: str, namespace: Dict[str, Any]) -> Any:
    """Safely evaluate an arithmetic expression against a variable namespace."""
    code = compile(expr, "<config_var>", "eval")
    for name in code.co_names:
        if name not in namespace and name not in _SAFE_BUILTINS:
            raise NameError(f"config _VARS: unknown name '{name}' in expression '${{{expr}}}'")
    return eval(code, {"__builtins__": {}}, {**_SAFE_BUILTINS, **namespace})  # noqa: S307


def _resolve_vars_section(vars_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve ``_VARS`` entries in order (later entries may reference earlier)."""
    resolved: Dict[str, Any] = {}
    for key, value in vars_dict.items():
        if isinstance(value, str):
            m = _VAR_PATTERN.match(value)
            if m:
                resolved[key] = _eval_expr(m.group(1), resolved)
            else:
                resolved[key] = value
        else:
            resolved[key] = value
    return resolved


def _interpolate(obj: Any, namespace: Dict[str, Any]) -> Any:
    """Recursively replace ``${expr}`` strings in a config tree."""
    if isinstance(obj, str):
        m = _VAR_PATTERN.match(obj)
        if m:
            return _eval_expr(m.group(1), namespace)
        return obj
    if isinstance(obj, dict):
        return {k: _interpolate(v, namespace) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_interpolate(v, namespace) for v in obj]
    return obj


def resolve_config_vars(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve ``_VARS`` and interpolate ``${expr}`` references throughout."""
    config = copy.deepcopy(config_dict)
    vars_section = config.pop("_VARS", None)
    if vars_section is None:
        return config
    resolved_vars = _resolve_vars_section(vars_section)
    config = _interpolate(config, resolved_vars)
    return config


def load_config_dict(config_path: str) -> Dict[str, Any]:
    """Load a YAML config file into a plain dict, resolving ``_VARS``."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config {config_path} does not exist!")
    with open(config_path) as f:
        config_dict: Dict[str, Any] = yaml.load(f, Loader=yaml.SafeLoader)
    config_dict = resolve_config_vars(config_dict)
    logger.info("Loaded config %s", config_path)
    return config_dict


def apply_overrides(config_dict: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """Apply ``exec``-style overrides, e.g.
    --override "config['FLOW']['trainer']['max_steps'] = 10"."""
    config = copy.deepcopy(config_dict)
    for override in overrides:
        exec(override, {"config": config})  # noqa: S102 - intentional config override
    return config


def save_config_to_job_dir(config_dict: Dict[str, Any], job_dir: str) -> None:
    """Persist the effective config into the job dir for reproducibility."""
    if not job_dir:
        return
    os.makedirs(job_dir, exist_ok=True)
    out_path = os.path.join(job_dir, "config.yaml")
    with open(out_path, "w") as f:
        yaml.safe_dump(config_dict, f, sort_keys=False, allow_unicode=True)
    logger.info("Saved effective config to %s", out_path)
