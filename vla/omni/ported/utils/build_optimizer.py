"""build_optimizer: config-driven optimizer factory (ai_foundation ``utils`` style).

Supports weight-decay parameter grouping (bias / LayerNorm / 1-D params are
excluded from decay), mirroring InternNav's ``BaseTrainer.create_optimizer``.
It also supports config-defined module groups with independent learning rates.
"""

import fnmatch
from typing import Any, Dict, Iterable, List, Optional

import torch
from torch.nn import Module
from torch.optim import Optimizer

_OPTIMIZERS = {
    "AdamW": torch.optim.AdamW,
    "Adam": torch.optim.Adam,
    "SGD": torch.optim.SGD,
}


def _grouped_parameters(model: Module, weight_decay: float):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _iter_optimizer_params(params: Any) -> Iterable[torch.nn.Parameter]:
    for item in params:
        if isinstance(item, dict):
            yield from item.get("params", [])
        else:
            yield item


def _configured_parameter_groups(
    model: Module,
    group_configs: List[Dict[str, Any]],
    *,
    default_lr: Optional[float],
    default_weight_decay: float,
    split_weight_decay: bool,
) -> List[Dict[str, Any]]:
    """Build ordered optimizer groups from glob patterns on parameter names.

    Exactly one rule may use ``default: true``.  Non-default rules match with
    shell-style globs from ``patterns``; every trainable parameter must resolve
    to exactly one rule.  The returned order follows the config order so a
    scheduler such as OneCycleLR can use a correspondingly ordered ``max_lr``
    list.
    """
    if not isinstance(group_configs, list) or not group_configs:
        raise ValueError("optimizer.parameter_groups must be a non-empty list")

    rules = []
    seen_names = set()
    default_indices = []
    for index, raw_config in enumerate(group_configs):
        if not isinstance(raw_config, dict):
            raise TypeError("each optimizer.parameter_groups entry must be a mapping")
        config = dict(raw_config)
        name = str(config.pop("name", "")).strip()
        if not name:
            raise ValueError("each optimizer.parameter_groups entry requires a name")
        if name in seen_names:
            raise ValueError(f"duplicate optimizer parameter group name: {name}")
        seen_names.add(name)

        is_default = bool(config.pop("default", False))
        patterns = config.pop("patterns", [])
        if isinstance(patterns, str):
            patterns = [patterns]
        if not isinstance(patterns, list) or not all(
            isinstance(pattern, str) and pattern for pattern in patterns
        ):
            raise TypeError(f"optimizer parameter group {name!r} patterns must be strings")
        if is_default:
            default_indices.append(index)
            if patterns:
                raise ValueError(
                    f"default optimizer parameter group {name!r} must not define patterns"
                )
        elif not patterns:
            raise ValueError(
                f"optimizer parameter group {name!r} requires patterns or default: true"
            )

        lr = config.pop("lr", default_lr)
        if lr is None:
            raise ValueError(
                f"optimizer parameter group {name!r} has no lr and optimizer.lr is unset"
            )
        group_weight_decay = float(
            config.pop("weight_decay", default_weight_decay)
        )
        if config:
            raise TypeError(
                f"unexpected optimizer parameter group {name!r} arguments: "
                f"{sorted(config)}"
            )
        rules.append(
            {
                "name": name,
                "default": is_default,
                "patterns": patterns,
                "lr": float(lr),
                "weight_decay": group_weight_decay,
                "named_params": [],
            }
        )

    if len(default_indices) > 1:
        raise ValueError("optimizer.parameter_groups may contain only one default group")
    default_index = default_indices[0] if default_indices else None

    unmatched = []
    for parameter_name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        matches = [
            index
            for index, rule in enumerate(rules)
            if not rule["default"]
            and any(
                fnmatch.fnmatchcase(parameter_name, pattern)
                for pattern in rule["patterns"]
            )
        ]
        if len(matches) > 1:
            names = [rules[index]["name"] for index in matches]
            raise ValueError(
                f"trainable parameter {parameter_name!r} matches multiple optimizer "
                f"groups: {names}"
            )
        if matches:
            group_index = matches[0]
        elif default_index is not None:
            group_index = default_index
        else:
            unmatched.append(parameter_name)
            continue
        rules[group_index]["named_params"].append((parameter_name, parameter))

    if unmatched:
        preview = ", ".join(unmatched[:10])
        suffix = " ..." if len(unmatched) > 10 else ""
        raise ValueError(
            f"{len(unmatched)} trainable parameters do not match an optimizer group: "
            f"{preview}{suffix}"
        )

    parameter_groups = []
    for rule in rules:
        named_params = rule["named_params"]
        if not named_params:
            raise ValueError(
                f"optimizer parameter group {rule['name']!r} matched no trainable parameters"
            )
        common = {"lr": rule["lr"]}
        if split_weight_decay and rule["weight_decay"] > 0:
            decay, no_decay = [], []
            for parameter_name, parameter in named_params:
                if (
                    parameter.ndim <= 1
                    or parameter_name.endswith(".bias")
                    or "norm" in parameter_name.lower()
                ):
                    no_decay.append(parameter)
                else:
                    decay.append(parameter)
            if decay:
                parameter_groups.append(
                    {
                        **common,
                        "params": decay,
                        "weight_decay": rule["weight_decay"],
                        "group_name": f"{rule['name']}/decay",
                    }
                )
            if no_decay:
                parameter_groups.append(
                    {
                        **common,
                        "params": no_decay,
                        "weight_decay": 0.0,
                        "group_name": f"{rule['name']}/no_decay",
                    }
                )
        else:
            parameter_groups.append(
                {
                    **common,
                    "params": [parameter for _, parameter in named_params],
                    "weight_decay": rule["weight_decay"],
                    "group_name": rule["name"],
                }
            )
    return parameter_groups


def build_optimizer(cfg: Dict[str, Any], model: Module) -> Optimizer:
    cfg = dict(cfg)
    opt_type = cfg.pop("type")
    if opt_type not in _OPTIMIZERS:
        raise KeyError(f"Unknown optimizer type '{opt_type}'. Available: {list(_OPTIMIZERS)}")
    if "betas" in cfg and isinstance(cfg["betas"], list):
        cfg["betas"] = tuple(cfg["betas"])
    weight_decay = cfg.pop("weight_decay", 0.0)
    use_param_groups = cfg.pop("use_param_groups", True)
    configured_groups = cfg.pop("parameter_groups", None)
    if configured_groups is not None:
        params = _configured_parameter_groups(
            model,
            configured_groups,
            default_lr=cfg.get("lr"),
            default_weight_decay=weight_decay,
            split_weight_decay=use_param_groups,
        )
    elif use_param_groups and weight_decay > 0:
        params = _grouped_parameters(model, weight_decay)
    else:
        params = [p for p in model.parameters() if p.requires_grad]
        cfg["weight_decay"] = weight_decay

    # torch AdamW fused kernels require CUDA params. Keep the config portable for
    # CPU smoke tests and non-CUDA validation by falling back to unfused AdamW.
    if cfg.get("fused") is True and opt_type == "AdamW":
        if not any(param.is_cuda for param in _iter_optimizer_params(params)):
            cfg["fused"] = False

    return _OPTIMIZERS[opt_type](params, **cfg)
