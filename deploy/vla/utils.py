"""Shared helpers for VLA deployment scripts."""
from collections.abc import Mapping
from typing import Any, Dict

import torch

from vla.config import (
    ExecutionStage,
    ModelExtensionContext,
    ModelExtensionsConfig,
)


def load_checkpoint(path: str, map_location: str = "cpu") -> Dict[str, Any]:
    """Load a PyTorch checkpoint safely.

    Tries ``weights_only=True`` first to avoid arbitrary code execution; falls
    back to ``weights_only=False`` only when the checkpoint contains objects
    that cannot be deserialized under the safe mode (common for Lightning
    checkpoints that store hyper-parameters / callbacks as plain Python
    objects).
    """
    try:
        checkpoint = torch.load(
            path,
            map_location=map_location,
            weights_only=True,
        )
    except Exception as e:  # noqa: BLE001
        print(
            f"[Warn] Safe checkpoint load failed ({e}); "
            "falling back to weights_only=False."
        )
        checkpoint = torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )

    hparams = (
        checkpoint.get("hyper_parameters", {})
        if isinstance(checkpoint, Mapping)
        else {}
    )
    if not isinstance(hparams, Mapping):
        raise TypeError("Checkpoint hyper_parameters must be a mapping")
    model_extensions = ModelExtensionsConfig.resolve_checkpoint(
        model_config={},
        checkpoint_hparams=hparams,
    )
    state_dict = (
        checkpoint.get("state_dict", checkpoint)
        if isinstance(checkpoint, Mapping)
        else {}
    )
    has_compressor_weights = isinstance(state_dict, Mapping) and any(
        str(key).startswith(
            ("model.history_compressor.", "history_compressor.")
        )
        for key in state_dict
    )
    if has_compressor_weights:
        raise RuntimeError(
            "History-token compression is currently supported only by the "
            "PyTorch TrackVLA/InternVL3 path; ONNX and TensorRT deployment are "
            "not implemented."
        )
    try:
        model_extensions.validate_for(
            ModelExtensionContext(stage=ExecutionStage.EXPORT)
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    return checkpoint
