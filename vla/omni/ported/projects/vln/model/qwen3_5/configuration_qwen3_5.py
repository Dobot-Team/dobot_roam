"""Qwen3.5 configuration - faithful, dependency-light port of the official
``transformers.models.qwen3_5.configuration_qwen3_5`` dataclasses.

Only ``torch``/stdlib are used (no ``transformers``) so the config can live inside
this framework. Field names, defaults and the derived ``layer_types`` pattern are
kept identical to the official ``Qwen3_5TextConfig`` / ``Qwen3_5VisionConfig`` /
``Qwen3_5Config`` so an official checkpoint's ``config.json`` loads verbatim via
``from_dict`` and the resulting module tree matches the official state-dict keys.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def _default_rope() -> Dict[str, Any]:
    return {
        "rope_type": "default",
        "rope_theta": 1e7,
        "partial_rotary_factor": 0.25,
        "mrope_interleaved": True,
        "mrope_section": [11, 11, 10],
    }


@dataclass
class Qwen3_5TextConfig:
    """Text (LLM) config - mirrors official ``Qwen3_5TextConfig``."""

    vocab_size: int = 248320
    hidden_size: int = 1024
    intermediate_size: int = 3584
    num_hidden_layers: int = 24
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    hidden_act: str = "silu"
    max_position_embeddings: int = 262144
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = True
    rope_parameters: Optional[Dict[str, Any]] = None
    attention_bias: bool = False
    attention_dropout: float = 0.0
    head_dim: int = 256
    # gated (Qwen3-Next style) full-attention output gate
    attn_output_gate: bool = True
    # linear-attention (GatedDeltaNet) params
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 16
    # hybrid layer schedule
    full_attention_interval: int = 4
    layer_types: Optional[List[str]] = None
    mlp_only_layers: List[int] = field(default_factory=list)
    pad_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = 248044

    def __post_init__(self) -> None:
        if self.rope_parameters is None:
            self.rope_parameters = _default_rope()
        else:
            merged = _default_rope()
            merged.update(self.rope_parameters)
            self.rope_parameters = merged
        if self.layer_types is None:
            # every ``full_attention_interval``-th layer is full attention (1-indexed),
            # identical to the official __post_init__.
            self.layer_types = [
                "linear_attention"
                if bool((i + 1) % self.full_attention_interval)
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Qwen3_5TextConfig":
        known = {f: d[f] for f in cls.__dataclass_fields__ if f in d}
        return cls(**known)


@dataclass
class Qwen3_5VisionConfig:
    """Vision (ViT) config - mirrors official ``Qwen3_5VisionConfig``."""

    depth: int = 12
    hidden_size: int = 768
    hidden_act: str = "gelu_pytorch_tanh"
    intermediate_size: int = 3072
    num_heads: int = 12
    in_channels: int = 3
    patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    out_hidden_size: int = 1024
    num_position_embeddings: int = 2304
    initializer_range: float = 0.02

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Qwen3_5VisionConfig":
        known = {f: d[f] for f in cls.__dataclass_fields__ if f in d}
        return cls(**known)


@dataclass
class Qwen3_5Config:
    """Top-level VLM config - mirrors official ``Qwen3_5Config``."""

    text_config: Any = None
    vision_config: Any = None
    image_token_id: int = 248056
    video_token_id: int = 248057
    vision_start_token_id: int = 248053
    vision_end_token_id: int = 248054
    tie_word_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.vision_config is None:
            self.vision_config = Qwen3_5VisionConfig()
        elif isinstance(self.vision_config, dict):
            self.vision_config = Qwen3_5VisionConfig.from_dict(self.vision_config)
        if self.text_config is None:
            self.text_config = Qwen3_5TextConfig()
        elif isinstance(self.text_config, dict):
            self.text_config = Qwen3_5TextConfig.from_dict(self.text_config)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Qwen3_5Config":
        known = {f: d[f] for f in cls.__dataclass_fields__ if f in d}
        return cls(**known)
