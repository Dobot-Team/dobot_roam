"""Qwen3.5 vision (ViT) model - faithful, dependency-light port of the official
``transformers.models.qwen3_5.modeling_qwen3_5`` vision stack + the
``transformers.vision_utils`` helpers it depends on.

Preserves module names / shapes so an official ``model.visual.*`` state-dict loads
1:1:

* ``Qwen3_5VisionPatchEmbed``  - Conv3d patchifier (temporal x spatial)
* ``pos_embed``                - learned grid position embedding, bilinearly interpolated
* ``Qwen3_5VisionBlock``       - norm1 -> attn -> norm2 -> mlp (linear_fc1/fc2)
* ``Qwen3_5VisionPatchMerger`` - spatial-merge -> LayerNorm -> MLP -> ``out_hidden_size``

The public ``forward(hidden_states, grid_thw)`` matches the official contract:
flattened patches ``[sum(t*h*w), C*t_patch*p*p]`` + ``grid_thw`` -> merged visual
tokens ``[sum(t*h*w)/merge^2, out_hidden_size]``.  Only the eager attention path is
implemented (per-image full attention over ``cu_seqlens`` chunks).
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from vla.omni.ported.projects.vln.model.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
from vla.omni.ported.projects.vln.model.qwen3_5.modeling_text import ACT2FN, rotate_half


# ---------------------------------------------------------------------------
# vision_utils helpers (verbatim port)
# ---------------------------------------------------------------------------
def get_vision_cu_seqlens(grid_thw: torch.Tensor) -> torch.Tensor:
    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0, dtype=torch.int32
    )
    return F.pad(cu_seqlens, (1, 0), value=0)


def get_vision_position_ids(grid_thw: torch.Tensor, spatial_merge_size: int) -> torch.Tensor:
    device = grid_thw.device
    position_ids = []
    for t, h, w in grid_thw.tolist():
        t, h, w = int(t), int(h), int(w)
        hpos_ids, wpos_ids = torch.meshgrid(
            torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij"
        )
        block_shape = (h // spatial_merge_size, spatial_merge_size, w // spatial_merge_size, spatial_merge_size)
        hpos_ids = hpos_ids.reshape(block_shape).transpose(1, 2).flatten()
        wpos_ids = wpos_ids.reshape(block_shape).transpose(1, 2).flatten()
        position_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
    return torch.cat(position_ids, dim=0)


def get_vision_bilinear_indices_and_weights(
    grid_thw: torch.Tensor, num_grid_per_side: int, spatial_merge_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    side = num_grid_per_side
    merge_size = spatial_merge_size
    device = grid_thw.device
    idx_parts = [[] for _ in range(4)]
    weight_parts = [[] for _ in range(4)]

    for t, h, w in grid_thw.tolist():
        t, h, w = int(t), int(h), int(w)
        h_grid = torch.linspace(0, side - 1, h, device=device)
        w_grid = torch.linspace(0, side - 1, w, device=device)
        h_floor = h_grid.int()
        w_floor = w_grid.int()
        h_ceil = (h_floor + 1).clamp(max=side - 1)
        w_ceil = (w_floor + 1).clamp(max=side - 1)
        h_frac = h_grid - h_floor
        w_frac = w_grid - w_floor
        h_floor_offset = h_floor * side
        h_ceil_offset = h_ceil * side
        corner_indices = [
            (h_floor_offset[:, None] + w_floor[None, :]).flatten(),
            (h_floor_offset[:, None] + w_ceil[None, :]).flatten(),
            (h_ceil_offset[:, None] + w_floor[None, :]).flatten(),
            (h_ceil_offset[:, None] + w_ceil[None, :]).flatten(),
        ]
        corner_weights = [
            ((1 - h_frac)[:, None] * (1 - w_frac)[None, :]).flatten(),
            ((1 - h_frac)[:, None] * w_frac[None, :]).flatten(),
            (h_frac[:, None] * (1 - w_frac)[None, :]).flatten(),
            (h_frac[:, None] * w_frac[None, :]).flatten(),
        ]
        h_idx = torch.arange(h, device=device).view(h // merge_size, merge_size)
        w_idx = torch.arange(w, device=device).view(w // merge_size, merge_size)
        reorder = (h_idx[:, :, None, None] * w + w_idx[None, None, :, :]).transpose(1, 2).flatten().repeat(t)
        for i in range(4):
            idx_parts[i].append(corner_indices[i][reorder])
            weight_parts[i].append(corner_weights[i][reorder])

    bilinear_indices = torch.stack([torch.cat(p) for p in idx_parts])
    bilinear_weights = torch.stack([torch.cat(p) for p in weight_parts])
    return bilinear_indices, bilinear_weights


# ---------------------------------------------------------------------------
# vision rotary embedding
# ---------------------------------------------------------------------------
class Qwen3_5VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        return (position_ids.unsqueeze(-1) * self.inv_freq).flatten(1)


def apply_rotary_pos_emb_vision(q, k, cos, sin):
    orig_q_dtype, orig_k_dtype = q.dtype, k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(orig_q_dtype), k_embed.to(orig_k_dtype)


# ---------------------------------------------------------------------------
# vision submodules
# ---------------------------------------------------------------------------
class Qwen3_5VisionMLP(nn.Module):
    def __init__(self, config: Qwen3_5VisionConfig) -> None:
        super().__init__()
        self.linear_fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))


class Qwen3_5VisionPatchEmbed(nn.Module):
    def __init__(self, config: Qwen3_5VisionConfig) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size
        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(self.in_channels, self.embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states


class Qwen3_5VisionPatchMerger(nn.Module):
    def __init__(self, config: Qwen3_5VisionConfig, use_postshuffle_norm: bool = False) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = nn.LayerNorm(self.hidden_size if use_postshuffle_norm else config.hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x.view(-1, self.hidden_size) if self.use_postshuffle_norm else x).view(-1, self.hidden_size)
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


class Qwen3_5VisionAttention(nn.Module):
    def __init__(self, config: Qwen3_5VisionConfig) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.scaling = self.head_dim**-0.5

    def forward(self, hidden_states, cu_seqlens, position_embeddings):
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        # [seq, heads, dim] -> [1, heads, seq, dim]
        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        # full attention within each image (cu_seqlens chunk)
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        q_splits = torch.split(query_states, lengths, dim=2)
        k_splits = torch.split(key_states, lengths, dim=2)
        v_splits = torch.split(value_states, lengths, dim=2)
        outs = []
        for q, k, v in zip(q_splits, k_splits, v_splits):
            attn = torch.matmul(q, k.transpose(2, 3)) * self.scaling
            attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
            outs.append(torch.matmul(attn, v).transpose(1, 2))  # [1, seq_i, heads, dim]
        attn_output = torch.cat(outs, dim=1)
        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        return self.proj(attn_output)


class Qwen3_5VisionBlock(nn.Module):
    def __init__(self, config: Qwen3_5VisionConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen3_5VisionAttention(config=config)
        self.mlp = Qwen3_5VisionMLP(config=config)

    def forward(self, hidden_states, cu_seqlens, position_embeddings):
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), cu_seqlens=cu_seqlens, position_embeddings=position_embeddings
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen3_5VisionModel(nn.Module):
    """Qwen3.5 ViT. ``forward(hidden_states, grid_thw)`` -> merged visual tokens."""

    def __init__(self, config: Qwen3_5VisionConfig) -> None:
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.spatial_merge_unit = config.spatial_merge_size * config.spatial_merge_size

        self.patch_embed = Qwen3_5VisionPatchEmbed(config=config)
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen3_5VisionRotaryEmbedding(head_dim // 2)
        self.blocks = nn.ModuleList([Qwen3_5VisionBlock(config) for _ in range(config.depth)])
        self.merger = Qwen3_5VisionPatchMerger(config=config, use_postshuffle_norm=False)
        self.gradient_checkpointing = False
        self._gradient_checkpointing_kwargs = {"use_reentrant": False}

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None) -> None:
        """Recompute vision blocks during backward to reduce activation memory."""
        self.gradient_checkpointing = True
        self._gradient_checkpointing_kwargs = {
            "use_reentrant": False,
            **(gradient_checkpointing_kwargs or {}),
        }

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        bilinear_indices, bilinear_weights = get_vision_bilinear_indices_and_weights(
            grid_thw, self.num_grid_per_side, self.config.spatial_merge_size
        )
        position_ids = get_vision_position_ids(grid_thw, self.spatial_merge_size)
        cu_seqlens = get_vision_cu_seqlens(grid_thw)

        hidden_states = self.patch_embed(hidden_states)
        pos_embeds = (self.pos_embed(bilinear_indices) * bilinear_weights[:, :, None]).sum(0)
        hidden_states = hidden_states + pos_embeds.to(hidden_states.dtype)
        rotary_pos_emb = self.rotary_pos_emb(position_ids)

        seq_len, _ = hidden_states.size()
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        for blk in self.blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                hidden_states = activation_checkpoint(
                    blk,
                    hidden_states,
                    cu_seqlens,
                    position_embeddings,
                    **self._gradient_checkpointing_kwargs,
                )
            else:
                hidden_states = blk(
                    hidden_states,
                    cu_seqlens=cu_seqlens,
                    position_embeddings=position_embeddings,
                )

        return self.merger(hidden_states)
