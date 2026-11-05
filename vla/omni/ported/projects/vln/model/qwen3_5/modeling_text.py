"""Qwen3.5 text (LLM) model - faithful, dependency-light port of the official
``transformers.models.qwen3_5.modeling_qwen3_5`` text stack.

Ported verbatim (module names, math and parameter shapes preserved so an official
``model.language_model.*`` state-dict loads 1:1):

* ``Qwen3_5RMSNorm`` / ``Qwen3_5RMSNormGated`` - (1 + weight) RMSNorm & gated variant
* ``Qwen3_5TextRotaryEmbedding`` - interleaved multimodal RoPE (partial rotary)
* ``Qwen3_5GatedDeltaNet`` - linear-attention token mixer (torch fallback path)
* ``Qwen3_5Attention`` - gated full attention with per-head q/k RMSNorm
* ``Qwen3_5MLP`` - SwiGLU
* ``Qwen3_5DecoderLayer`` - hybrid (linear / full) block
* ``Qwen3_5TextModel`` - the LLM; exposes ``embed_tokens`` and ``config`` and
  returns the last hidden state tensor so this project's ``LLMDecoder`` can drive it.

Only the eager/torch execution path is implemented (training-forward, no KV/conv
cache, no fused FLA kernels), which is all the planner needs. The chunked
gated-delta-rule kernel is copied verbatim from the reference.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from vla.omni.ported.projects.vln.model.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

ACT2FN = {
    "silu": F.silu,
    "gelu": F.gelu,
    "gelu_pytorch_tanh": lambda x: F.gelu(x, approximate="tanh"),
    "relu": F.relu,
}


# ---------------------------------------------------------------------------
# Norms
# ---------------------------------------------------------------------------
class Qwen3_5RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class Qwen3_5RMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor, gate: Optional[torch.Tensor] = None) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        if gate is not None:
            hidden_states = hidden_states * F.silu(gate.to(torch.float32))
        return hidden_states.to(input_dtype)


# ---------------------------------------------------------------------------
# Rotary embedding (interleaved MRoPE, partial rotary)
# ---------------------------------------------------------------------------
class Qwen3_5TextRotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig, device=None) -> None:
        super().__init__()
        self.config = config
        base = config.rope_parameters["rope_theta"]
        partial_rotary_factor = config.rope_parameters.get("partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        dim = int(head_dim * partial_rotary_factor)
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        self.attention_scaling = 1.0
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.mrope_section = config.rope_parameters.get("mrope_section", [11, 11, 10])

    def apply_interleaved_mrope(self, freqs: torch.Tensor, mrope_section) -> torch.Tensor:
        """Reorganize [T..H..W..] chunked freqs into interleaved [THWTHW...TT]."""
        freqs_t = freqs[0]
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        # position_ids: (3, bs, L) for T/H/W, or (bs, L) -> expanded to 3 identical axes.
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        inv_freq_expanded = (
            self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1).to(x.device)
        )
        position_ids_expanded = position_ids[:, :, None, :].float()
        device_type = x.device.type if x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim: int = 1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(module, query, key, value, attention_mask, scaling, dropout: float = 0.0):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


# ---------------------------------------------------------------------------
# Full attention (gated)
# ---------------------------------------------------------------------------
class Qwen3_5Attention(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.attn_output_gate = getattr(config, "attn_output_gate", True)
        q_out = config.num_attention_heads * self.head_dim * (2 if self.attn_output_gate else 1)
        self.q_proj = nn.Linear(config.hidden_size, q_out, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        self.q_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings, attention_mask=None):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        if self.attn_output_gate:
            query_states, gate = torch.chunk(
                self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
            )
            gate = gate.reshape(*input_shape, -1)
        else:
            query_states = self.q_proj(hidden_states).view(hidden_shape)
            gate = None

        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        attn_output, _ = eager_attention_forward(
            self, query_states, key_states, value_states, attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout, scaling=self.scaling,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        if gate is not None:
            attn_output = attn_output * torch.sigmoid(gate)
        attn_output = self.o_proj(attn_output)
        return attn_output


# ---------------------------------------------------------------------------
# Linear attention (GatedDeltaNet)
# ---------------------------------------------------------------------------
def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def torch_chunk_gated_delta_rule(
    query, key, value, g, beta, chunk_size: int = 64,
    initial_state=None, output_final_state: bool = False, use_qk_l2norm_in_kernel: bool = False,
):
    """Verbatim torch reference of the chunked gated delta rule."""
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


class Qwen3_5GatedDeltaNet(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]
        self.layer_norm_epsilon = config.rms_norm_eps

        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim, out_channels=self.conv_dim, bias=False,
            kernel_size=self.conv_kernel_size, groups=self.conv_dim, padding=self.conv_kernel_size - 1,
        )
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        A = torch.empty(self.num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.norm = Qwen3_5RMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        self.in_proj_qkv = nn.Linear(self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        # chunk size for the (exact) chunked gated-delta-rule; official default is 64.
        # The result is identical for any chunk size, so short sequences use a smaller
        # chunk to avoid padding to 64 and shrink the (autograd) python loop.
        self.chunk_size = 64

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        # attention_mask here is the 2D (B, L) padding mask (mamba-style zeroing).
        if attention_mask is not None and attention_mask.ndim == 2 and attention_mask.shape[0] > 1:
            hidden_states = hidden_states * attention_mask[:, :, None].to(hidden_states.dtype)
        batch_size, seq_len, _ = hidden_states.shape

        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)  # [B, conv_dim, L]
        z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        # causal depthwise conv + silu (torch fallback, no cache)
        mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])
        mixed_qkv = mixed_qkv.transpose(1, 2)

        query, key, value = torch.split(
            mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1
        )
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        # exact for any chunk size; pick the smallest power-of-two >= seq_len (capped at
        # the official default 64) so short sequences avoid padding to 64.
        eff_chunk = min(self.chunk_size, 1 << max(3, (max(seq_len, 1) - 1).bit_length()))
        core_attn_out, _ = torch_chunk_gated_delta_rule(
            query, key, value, g=g, beta=beta, chunk_size=eff_chunk,
            initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=True,
        )

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        return self.out_proj(core_attn_out)


# ---------------------------------------------------------------------------
# MLP / decoder layer / model
# ---------------------------------------------------------------------------
class Qwen3_5MLP(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen3_5DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int) -> None:
        super().__init__()
        self.block_type = config.layer_types[layer_idx]
        if self.block_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_idx)
        elif self.block_type == "full_attention":
            self.self_attn = Qwen3_5Attention(config, layer_idx)
        self.mlp = Qwen3_5MLP(config, config.intermediate_size)
        self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings, full_attention_mask=None, padding_mask=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.block_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states=hidden_states, attention_mask=padding_mask)
        else:
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=full_attention_mask,
            )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Qwen3_5TextModel(nn.Module):
    """Qwen3.5 LLM. Returns the last hidden-state tensor ``[B, L, hidden_size]``."""

    def __init__(self, config: Qwen3_5TextConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [Qwen3_5DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self._gradient_checkpointing_kwargs = {"use_reentrant": False}

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None) -> None:
        """Recompute decoder blocks during backward to reduce activation memory."""
        self.gradient_checkpointing = True
        self._gradient_checkpointing_kwargs = {
            "use_reentrant": False,
            **(gradient_checkpointing_kwargs or {}),
        }

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing = False

    def _build_full_attention_mask(self, padding_mask, L, dtype, device):
        min_val = torch.finfo(dtype).min
        causal = torch.full((L, L), min_val, dtype=dtype, device=device)
        causal = torch.triu(causal, diagonal=1)  # [L, L]
        causal = causal[None, None, :, :]  # [1,1,L,L]
        if padding_mask is not None:
            pad = (1.0 - padding_mask[:, None, None, :].to(dtype)) * min_val  # [B,1,1,L]
            causal = causal + pad
        return causal

    def forward(self, inputs_embeds=None, attention_mask=None, position_ids=None, input_ids=None):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        B, L, _ = inputs_embeds.shape
        device = inputs_embeds.device

        if position_ids is None:
            position_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)

        padding_mask = attention_mask if (attention_mask is not None and attention_mask.ndim == 2) else None
        full_attention_mask = self._build_full_attention_mask(padding_mask, L, inputs_embeds.dtype, device)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer in self.layers:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                hidden_states = activation_checkpoint(
                    layer,
                    hidden_states,
                    position_embeddings,
                    full_attention_mask,
                    padding_mask,
                    **self._gradient_checkpointing_kwargs,
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    full_attention_mask=full_attention_mask,
                    padding_mask=padding_mask,
                )

        return self.norm(hidden_states)
