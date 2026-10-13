"""DiT denoiser layers of the FlashMap drafter.

Each layer is self-attention within a draft block, cross-attention onto the
verifier context (the fused auxiliary hidden states before the anchor plus the
anchor token), and an MLP, each modulated by AdaLN-zero on the flow time(s).
Rotary embeddings use absolute position ids with interleaved (GPT-J style)
pairs, so draft slots and context tokens share one coordinate system.
"""

import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn
from torch.nn.attention.flex_attention import BlockMask, flex_attention
from transformers.models.qwen3.modeling_qwen3 import (
    GradientCheckpointingLayer,
    Qwen3RMSNorm,
)

__all__ = [
    "FlashMapAdaLNZero",
    "FlashMapAttention",
    "FlashMapDiTLayer",
    "FlashMapMLP",
    "FlashMapTimestepEmbedding",
    "apply_interleaved_rope",
    "rope_cos_sin",
    "timestep_embedding",
]

# Compiled once: eager flex attention materializes the full score matrix.
_compiled_flex_attention = torch.compile(flex_attention, dynamic=False)


def rope_cos_sin(
    position_ids: torch.Tensor, head_dim: int, base: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Position ids ``[..., T]`` -> float32 ``(cos, sin)`` of ``[..., T, head_dim/2]``."""
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, head_dim, 2, device=position_ids.device).float() / head_dim
        )
    )
    angles = position_ids.float()[..., None] * inv_freq
    return angles.cos(), angles.sin()


def apply_interleaved_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Rotate ``x [B, T, heads, head_dim]`` by ``cos/sin [B, T, head_dim/2]``,
    pairing dims ``(2i, 2i+1)``."""
    x_even, x_odd = x.float().unflatten(-1, (-1, 2)).unbind(-1)
    cos, sin = cos.unsqueeze(-2), sin.unsqueeze(-2)
    rotated = torch.stack(
        [x_even * cos - x_odd * sin, x_odd * cos + x_even * sin], dim=-1
    )
    return rotated.flatten(-2).type_as(x)


def timestep_embedding(
    t: torch.Tensor, dim: int, max_period: float = 10000.0
) -> torch.Tensor:
    """Sinusoidal embedding ``[..., dim]`` (cosines first) of float times ``t``."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=t.device)
        / half
    )
    args = t.float()[..., None] * freqs
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class FlashMapTimestepEmbedding(nn.Module):
    def __init__(self, hidden_size: int, frequency_size: int = 256):
        super().__init__()
        self.frequency_size = frequency_size
        self.linear_1 = nn.Linear(frequency_size, hidden_size)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = timestep_embedding(t, self.frequency_size)
        return self.linear_2(self.act(self.linear_1(emb)))


class FlashMapAdaLNZero(nn.Module):
    """LayerNorm modulated by the time conditioning; also returns the residual
    gate. Zero-initialized, so every layer starts as the identity."""

    def __init__(self, hidden_size: int, cond_dim: int, eps: float = 1e-6):
        super().__init__()
        self.linear = nn.Linear(cond_dim, 3 * hidden_size)
        self.linear._flashmap_zero_init = True  # type: ignore[assignment]
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=eps)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scale, shift, gate = self.linear(F.silu(cond)).chunk(3, dim=-1)
        return self.norm(x) * (1 + scale) + shift, gate


class FlashMapAttention(nn.Module):
    """QK-normed rotary attention whose keys/values come from ``project_kv``, so
    the cross-attention context is projected once per forward and reused by
    every denoiser evaluation."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        kv_input_size: int | None = None,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        kv_input_size = kv_input_size or hidden_size
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(kv_input_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(kv_input_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=eps)

    def project_kv(
        self, kv: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``kv [B, T, kv_input_size]`` -> rotated keys and values
        ``[B, heads, T, head_dim]``."""
        bsz, seq_len, _ = kv.shape
        keys = self.k_proj(kv).view(bsz, seq_len, self.num_heads, self.head_dim)
        values = self.v_proj(kv).view(bsz, seq_len, self.num_heads, self.head_dim)
        keys = apply_interleaved_rope(self.k_norm(keys), *rope)
        return keys.transpose(1, 2), values.transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor],
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: BlockMask | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``mask`` is a flex ``BlockMask`` or a boolean ``[1, 1, T, T_kv]``
        (True = attend); ``None`` attends everywhere."""
        bsz, seq_len, _ = x.shape
        query = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        query = apply_interleaved_rope(self.q_norm(query), *rope).transpose(1, 2)
        query, keys = query.to(values.dtype), keys.to(values.dtype)
        if isinstance(mask, BlockMask):
            out = _compiled_flex_attention(query, keys, values, block_mask=mask)
        else:
            out = F.scaled_dot_product_attention(query, keys, values, attn_mask=mask)
        return self.o_proj(out.transpose(1, 2).reshape(bsz, seq_len, -1))


class FlashMapMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class FlashMapDiTLayer(GradientCheckpointingLayer):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        cond_dim: int,
        intermediate_size: int,
        adaln_eps: float = 1e-6,
        qk_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.self_attn_adaln = FlashMapAdaLNZero(hidden_size, cond_dim, adaln_eps)
        self.self_attn = FlashMapAttention(hidden_size, num_heads, eps=qk_norm_eps)
        self.cross_attn_adaln = FlashMapAdaLNZero(hidden_size, cond_dim, adaln_eps)
        self.cross_attn = FlashMapAttention(hidden_size, num_heads, eps=qk_norm_eps)
        self.mlp_adaln = FlashMapAdaLNZero(hidden_size, cond_dim, adaln_eps)
        self.mlp = FlashMapMLP(hidden_size, intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,  # [1, num_blocks * block_size, hidden_size]
        cond: torch.Tensor,  # [1, num_blocks * block_size, cond_dim]
        rope: tuple[torch.Tensor, torch.Tensor],  # per slot, [1, slots, head_dim/2]
        ctx_keys: torch.Tensor,  # [1, heads, ctx_len, head_dim]
        ctx_values: torch.Tensor,  # [1, heads, ctx_len, head_dim]
        ctx_mask: BlockMask | torch.Tensor | None,
        block_size: int,
    ) -> torch.Tensor:
        bsz, seq_len, dim = hidden_states.shape

        # A draft block only sees itself, so self-attention runs with the blocks
        # folded into the batch instead of under a mask.
        h, gate = self.self_attn_adaln(hidden_states, cond)
        h = h.reshape(-1, block_size, dim)
        block_rope = tuple(r.reshape(-1, block_size, r.shape[-1]) for r in rope)
        keys, values = self.self_attn.project_kv(h, block_rope)  # type: ignore[arg-type]
        h = self.self_attn(h, block_rope, keys, values)  # type: ignore[arg-type]
        hidden_states = hidden_states + gate * h.reshape(bsz, seq_len, dim)

        h, gate = self.cross_attn_adaln(hidden_states, cond)
        hidden_states = hidden_states + gate * self.cross_attn(
            h, rope, ctx_keys, ctx_values, ctx_mask
        )

        h, gate = self.mlp_adaln(hidden_states, cond)
        return hidden_states + gate * self.mlp(h)
