"""Triton kernels for FlashMap drafting in vLLM.

The FlashMap DiT normalizes with ``x * bf16(weight * rsqrt(mean(x^2) + eps))``
(the weight is ``1 + scale`` of the training code) and rotates interleaved pairs
``(2i, 2i + 1)`` in float32 before rounding back to bf16. These kernels reproduce
that rounding so that vLLM drafts what the reference implementation drafts. The
ones the draft step calls are registered as custom ops so torch.compile keeps
them whole.
"""

import torch
import triton
import triton.language as tl

__all__ = [
    "block_self_attention",
    "rms_norm",
    "rms_norm_rope",
    "rope_table",
    "write_context_kv",
]


def rope_table(max_positions: int, head_dim: int, base: float, device) -> torch.Tensor:
    """``(max_positions, head_dim / 2, 2)`` float32 table of ``(cos, sin)`` of
    ``position * base^(-2i / head_dim)``, as the training code computes them."""
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    angles = torch.arange(max_positions, device=device).float()[:, None] * inv_freq
    return torch.stack([angles.cos(), angles.sin()], dim=-1).contiguous()


@triton.jit
def _rms_norm_kernel(x_ptr, weight_ptr, out_ptr, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    rms = tl.rsqrt(tl.sum(x * x, 0) / n_cols + eps)
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    scale = (weight * rms).to(tl.bfloat16).to(tl.float32)
    tl.store(out_ptr + row * n_cols + cols, (x * scale).to(tl.bfloat16), mask=mask)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """FlashMap RMSNorm of a contiguous bf16 ``x (..., C)``, in one launch."""
    assert x.dtype == torch.bfloat16 and x.is_contiguous()
    out = torch.empty_like(x)
    n_cols = x.shape[-1]
    if x.numel():
        _rms_norm_kernel[(x.numel() // n_cols,)](
            x, weight, out, n_cols, eps, BLOCK=triton.next_power_of_2(n_cols)
        )
    return out


@triton.jit
def _rms_norm_rope_kernel(
    x_ptr,
    weight_ptr,
    cos_sin_ptr,
    pos_ptr,
    out_ptr,
    eps,
    H: tl.constexpr,
    D: tl.constexpr,
):
    # One program per (token, head) of a contiguous (tokens, H, D) input.
    token = tl.program_id(0) // H
    head = tl.program_id(0) % H
    HALF: tl.constexpr = D // 2
    i = tl.arange(0, HALF)
    row = x_ptr + (token * H + head) * D
    even = tl.load(row + 2 * i).to(tl.float32)
    odd = tl.load(row + 2 * i + 1).to(tl.float32)
    rms = tl.rsqrt((tl.sum(even * even, 0) + tl.sum(odd * odd, 0)) / D + eps)
    w_even = tl.load(weight_ptr + 2 * i).to(tl.float32)
    w_odd = tl.load(weight_ptr + 2 * i + 1).to(tl.float32)
    even = (even * (w_even * rms).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
    odd = (odd * (w_odd * rms).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
    even = even.to(tl.float32)
    odd = odd.to(tl.float32)
    pos = tl.load(pos_ptr + token).to(tl.int64)
    cos = tl.load(cos_sin_ptr + pos * D + 2 * i)
    sin = tl.load(cos_sin_ptr + pos * D + 2 * i + 1)
    out = out_ptr + (token * H + head) * D
    tl.store(out + 2 * i, (even * cos - odd * sin).to(tl.bfloat16))
    tl.store(out + 2 * i + 1, (odd * cos + even * sin).to(tl.bfloat16))


@torch.library.custom_op("flashmap::rms_norm_rope", mutates_args=())
def rms_norm_rope(
    x: torch.Tensor,
    weight: torch.Tensor,
    cos_sin: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """RMSNorm then RoPE at ``positions (tokens,)`` of ``x (tokens, heads, head_dim)``
    with the ``rope_table`` ``cos_sin``, in one launch."""
    tokens, heads, head_dim = x.shape
    assert x.is_contiguous() and cos_sin.dtype == torch.float32
    out = torch.empty_like(x)
    if tokens:
        _rms_norm_rope_kernel[(tokens * heads,)](
            x, weight, cos_sin, positions, out, eps, H=heads, D=head_dim
        )
    return out


@rms_norm_rope.register_fake
def _(x, weight, cos_sin, positions, eps):
    return torch.empty_like(x)


@triton.jit
def _block_self_attention_kernel(
    qkv_ptr,
    q_norm_ptr,
    k_norm_ptr,
    cos_sin_ptr,
    pos_ptr,
    out_ptr,
    eps,
    sm_scale,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
):
    # One program per (block, head): the block's T queries attend to its T keys.
    b = tl.program_id(0) // H
    h = tl.program_id(0) % H
    HALF: tl.constexpr = D // 2
    t = tl.arange(0, T)[:, None]
    i = tl.arange(0, HALF)[None, :]
    d = tl.arange(0, D)[None, :]
    # qkv is (blocks, T, 3, H, D); RoPE pairs dims (2i, 2i + 1).
    row = qkv_ptr + (b * T + t) * (3 * H * D) + h * D
    pos = tl.load(pos_ptr + b * T + t).to(tl.int64)
    cos = tl.load(cos_sin_ptr + pos * D + 2 * i)
    sin = tl.load(cos_sin_ptr + pos * D + 2 * i + 1)

    q_even = tl.load(row + 2 * i).to(tl.float32)
    q_odd = tl.load(row + 2 * i + 1).to(tl.float32)
    k_even = tl.load(row + H * D + 2 * i).to(tl.float32)
    k_odd = tl.load(row + H * D + 2 * i + 1).to(tl.float32)

    q_rms = tl.rsqrt((tl.sum(q_even * q_even, 1) + tl.sum(q_odd * q_odd, 1)) / D + eps)[
        :, None
    ]
    k_rms = tl.rsqrt((tl.sum(k_even * k_even, 1) + tl.sum(k_odd * k_odd, 1)) / D + eps)[
        :, None
    ]
    qw_even = tl.load(q_norm_ptr + 2 * i).to(tl.float32)
    qw_odd = tl.load(q_norm_ptr + 2 * i + 1).to(tl.float32)
    kw_even = tl.load(k_norm_ptr + 2 * i).to(tl.float32)
    kw_odd = tl.load(k_norm_ptr + 2 * i + 1).to(tl.float32)
    q_even = (q_even * (qw_even * q_rms).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
    q_odd = (q_odd * (qw_odd * q_rms).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
    k_even = (k_even * (kw_even * k_rms).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
    k_odd = (k_odd * (kw_odd * k_rms).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
    q_even = q_even.to(tl.float32)
    q_odd = q_odd.to(tl.float32)
    k_even = k_even.to(tl.float32)
    k_odd = k_odd.to(tl.float32)

    q_rot_even = (q_even * cos - q_odd * sin).to(tl.bfloat16)
    q_rot_odd = (q_odd * cos + q_even * sin).to(tl.bfloat16)
    k_rot_even = (k_even * cos - k_odd * sin).to(tl.bfloat16)
    k_rot_odd = (k_odd * cos + k_even * sin).to(tl.bfloat16)

    scores = tl.dot(q_rot_even, tl.trans(k_rot_even)) + tl.dot(
        q_rot_odd, tl.trans(k_rot_odd)
    )
    scores = scores * sm_scale
    probs = tl.exp(scores - tl.max(scores, 1)[:, None])
    denom = tl.sum(probs, 1)[:, None]
    values = tl.load(row + 2 * H * D + d)
    out = tl.dot(probs.to(tl.bfloat16), values) / denom
    tl.store(out_ptr + (b * T + t) * (H * D) + h * D + d, out.to(tl.bfloat16))


@torch.library.custom_op("flashmap::block_self_attention", mutates_args=())
def block_self_attention(
    qkv: torch.Tensor,
    q_norm: torch.Tensor,
    k_norm: torch.Tensor,
    cos_sin: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Unmasked self-attention within each draft block, from its projected
    ``qkv (blocks, T, 3, heads, head_dim)``: q/k RMSNorms, RoPE at ``positions
    (blocks * T,)``, softmax attention. Returns ``(blocks * T, heads * head_dim)``."""
    blocks, block_size, _, heads, head_dim = qkv.shape
    assert qkv.is_contiguous() and cos_sin.dtype == torch.float32
    out = torch.empty(
        (blocks * block_size, heads * head_dim), dtype=qkv.dtype, device=qkv.device
    )
    if blocks:
        _block_self_attention_kernel[(blocks * heads,)](
            qkv,
            q_norm,
            k_norm,
            cos_sin,
            positions,
            out,
            eps,
            head_dim**-0.5,
            T=block_size,
            H=heads,
            D=head_dim,
        )
    return out


@block_self_attention.register_fake
def _(qkv, q_norm, k_norm, cos_sin, positions, eps):
    blocks, block_size, _, heads, head_dim = qkv.shape
    return qkv.new_empty((blocks * block_size, heads * head_dim))


@triton.jit
def _write_context_kv_kernel(
    kv_ptr,  # (tokens, L, 2, H, D) bf16, contiguous
    pos_ptr,  # (tokens,)
    slot_ptr,  # (tokens,) int64; negative = skip
    k_norm_ptr,  # (L, D)
    cos_sin_ptr,  # (max_positions, D / 2, 2) float32
    cache_ptrs,  # (L,) int64 data pointers of each layer's KV cache
    eps,
    block_size,
    stride_block,
    stride_head,
    stride_token,
    v_offset,
    L: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
):
    # One program per (token, layer); all heads at once.
    token = tl.program_id(0).to(tl.int64)
    layer = tl.program_id(1)
    slot = tl.load(slot_ptr + token).to(tl.int64)
    if slot < 0:
        return
    HALF: tl.constexpr = D // 2
    h = tl.arange(0, H)[:, None]
    i = tl.arange(0, HALF)[None, :]
    d = tl.arange(0, D)[None, :]
    k_row = kv_ptr + (token * L + layer) * (2 * H * D) + h * D
    v_row = k_row + H * D
    even = tl.load(k_row + 2 * i).to(tl.float32)
    odd = tl.load(k_row + 2 * i + 1).to(tl.float32)
    rms = tl.rsqrt((tl.sum(even * even, 1) + tl.sum(odd * odd, 1)) / D + eps)[:, None]
    w_even = tl.load(k_norm_ptr + layer * D + 2 * i).to(tl.float32)
    w_odd = tl.load(k_norm_ptr + layer * D + 2 * i + 1).to(tl.float32)
    even = (even * (w_even * rms).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
    odd = (odd * (w_odd * rms).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
    even = even.to(tl.float32)
    odd = odd.to(tl.float32)
    pos = tl.load(pos_ptr + token).to(tl.int64)
    cos = tl.load(cos_sin_ptr + pos * D + 2 * i)
    sin = tl.load(cos_sin_ptr + pos * D + 2 * i + 1)

    cache = tl.load(cache_ptrs + layer).to(tl.pointer_type(tl.bfloat16))
    dst = cache + (slot // block_size) * stride_block + h * stride_head
    dst += (slot % block_size) * stride_token
    tl.store(dst + 2 * i, (even * cos - odd * sin).to(tl.bfloat16))
    tl.store(dst + 2 * i + 1, (odd * cos + even * sin).to(tl.bfloat16))
    tl.store(dst + v_offset + d, tl.load(v_row + d))


def write_context_kv(
    kv: torch.Tensor,
    positions: torch.Tensor,
    slots: torch.Tensor,
    k_norm: torch.Tensor,
    cos_sin: torch.Tensor,
    cache_ptrs: torch.Tensor,
    cache_strides: tuple[int, int, int, int],
    block_size: int,
    eps: float,
) -> None:
    """Write every layer's context keys (k-normed and rotated) and values for
    ``kv (tokens, layers, 2, heads, head_dim)`` into the paged KV caches at
    ``slots`` (negative slots are skipped), in one launch.

    ``cache_strides`` are ``(block, head, token, value offset)`` in elements of
    the vLLM FlashAttention cache ``(blocks, heads, block_size, 2 * head_dim)``.
    """
    tokens, layers, _, heads, head_dim = kv.shape
    if not tokens:
        return
    stride_block, stride_head, stride_token, v_offset = cache_strides
    _write_context_kv_kernel[(tokens, layers)](
        kv,
        positions,
        slots,
        k_norm,
        cos_sin,
        cache_ptrs,
        eps,
        block_size,
        stride_block,
        stride_head,
        stride_token,
        v_offset,
        L=layers,
        H=heads,
        D=head_dim,
    )
