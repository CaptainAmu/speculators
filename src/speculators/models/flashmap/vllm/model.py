"""FlashMap draft model for vLLM.

The model is driven by ``FlashMapSpeculator`` (speculator.py), which reuses
vLLM's DFlash machinery: the target's aux hidden states are fused by ``fc``
(``combine_hidden_states``) and every committed token's cross-attention keys and
values are written to the draft's paged KV cache
(``precompute_and_store_context_kv``); each draft step then runs ``forward`` on
``block_size`` query rows per request, laid out as in DSpark: row 0 carries the
anchor token at the anchor position ``p`` and rows ``k`` sit at ``p + k``.

A FlashMap draft step, for each request:

* writes the anchor's keys/values (its embedding through ``anchor_proj`` and
  ``anchor_norm``) into the KV cache at ``p``, where the target's own context
  entry for that token lands once it is verified;
* evaluates the DiT on ``block_size`` slots at positions ``p + 1 ..`` from noise,
  their cross-attention reading positions ``0 .. p`` straight from the paged KV
  cache with FlashAttention;
* returns the final hidden states, which the speculator maps to logits through
  the DiT head folded into the target's LM head.

One jump (the default) needs no vocabulary-sized tensor: the state at ``t = 0``
embeds to a constant (its posterior is uniform) plus the noise's embedding
``x0 @ E ~ N(0, E^T E)``, drawn as ``z @ A`` with ``A^T A = E^T E`` from
``z ~ N(0, I)``. ``A = sqrt(Lambda) Q^T`` from ``E^T E = Q Lambda Q^T`` makes the
RMS of the drawn embedding elementwise in ``z`` and lets ``A``, the noise norm's
weight and ``x_embed`` fold into one ``(hidden, dim)`` matrix. Everything that
depends only on the jump times (every AdaLN modulation) is computed once.
More jumps run the reference algorithm exactly, with a vocabulary-sized state.
"""

import os
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention

from speculators.models.flashmap.vllm.kernels import (
    block_self_attention,
    rms_norm,
    rms_norm_rope,
    rope_table,
    write_context_kv,
)

logger = init_logger(__name__)

__all__ = ["FlashMapDraftModel"]


@torch.library.custom_op("flashmap::paged_cross_attention", mutates_args=())
def paged_cross_attention(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    block_table: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    fa_version: int,
) -> torch.Tensor:
    """Non-causal FlashAttention of ``query (tokens, heads, head_dim)`` onto the
    first ``seqused_k`` cached positions of each request."""
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

    head_dim = query.shape[-1]
    key_cache, value_cache = kv_cache.transpose(1, 2).split(head_dim, dim=-1)
    return flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max_seqlen_q,
        seqused_k=seqused_k,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=False,
        block_table=block_table,
        fa_version=fa_version,
    )


@paged_cross_attention.register_fake
def _(query, kv_cache, cu_seqlens_q, seqused_k, block_table, *args):
    return torch.empty_like(query)


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0):
    half = dim // 2
    freqs = torch.exp(
        -torch.log(torch.tensor(max_period, device=t.device))
        * torch.arange(half, dtype=torch.float32, device=t.device)
        / half
    )
    args = t.float()[..., None] * freqs
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class RMSNorm(nn.Module):
    """FlashMap's RMSNorm: ``x * bf16(weight * rsqrt(mean(x^2) + eps))``."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x.contiguous(), self.weight, self.eps)


class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.linear_1 = nn.Linear(dim, dim)
        self.linear_2 = nn.Linear(dim, dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = timestep_embedding(t, self.dim).to(self.linear_1.weight.dtype)
        return self.linear_2(F.silu(self.linear_1(emb)))


class AdaLN(nn.Module):
    """AdaLN-zero's condition projection; returns ``(scale, shift, gate)``."""

    def __init__(self, cond_dim: int, dim: int):
        super().__init__()
        self.linear = nn.Linear(cond_dim, 3 * dim)

    def forward(self, cond: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(self.linear(F.silu(cond)).chunk(3, dim=-1))


class FlashMapAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, head_dim: int, eps: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, dim, bias=False)
        self.q_norm = RMSNorm(head_dim, eps)
        self.k_norm = RMSNorm(head_dim, eps)


class FlashMapMLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)


class FlashMapLayer(nn.Module):
    def __init__(self, config, layer_prefix: str, vllm_config: VllmConfig):
        super().__init__()
        dim, cond_dim = config.hidden_size, config.timestep_embed_dim
        heads = config.num_attention_heads
        head_dim = getattr(config, "head_dim", None) or dim // heads
        self.self_attn_adaln = AdaLN(cond_dim, dim)
        self.self_attn = FlashMapAttention(dim, heads, head_dim, config.qk_norm_eps)
        self.cross_attn_adaln = AdaLN(cond_dim, dim)
        self.cross_attn = FlashMapAttention(dim, heads, head_dim, config.qk_norm_eps)
        self.mlp_adaln = AdaLN(cond_dim, dim)
        self.mlp = FlashMapMLP(dim, config.intermediate_size)
        # The cross-attention context lives in vLLM's paged KV cache; this layer
        # only owns the cache (the draft step reads it with its own kernels).
        self.cross_attn.attn = Attention(
            heads,
            head_dim,
            head_dim**-0.5,
            num_kv_heads=heads,
            cache_config=vllm_config.cache_config,
            prefix=f"{layer_prefix}.cross_attn.attn",
        )


class FlashMapDraftModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        self.config = config = speculative_config.draft_model_config.hf_config
        self.dtype = vllm_config.model_config.dtype
        self.max_model_len = vllm_config.model_config.max_model_len
        dim = config.hidden_size
        target_hidden = config.target_hidden_size
        num_aux = len(config.eagle_aux_hidden_state_layer_ids)
        eps = config.rms_norm_eps
        self.block_size = int(config.block_size)
        self.num_steps = int(os.environ.get("FLASHMAP_NUM_STEPS", config.num_steps))
        self.time_grid = [float(t) for t in config.time_grids[self.num_steps - 1]]
        # Scales the starting noise (tests set 0 to compare deterministic drafts).
        self.noise_scale = float(os.environ.get("FLASHMAP_NOISE_SCALE", "1"))

        self.fc = nn.Linear(num_aux * target_hidden, dim, bias=False)
        self.hidden_norm = RMSNorm(dim, eps)
        self.anchor_proj = nn.Linear(target_hidden, dim)
        self.anchor_norm = RMSNorm(dim, eps)
        self.state_norm = RMSNorm(target_hidden, eps)
        self.noise_norm = RMSNorm(target_hidden, eps)
        self.noise_gate = nn.Parameter(torch.ones(1), requires_grad=False)
        self.x_embed = nn.Linear(target_hidden, dim)
        self.time_embed = TimestepEmbedding(config.timestep_embed_dim)
        self.jump_embed = TimestepEmbedding(config.timestep_embed_dim)
        start = vllm_config.model_config.get_total_num_hidden_layers()
        model_prefix = f"{prefix}.model" if prefix else "model"
        self.layers = nn.ModuleList(
            [
                FlashMapLayer(config, f"{model_prefix}.layers.{start + i}", vllm_config)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.final_adaln = AdaLN(config.timestep_embed_dim, dim)
        self.head = nn.Linear(dim, target_hidden)

        # Draft vocabulary: a restricted head drafts over d2t's tokens only.
        vocab_size = int(config.vocab_size)
        draft_vocab_size = int(getattr(config, "draft_vocab_size", None) or vocab_size)
        self.d2t: torch.Tensor | None = None
        if draft_vocab_size != vocab_size:
            self.d2t = nn.Parameter(
                torch.zeros(draft_vocab_size, dtype=torch.long), requires_grad=False
            )
        self.vocab_size = vocab_size
        self._compiled = None

    # ------------------------------------------------------------------ loading

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        loaded = set()
        for name, tensor in weights:
            if name in ("embed_tokens.weight", "lm_head.weight", "t2d"):
                continue  # the target's own, shared at load
            param = params.get(name)
            if param is None:
                raise ValueError(f"unexpected FlashMap weight {name!r}")
            assert param.shape == tensor.shape, (name, param.shape, tensor.shape)
            param.data.copy_(tensor)
            loaded.add(name)
        # The attention layers' own (KV-cache scale) parameters are not weights.
        missing = {name for name in params if ".attn." not in name} - loaded
        if missing:
            raise ValueError(f"FlashMap checkpoint is missing {sorted(missing)}")
        return set(params)

    @torch.no_grad()
    def prepare(self, embed_tokens: torch.Tensor, lm_head: torch.Tensor) -> None:
        """Precompute what the draft step needs from the target's embedding
        ``(vocab, hidden)`` and LM head and from the fixed jump times."""
        device = self.fc.weight.device
        dtype = self.dtype
        embed_tokens = embed_tokens[: self.vocab_size]
        lm_head = lm_head[: self.vocab_size]
        self.embed_tokens = embed_tokens
        layer0 = self.layers[0]
        attn = layer0.cross_attn
        self.num_heads, self.head_dim = attn.num_heads, attn.head_dim
        self.dim = self.num_heads * self.head_dim
        self.softmax_scale = self.head_dim**-0.5
        self.fa_version = attn.attn.impl.vllm_flash_attn_version

        # AdaLN modulations for each jump (s, t) of the grid.
        self.modulations = []
        self.final_modulations = []
        grid = torch.tensor(self.time_grid, device=device, dtype=torch.float32)
        scale = float(self.config.time_scale)
        for step in range(self.num_steps):
            cond = self.time_embed(grid[step : step + 1] * scale) + self.jump_embed(
                grid[step + 1 : step + 2] * scale
            )
            per_layer = []
            for layer in self.layers:
                for adaln in (layer.self_attn_adaln, layer.cross_attn_adaln, layer.mlp_adaln):
                    mod_scale, shift, gate = adaln(cond)
                    per_layer += [(1 + mod_scale), shift, gate]
            # (layers, 9, dim): (1 + scale, shift, gate) for self, cross, mlp.
            self.modulations.append(
                torch.cat(per_layer).view(len(self.layers), 9, -1).contiguous()
            )
            f_scale, f_shift, _ = self.final_adaln(cond)
            self.final_modulations.append(torch.cat([1 + f_scale, f_shift]).contiguous())
        gate = self.noise_gate.float()
        if self.config.noise_gate == "jump":
            self.noise_gates = [
                float(gate * (grid[s + 1] - grid[s])) for s in range(self.num_steps)
            ]
        else:
            self.noise_gates = [float(gate)] * self.num_steps

        # embed_state at t = 0: the uniform posterior's embedding (bf16, as the
        # reference forms softmax(0) @ E) through state_norm.
        vocab = embed_tokens.shape[0]
        uniform = torch.full((1, vocab), 1.0 / vocab, device=device, dtype=dtype)
        self.uniform_state = self.state_norm(uniform @ embed_tokens)  # (1, hidden)
        self.first_hidden = self.x_embed(self.uniform_state)[0]  # (dim,)

        # One-jump noise: x0 @ E ~ N(0, E^T E) = z @ A, A = sqrt(Lambda) Q^T.
        gram = torch.zeros(
            embed_tokens.shape[1], embed_tokens.shape[1], device=device, dtype=torch.float32
        )
        for rows in embed_tokens.split(16384):
            rows = rows.float()
            gram += rows.T @ rows
        eigvals, eigvecs = torch.linalg.eigh(gram.double())
        eigvals = eigvals.clamp_min(0)
        factor = eigvals.sqrt()[:, None] * eigvecs.T  # (hidden, hidden), A
        # noise_norm(z A) = (z A) * w * r with r = rsqrt(sum(lambda z^2) / H + eps),
        # and x_embed is linear: fold A, w and x_embed's weight together.
        weight = self.noise_norm.weight.double()
        self.noise_embed = (
            self.x_embed.weight.double() @ (factor * weight[None, :]).T
        ).to(dtype)  # (dim, hidden), as F.linear takes it
        self.noise_eigvals = (eigvals / eigvals.numel()).float()  # lambda / H
        del gram, eigvecs, factor

        # Context keys/values of every layer in one GEMM.
        self.context_kv_weight = torch.cat(
            [
                torch.cat([layer.cross_attn.k_proj.weight, layer.cross_attn.v_proj.weight])
                for layer in self.layers
            ]
        )
        self.context_k_norm = torch.stack(
            [layer.cross_attn.k_norm.weight.float() for layer in self.layers]
        ).contiguous()
        self.self_qkv_weights = [
            torch.cat(
                [
                    layer.self_attn.q_proj.weight,
                    layer.self_attn.k_proj.weight,
                    layer.self_attn.v_proj.weight,
                ]
            )
            for layer in self.layers
        ]
        # Drop the now-duplicated projections.
        for layer in self.layers:
            for proj in (
                layer.cross_attn.k_proj,
                layer.cross_attn.v_proj,
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
            ):
                proj.weight = nn.Parameter(proj.weight.new_empty(0), requires_grad=False)
        self.cos_sin = rope_table(
            self.max_model_len + self.block_size + 1,
            self.head_dim,
            float(self.config.rope_parameters["rope_theta"])
            if getattr(self.config, "rope_parameters", None)
            else float(getattr(self.config, "rope_theta", 10000.0)),
            device,
        )

        # The DiT head folded into the LM head (no nonlinearity between them).
        self.hot_ids = None
        if self.d2t is not None:
            self.hot_ids = torch.arange(self.d2t.shape[0], device=device) + self.d2t.data
        self.head_weight, self.head_bias = self._fold_head(lm_head, self.hot_ids)
        self.jump_head = None
        if self.num_steps > 1:
            # Between jumps the state mixes in a full-vocabulary distribution.
            self.jump_head = (
                (self.head_weight, self.head_bias)
                if self.hot_ids is None
                else self._fold_head(lm_head, None)
            )
        self._cache_args = None
        if os.environ.get("FLASHMAP_COMPILE", "1") == "1":
            self._compiled = torch.compile(self._dit, dynamic=True, fullgraph=True)
        torch.cuda.empty_cache()
        logger.info(
            "FlashMap draft: %d jump(s) on grid %s, draft vocabulary %d",
            self.num_steps,
            self.time_grid,
            self.head_weight.shape[0],
        )

    def _fold_head(self, lm_head: torch.Tensor, rows: torch.Tensor | None, chunk=8192):
        count = lm_head.shape[0] if rows is None else rows.numel()
        weight = torch.empty(count, self.head.weight.shape[1], dtype=lm_head.dtype,
                             device=lm_head.device)
        bias = torch.empty(count, dtype=lm_head.dtype, device=lm_head.device)
        head_weight, head_bias = self.head.weight.float(), self.head.bias.float()
        for start in range(0, count, chunk):
            part = (
                lm_head[start : start + chunk]
                if rows is None
                else lm_head[rows[start : start + chunk]]
            ).float()
            weight[start : start + chunk] = (part @ head_weight).to(lm_head.dtype)
            bias[start : start + chunk] = (part @ head_bias).to(lm_head.dtype)
        return weight, bias

    # ------------------------------------------------------------------ context

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc(hidden_states)

    def cache_layout(self):
        """Data pointers and strides of every layer's KV cache, or None before
        the caches are allocated."""
        caches = [layer.cross_attn.attn.kv_cache for layer in self.layers]
        if not all(isinstance(c, torch.Tensor) and c.numel() for c in caches):
            return None
        ptrs = [c.data_ptr() for c in caches]
        if self._cache_args is None or self._cache_args[0] != ptrs:
            key, value = caches[0].transpose(1, 2).split(self.head_dim, dim=-1)
            assert key.stride(-1) == 1 and all(
                c.stride() == caches[0].stride() for c in caches
            )
            strides = (
                key.stride(0),
                key.stride(2),
                key.stride(1),
                (value.data_ptr() - key.data_ptr()) // key.element_size(),
            )
            ptr_tensor = torch.tensor(ptrs, dtype=torch.int64, device=caches[0].device)
            self._cache_args = (ptrs, ptr_tensor, strides, key.shape[1])
        return self._cache_args

    def _store_context(
        self,
        context: torch.Tensor,
        positions: torch.Tensor,
        slots: torch.Tensor | None,
    ) -> None:
        """Every layer's keys/values of normed ``context (N, dim)`` at
        ``positions``, written to the KV caches at ``slots``."""
        kv = F.linear(context, self.context_kv_weight).view(
            -1, len(self.layers), 2, self.num_heads, self.head_dim
        )
        layout = self.cache_layout() if slots is not None else None
        if layout is None:
            return
        _, ptrs, strides, block_size = layout
        write_context_kv(
            kv,
            positions,
            slots,
            self.context_k_norm,
            self.cos_sin,
            ptrs,
            strides,
            block_size,
            self.layers[0].cross_attn.k_norm.eps,
        )

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | list | None = None,
    ) -> None:
        if isinstance(context_slot_mapping, (list, tuple)):
            assert all(s is context_slot_mapping[0] for s in context_slot_mapping)
            context_slot_mapping = context_slot_mapping[0]
        self._store_context(
            self.hidden_norm(context_states), context_positions, context_slot_mapping
        )

    # ------------------------------------------------------------------ drafting

    def _dit(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        modulations: torch.Tensor,
        final_modulations: torch.Tensor,
        kv_caches: list[torch.Tensor],
        cu_seqlens_q: torch.Tensor | None,
        seqused_k: torch.Tensor | None,
        block_table: torch.Tensor | None,
    ) -> torch.Tensor:
        """The DiT on ``hidden (tokens, dim)`` (slots at ``positions``) with one
        jump's modulations; returns the final AdaLN output."""
        tokens = hidden.shape[0]
        blocks = tokens // self.block_size
        heads, head_dim, dim = self.num_heads, self.head_dim, self.dim
        for layer_id, layer in enumerate(self.layers):
            mods = modulations[layer_id]
            self_attn, cross_attn = layer.self_attn, layer.cross_attn
            normed = F.layer_norm(hidden, (dim,), eps=self.config.adaln_eps)
            normed = normed * mods[0] + mods[1]
            qkv = F.linear(normed, self.self_qkv_weights[layer_id])
            attn = block_self_attention(
                qkv.view(blocks, self.block_size, 3, heads, head_dim),
                self_attn.q_norm.weight,
                self_attn.k_norm.weight,
                self.cos_sin,
                positions,
                self_attn.q_norm.eps,
            )
            hidden = hidden + mods[2] * F.linear(attn, self_attn.o_proj.weight)

            normed = F.layer_norm(hidden, (dim,), eps=self.config.adaln_eps)
            normed = normed * mods[3] + mods[4]
            query = rms_norm_rope(
                F.linear(normed, cross_attn.q_proj.weight).view(tokens, heads, head_dim),
                cross_attn.q_norm.weight,
                self.cos_sin,
                positions,
                cross_attn.q_norm.eps,
            )
            if block_table is None:  # profiling run: no KV cache yet
                context = torch.zeros_like(query)
            else:
                context = paged_cross_attention(
                    query,
                    kv_caches[layer_id],
                    cu_seqlens_q,
                    seqused_k,
                    block_table,
                    self.block_size,
                    self.max_model_len,
                    self.softmax_scale,
                    self.fa_version,
                )
            hidden = hidden + mods[5] * F.linear(
                context.view(tokens, dim), cross_attn.o_proj.weight
            )

            normed = F.layer_norm(hidden, (dim,), eps=self.config.adaln_eps)
            normed = normed * mods[6] + mods[7]
            mlp = layer.mlp
            hidden = hidden + mods[8] * F.linear(
                F.gelu(F.linear(normed, mlp.fc1.weight, mlp.fc1.bias)),
                mlp.fc2.weight,
                mlp.fc2.bias,
            )
        normed = F.layer_norm(hidden, (dim,), eps=self.config.adaln_eps)
        return normed * final_modulations[0] + final_modulations[1]

    def _first_jump_input(self, tokens: int, device) -> torch.Tensor:
        """``x_embed(embed_state(x0, 0, t_1))`` for fresh noise, drawn in the
        embedding's eigenbasis (see the module docstring)."""
        z = torch.randn(tokens, self.noise_embed.shape[1], device=device,
                        dtype=torch.float32)
        if self.noise_scale != 1.0:
            z = z * self.noise_scale
        rms = torch.rsqrt((z * z) @ self.noise_eigvals + self.noise_norm.eps)
        noise = F.linear(z.to(self.dtype), self.noise_embed)
        return self.first_hidden + (self.noise_gates[0] * rms)[:, None].to(
            self.dtype
        ) * noise

    def _embed_state(self, state: torch.Tensor, step: int) -> torch.Tensor:
        """``x_embed(embed_state(state, s, t))`` of a vocabulary-sized float32
        ``state`` at jump ``step``, as the reference computes it."""
        start = self.time_grid[step]
        weight = self.embed_tokens
        evidence = start / max(1.0 - start, 1e-3) ** 2
        posterior = (state * evidence).softmax(dim=-1)
        both = torch.cat([posterior, state]).to(weight.dtype) @ weight
        posterior_emb, state_emb = both.split(state.shape[0])
        embedded = self.state_norm(posterior_emb) + (
            self.noise_gates[step] * self.noise_norm(state_emb)
        ).to(weight.dtype)
        return self.x_embed(embedded)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tokens = input_ids.shape[0]
        blocks = tokens // self.block_size
        metadata = get_forward_context().attn_metadata
        kv_caches = [layer.cross_attn.attn.kv_cache for layer in self.layers]
        cu_seqlens_q = seqused_k = block_table = None
        if metadata is not None:
            metadata = metadata[self.layers[0].cross_attn.attn.layer_name]
            cu_seqlens_q = metadata.query_start_loc
            block_table = metadata.block_table
            # The context is positions 0..p (the anchor's), and the DFlash
            # attention metadata counts the block's other rows too.
            seqused_k = (metadata.seq_lens - (self.block_size - 1)).clamp_(min=0)
            anchor_rows = input_ids.view(blocks, self.block_size)[:, 0]
            anchor_context = self.anchor_norm(
                self.anchor_proj(F.embedding(anchor_rows, self.embed_tokens))
            )
            self._store_context(
                anchor_context,
                positions.view(blocks, self.block_size)[:, 0],
                metadata.slot_mapping[:tokens].view(blocks, self.block_size)[:, 0],
            )
        slot_positions = positions + 1
        dit = self._compiled or self._dit
        dit_args = (kv_caches, cu_seqlens_q, seqused_k, block_table)

        if self.num_steps == 1:
            hidden = self._first_jump_input(tokens, input_ids.device)
            return dit(
                hidden,
                slot_positions,
                self.modulations[0],
                self.final_modulations[0],
                *dit_args,
            )

        # Several jumps (Algorithm 8): x <- X_{t_n, t_n+1}(x) between jumps.
        state = self.noise_scale * torch.randn(
            tokens, self.vocab_size, device=input_ids.device, dtype=torch.float32
        )
        for step in range(self.num_steps):
            hidden = dit(
                self._embed_state(state, step),
                slot_positions,
                self.modulations[step],
                self.final_modulations[step],
                *dit_args,
            )
            if step + 1 < self.num_steps:
                weight, bias = self.jump_head
                delta = F.linear(hidden, weight, bias).float().softmax(dim=-1)
                start, end = self.time_grid[step], self.time_grid[step + 1]
                state = ((1.0 - end) * state + (end - start) * delta) / (1.0 - start)
        return hidden

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Logits over the draft vocabulary (``hot_ids`` when restricted)."""
        return F.linear(hidden_states, self.head_weight, self.head_bias)
