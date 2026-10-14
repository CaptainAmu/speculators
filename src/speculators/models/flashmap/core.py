import copy
import logging
from types import SimpleNamespace
from typing import ClassVar, NamedTuple

import torch
from torch import nn
from torch.nn.attention.flex_attention import create_mask
from transformers import PretrainedConfig
from transformers import initialization as init
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config, Qwen3RMSNorm

from speculators.losses import LossConfig, resolve_loss_config
from speculators.model import DraftVocabMixin, SpeculatorModel
from speculators.models.dflash.attention import create_anchor_block_mask_mod
from speculators.models.dflash.core import DFlashDraftModel, _compiled_create_block_mask
from speculators.models.dflash.utils import (
    get_base_indices_for_anchored_blocks,
    select_anchors,
)
from speculators.models.flashmap.config import FlashMapSpeculatorConfig
from speculators.models.flashmap.metrics import (
    compute_draft_metrics,
    compute_flow_loss,
    compute_map_loss,
)
from speculators.models.flashmap.model_definitions import (
    FlashMapAdaLNZero,
    FlashMapDiTLayer,
    FlashMapTimestepEmbedding,
    rope_cos_sin,
)
from speculators.models.flashmap.schedule import DecodingErrorSchedule
from speculators.models.utils import resolve_verifier_norm_class

__all__ = ["FlashMapDraftModel"]

logger = logging.getLogger(__name__)

# Block categories of the semigroup objective's batch split.
DIAGONAL, BOUNDARY, OFF_DIAGONAL = 0, 1, 2

ContextKV = list[tuple[torch.Tensor, torch.Tensor]]


class AnchoredBlocks(NamedTuple):
    anchor_positions: torch.Tensor  # [num_anchors]
    label_indices: torch.Tensor  # [num_anchors * block_size]
    labels: torch.Tensor  # [1, num_anchors * block_size]
    label_mask: torch.Tensor  # [1, num_anchors * block_size]
    position_ids: torch.Tensor  # [1, num_anchors * block_size]


@SpeculatorModel.register("flashmap")
class FlashMapDraftModel(DraftVocabMixin, SpeculatorModel):
    """FlashMap: a block drafter that maps Gaussian noise to a block of tokens.

    A block of ``block_size`` draft tokens after an anchor is a point on the
    vocabulary simplex per slot. A DiT denoiser, conditioned like DFlash on the
    verifier's auxiliary hidden states before the anchor and on the anchor token
    (both through cross-attention), predicts the clean block from a noised state
    ``(1 - t) z + t onehot(y)`` (Flow Map Language Models, arXiv:2602.16813).
    Vocabulary projections on both sides are the verifier's frozen embedding and
    lm_head. Training has two stages:

    1. ``config.flow_map=False``: the flow, the denoiser ``D_t(x)`` trained by
       cross-entropy against the data tokens at times from the decoding-error
       schedule.
    2. ``config.flow_map=True``: the flow map ``delta_{s,t}(x)``, initialized from
       the flow and distilled from it with the paper's semigroup objective. It
       is the drafter: one evaluation ``delta_{0,1}(z)`` drafts a whole block.
    """

    config_class: ClassVar[type[FlashMapSpeculatorConfig]] = FlashMapSpeculatorConfig  # type: ignore[misc]
    supports_gradient_checkpointing = True
    _no_split_modules = ["FlashMapDiTLayer"]
    # Verifier-owned weights are reloaded from the verifier, never serialized.
    _keys_to_ignore_on_load_missing: ClassVar[list[str]] = [  # type: ignore[misc]
        "embed_tokens.weight",
        "lm_head.weight",
        "verifier_lm_head.weight",
        "verifier_norm.weight",
        "t2d",
        "d2t",
    ]
    _keys_to_ignore_on_save: ClassVar[list[str]] = [  # type: ignore[misc,assignment]
        "embed_tokens.weight",
        "lm_head.weight",
        "verifier_lm_head.weight",
        "verifier_norm.weight",
    ]

    t2d: torch.Tensor | None
    d2t: torch.Tensor | None

    def __init__(self, config: FlashMapSpeculatorConfig) -> None:
        tl_config = config.transformer_layer_config
        if tl_config._attn_implementation is None:  # noqa: SLF001
            tl_config._attn_implementation = "simple_flex_attention"  # noqa: SLF001
        self._attn_impl = tl_config._attn_implementation  # noqa: SLF001
        super().__init__(config=config)
        # DraftVocabMixin sizes the frozen verifier matrices from
        # transformer_layer_config, which here describes the DiT.
        verifier_dims = copy.copy(tl_config)
        verifier_dims.hidden_size = config.target_hidden_size
        self._init_vocab(
            SimpleNamespace(
                transformer_layer_config=verifier_dims,
                draft_vocab_size=config.draft_vocab_size,
            )
        )
        if self.use_draft_vocab:
            raise NotImplementedError(
                "FlashMap trains over the verifier's full vocabulary; omit "
                "--draft-vocab-size (a restricted vocabulary is a serving option)."
            )
        # The frozen verifier lm_head and the draft lm_head are the same matrix.
        self.verifier_lm_head = self.lm_head

        hidden_size = config.target_hidden_size
        if hidden_size is None:
            raise ValueError("FlashMap needs the verifier's target_hidden_size.")
        dit_size = tl_config.hidden_size
        cond_dim = config.timestep_embed_dim
        norm_eps = tl_config.rms_norm_eps
        self.block_size = config.block_size
        self.head_dim = dit_size // tl_config.num_attention_heads
        self.rope_theta = float(
            (getattr(tl_config, "rope_parameters", None) or {}).get(
                "rope_theta", getattr(tl_config, "rope_theta", 10000.0)
            )
        )

        # Context: fused auxiliary hidden states, and the anchor token.
        self.fc = nn.Linear(
            len(self.target_layer_ids) * hidden_size, dit_size, bias=False
        )
        self.hidden_norm = Qwen3RMSNorm(dit_size, eps=norm_eps)
        self.anchor_proj = nn.Linear(hidden_size, dit_size)
        self.anchor_norm = Qwen3RMSNorm(dit_size, eps=norm_eps)
        # The state is read through the verifier embedding (see embed_state).
        self.state_norm = Qwen3RMSNorm(hidden_size, eps=norm_eps)
        if config.flow_map:
            self.noise_norm = Qwen3RMSNorm(hidden_size, eps=norm_eps)
            if config.noise_gate == "jump":
                # The released recipe: the weight fixed at 1, times the jump.
                self.register_buffer("noise_gate", torch.ones(1))
            else:
                # Starts shut, so a map initialized from its flow is the flow.
                self.noise_gate = nn.Parameter(torch.zeros(1))

        # The DiT denoiser.
        self.x_embed = nn.Linear(hidden_size, dit_size)
        self.time_embed = FlashMapTimestepEmbedding(cond_dim, cond_dim)
        if config.flow_map:
            # Zero output: a map initialized from its flow starts as the flow.
            self.jump_embed = FlashMapTimestepEmbedding(cond_dim, cond_dim)
            self.jump_embed.linear_2.flashmap_zero_init = True  # type: ignore[assignment]
        self.layers = nn.ModuleList(
            [
                FlashMapDiTLayer(
                    dit_size,
                    tl_config.num_attention_heads,
                    cond_dim,
                    intermediate_size=tl_config.intermediate_size,
                    adaln_eps=config.adaln_eps,
                    qk_norm_eps=config.qk_norm_eps,
                )
                for _ in range(tl_config.num_hidden_layers)
            ]
        )
        self.final_adaln = FlashMapAdaLNZero(dit_size, cond_dim, config.adaln_eps)
        # Zero output: start from a uniform distribution over tokens.
        self.head = nn.Linear(dit_size, hidden_size)
        self.head.flashmap_zero_init = True  # type: ignore[assignment]

        # Only for validation metrics: the verifier's own greedy tokens.
        self.verifier_norm = resolve_verifier_norm_class(config)(
            hidden_size, eps=norm_eps
        )
        self.verifier_norm.weight.requires_grad = False

        self.time_schedule = DecodingErrorSchedule(self.draft_vocab_size)
        self._teacher: list[FlashMapDraftModel] = []
        self.post_init()

    def _init_weights(self, module):
        super()._init_weights(module)
        if getattr(module, "flashmap_zero_init", False):
            init.zeros_(module.weight)
            init.zeros_(module.bias)

    @property
    def target_layer_ids(self) -> list[int]:
        """Target layer IDs for auxiliary hidden states."""
        return self.config.aux_hidden_state_layer_ids

    def load_verifier_weights(self):
        """Reconstruct the verifier weights omitted from FlashMap checkpoints."""
        self._load_verifier_weights(overwrite_embed_tokens=True, overwrite_lm_head=True)

    # ------------------------------------------------------------------
    # Construction for training
    # ------------------------------------------------------------------

    @classmethod
    def from_training_args(
        cls,
        verifier_config: PretrainedConfig,
        t2d: torch.Tensor | None = None,
        d2t: torch.Tensor | None = None,
        **kwargs,
    ) -> "FlashMapDraftModel":
        """Stage 1 (the flow) from the CLI's shaping args, or stage 2 (the flow
        map) from ``--flashmap-teacher``, a trained stage-1 checkpoint.

        The DiT depth is ``verifier_config.num_hidden_layers`` (``--num-layers``)
        and its width the ``--dit-*`` args. A flow map copies its architecture
        from the teacher, starts from the teacher's weights, and keeps the frozen
        teacher (unregistered, so it is neither trained nor saved).
        """
        teacher_path = kwargs.get("flashmap_teacher")
        if teacher_path:
            model = cls.from_teacher(
                teacher_path,
                t2d=t2d,
                d2t=d2t,
                draft_attn_impl=kwargs.get("draft_attn_impl"),
            )
            requested = kwargs.get("target_layer_ids")
            if requested is not None and list(requested) != model.target_layer_ids:
                raise ValueError(
                    f"--target-layer-ids {list(requested)} differ from the "
                    f"teacher's {model.target_layer_ids}; the flow map reads the "
                    "teacher's layers."
                )
            return model

        # transformer_layer_config describes the DiT; the verifier's hidden size
        # is target_hidden_size.
        dit_size = kwargs.get("dit_hidden_size", 2048)
        num_heads = kwargs.get("dit_num_heads", 16)
        dit_config = Qwen3Config(
            vocab_size=verifier_config.vocab_size,
            hidden_size=dit_size,
            intermediate_size=4 * dit_size,
            num_hidden_layers=verifier_config.num_hidden_layers,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
            head_dim=dit_size // num_heads,
            hidden_act="gelu",
            rms_norm_eps=1e-6,
            max_position_embeddings=verifier_config.max_position_embeddings,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
            tie_word_embeddings=False,
        )
        # Every slot is a draft token, so a block proposes block_size tokens.
        base = DFlashDraftModel._build_base_config_kwargs(  # noqa: SLF001
            "flashmap", dit_config, **{**kwargs, "sample_from_anchor": True}
        )
        # The jump grids depend on the vocabulary through the schedule.
        schedule = DecodingErrorSchedule(verifier_config.vocab_size)
        config = FlashMapSpeculatorConfig(
            **base,
            target_hidden_size=verifier_config.hidden_size,
            timestep_embed_dim=kwargs.get("timestep_embed_dim", 256),
            time_grids=[
                schedule.grid(num_steps, torch.device("cpu")).tolist()
                for num_steps in range(1, 5)
            ],
            flow_map=False,
        )
        model = cls(config=config)
        model.load_vocab_mappings(t2d, d2t)
        model.load_verifier_weights()
        return model

    @classmethod
    def from_teacher(
        cls,
        teacher: "str | FlashMapDraftModel",
        t2d: torch.Tensor | None = None,
        d2t: torch.Tensor | None = None,
        draft_attn_impl: str | None = None,
    ) -> "FlashMapDraftModel":
        """A flow map initialized from a trained flow, which it then distils.

        The flow map's two-time denoiser on the diagonal is the flow's denoiser
        (Proposition 3.3(iii) of the paper), so copying the flow's weights, with
        the end-time embedding zero at its output and the noise term zero on the
        diagonal, starts the map as the flow.
        """
        if isinstance(teacher, str):
            teacher = cls.from_pretrained(teacher, t2d=t2d, d2t=d2t)  # type: ignore[assignment]
        assert isinstance(teacher, FlashMapDraftModel)  # noqa: S101
        if teacher.config.flow_map:
            raise ValueError(
                "--flashmap-teacher must point at a stage-1 FlashMap flow "
                "(flow_map=false), not at a flow map."
            )
        config: FlashMapSpeculatorConfig = copy.deepcopy(teacher.config)  # type: ignore[assignment]
        config.flow_map = True
        if draft_attn_impl is not None:
            config.transformer_layer_config._attn_implementation = draft_attn_impl  # noqa: SLF001
        model = cls(config=config)
        model.load_vocab_mappings(t2d, d2t)
        missing, unexpected = model.load_state_dict(teacher.state_dict(), strict=False)
        new_keys = ("noise_norm.", "noise_gate", "jump_embed.")
        if unexpected or any(not key.startswith(new_keys) for key in missing):
            raise RuntimeError(
                f"Teacher does not match the flow map: missing={missing}, "
                f"unexpected={unexpected}"
            )
        model.attach_teacher(teacher)
        return model

    def attach_teacher(self, teacher: "FlashMapDraftModel") -> None:
        """Keep a frozen flow to distil from, outside ``parameters()`` and
        ``state_dict()``: a checkpoint holds exactly the trained network."""
        teacher.requires_grad_(False).eval()
        self._teacher = [teacher]

    @staticmethod
    def get_trainer_kwargs(**kwargs) -> tuple[dict, dict]:
        """Training and validation kwargs (the same for both stages)."""
        shared = {
            "loss_config": resolve_loss_config(
                kwargs.get("loss_fn") or "kl_div",
                kwargs.get("loss_implementation", "fused"),
            ),
            "gamma": kwargs.get("dflash_decay_gamma") or 7.0,
            "max_anchors": kwargs.get("max_anchors", 512),
            "diagonal_fraction": kwargs.get("diagonal_fraction", 0.5),
            "boundary_prob": kwargs.get("boundary_prob", 0.0625),
            "antithetic_sampling": kwargs.get("antithetic_sampling", True),
        }
        return dict(shared), dict(shared)

    # ------------------------------------------------------------------
    # The denoiser
    # ------------------------------------------------------------------

    def embed_state(
        self,
        state: torch.Tensor,  # [1, slots, vocab]
        times: torch.Tensor,  # [1, slots]
        end: torch.Tensor | None = None,  # [1, slots]
    ) -> torch.Tensor:
        """Continuous state at per-slot ``times`` -> ``[1, slots, hidden_size]``.

        The state is read as ``softmax(t x / (1 - t)^2)``, the exact posterior
        over the clean token under a uniform prior, embedded with the verifier's
        embedding. At ``t = 0`` that posterior is uniform whatever the noise, so a
        flow map also reads the state linearly, weighted by the jump ``end - t``
        (zero on the diagonal, where the posterior is a sufficient statistic).
        """
        weight = self.embed_tokens.weight
        evidence = (times / (1.0 - times).clamp_min(1e-3) ** 2).unsqueeze(-1)
        posterior = (state.float() * evidence).softmax(dim=-1)
        embedded = self.state_norm(posterior.to(weight.dtype) @ weight)
        if self.config.flow_map and end is not None:
            gate: torch.Tensor = self.noise_gate
            if self.config.noise_gate == "jump":
                gate = gate * (end - times)
            gate = gate.unsqueeze(-1)
            noise = self.noise_norm(state.to(weight.dtype) @ weight)
            embedded = embedded + gate.to(embedded.dtype) * noise
        return embedded

    def project_context(
        self,
        hidden_states: torch.Tensor,  # [1, seq_len, num_aux_layers * hidden_size]
        input_ids: torch.Tensor,  # [1, seq_len]
        position_ids: torch.Tensor,  # [1, seq_len]
        anchor_positions: torch.Tensor,  # [num_anchors]
    ) -> ContextKV:
        """Per-layer cross-attention keys/values over ``[context | anchors]``: the
        fused hidden states of every position, then one anchor token per block."""
        context = torch.cat(
            [
                self.hidden_norm(self.fc(hidden_states)),
                self.anchor_norm(
                    self.anchor_proj(self.embed_tokens(input_ids[:, anchor_positions]))
                ),
            ],
            dim=1,
        )
        context_position_ids = torch.cat(
            [position_ids, position_ids[:, anchor_positions]], dim=1
        )
        rope = rope_cos_sin(context_position_ids, self.head_dim, self.rope_theta)
        return [
            layer.cross_attn.project_kv(context, rope)  # type: ignore[union-attr,operator]
            for layer in self.layers
        ]

    def denoise(
        self,
        state: torch.Tensor,  # [1, slots, vocab]
        times: torch.Tensor,  # [1, slots]
        position_ids: torch.Tensor,  # [1, slots]
        ctx_kv: ContextKV,
        ctx_mask,
        end: torch.Tensor | None = None,  # [1, slots]
    ) -> torch.Tensor:
        """Logits ``[1, slots, vocab]`` of the flow's denoiser ``D_t(x)`` or, given
        ``end``, of the flow map's two-time denoiser ``delta_{t,end}(x)``."""
        scale = self.config.time_scale
        cond = self.time_embed(times * scale)
        if end is not None:
            cond = cond + self.jump_embed(end * scale)
        rope = rope_cos_sin(position_ids, self.head_dim, self.rope_theta)
        hidden = self.x_embed(self.embed_state(state, times, end))
        for layer, (ctx_keys, ctx_values) in zip(self.layers, ctx_kv, strict=True):
            hidden = layer(
                hidden, cond, rope, ctx_keys, ctx_values, ctx_mask, self.block_size
            )
        hidden, _ = self.final_adaln(hidden, cond)
        return self.lm_head(self.head(hidden))

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------

    @torch.compiler.disable
    def _sample_blocks(
        self,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        document_ids: torch.Tensor,
        position_ids: torch.Tensor,
        num_anchors: int,
    ) -> AnchoredBlocks:
        """Anchor blocks: an anchor needs itself and its first draft token in the
        loss mask and in one document; each block supervises the contiguous
        prefix of its slots that stays in the anchor's document and loss mask."""
        block_size = self.block_size
        loss_mask = loss_mask.bool()
        next_ok = torch.zeros_like(loss_mask)
        next_ok[:, :-1] = loss_mask[:, 1:] & (
            document_ids[:, 1:] == document_ids[:, :-1]
        )
        anchor_positions, anchor_valid = select_anchors(
            loss_mask & next_ok, num_anchors, block_size
        )
        label_indices = get_base_indices_for_anchored_blocks(
            anchor_positions + 1, block_size
        )
        anchor_docs = document_ids[:, anchor_positions].repeat_interleave(
            block_size, dim=1
        )
        slot_ok = (
            loss_mask[:, label_indices]
            & (document_ids[:, label_indices] == anchor_docs)
            & anchor_valid.repeat_interleave(block_size).unsqueeze(0)
        )
        label_mask = slot_ok.view(-1, block_size).int().cumprod(dim=-1).view(1, -1)
        # Slots sit at the positions after the anchor's, even past a document end.
        slot_position_ids = get_base_indices_for_anchored_blocks(
            position_ids[0, anchor_positions] + 1, block_size
        ).unsqueeze(0)
        return AnchoredBlocks(
            anchor_positions=anchor_positions,
            label_indices=label_indices,
            labels=input_ids[:, label_indices],
            label_mask=label_mask.bool(),
            position_ids=slot_position_ids,
        )

    @torch.compiler.disable
    def _context_mask(
        self,
        document_ids: torch.Tensor,
        anchor_positions: torch.Tensor,
    ):
        """Block ``j`` sees the context of its document strictly before its anchor,
        and its own anchor entry (DFlash's anchored-block mask, one key per anchor
        on the block side)."""
        mask_mod, q_len, kv_len = create_anchor_block_mask_mod(
            document_ids=document_ids.squeeze(0),
            total_seq_len=document_ids.shape[1],
            anchor_positions=anchor_positions,
            block_size=self.block_size,
            kv_block_size=1,
        )
        create = (
            _compiled_create_block_mask
            if self._attn_impl == "simple_flex_attention"
            else create_mask
        )
        return create(
            mask_mod,
            B=None,
            H=None,
            Q_LEN=q_len,
            KV_LEN=kv_len,
            device=document_ids.device,
        )

    def _interpolant(
        self, labels: torch.Tensor, times: torch.Tensor, dtype: torch.dtype
    ) -> torch.Tensor:
        """``(1 - t) z + t onehot(y)`` per slot, ``[1, slots, vocab]``."""
        state = torch.randn(
            *labels.shape, self.draft_vocab_size, device=labels.device, dtype=dtype
        )
        state *= (1.0 - times).unsqueeze(-1).to(dtype)
        state.scatter_add_(-1, labels.unsqueeze(-1), times.unsqueeze(-1).to(dtype))
        return state

    def forward(
        self,
        hidden_states: torch.Tensor,  # [1, seq_len, num_aux_layers * hidden_size]
        input_ids: torch.Tensor,  # [1, seq_len]
        loss_mask: torch.Tensor,  # [1, seq_len]
        document_ids: torch.Tensor,  # [1, seq_len]
        position_ids: torch.Tensor | None = None,  # [1, seq_len]
        verifier_last_hidden_states: torch.Tensor | None = None,
        loss_config: LossConfig | None = None,
        gamma: float = 7.0,
        max_anchors: int = 512,
        diagonal_fraction: float = 0.5,
        boundary_prob: float = 0.0625,
        antithetic_sampling: bool = True,
        **kwargs,  # noqa: ARG002 -- the remaining batch keys
    ):
        if position_ids is None:
            position_ids = torch.arange(
                input_ids.shape[1], device=input_ids.device
            ).unsqueeze(0)
        blocks = self._sample_blocks(
            input_ids, loss_mask, document_ids, position_ids, max_anchors
        )
        ctx_mask = self._context_mask(document_ids, blocks.anchor_positions)
        ctx_kv = self.project_context(
            hidden_states, input_ids, position_ids, blocks.anchor_positions
        )
        if self.config.flow_map:
            loss, metrics = self._semigroup_loss(
                blocks,
                ctx_kv,
                ctx_mask,
                hidden_states,
                input_ids,
                position_ids,
                gamma=gamma,
                loss_config=loss_config,
                diagonal_fraction=diagonal_fraction,
                boundary_prob=boundary_prob,
                antithetic_sampling=antithetic_sampling,
            )
        else:
            loss, metrics = self._flow_loss(
                blocks, ctx_kv, ctx_mask, hidden_states.dtype, gamma=gamma
            )
        if not self.training and verifier_last_hidden_states is not None:
            metrics.update(
                self._draft_metrics(
                    blocks, ctx_kv, ctx_mask, verifier_last_hidden_states
                )
            )
        return None, loss, metrics

    def _flow_loss(self, blocks, ctx_kv, ctx_mask, state_dtype, gamma):
        """Stage 1: every block noised to its own time, denoised to its tokens.

        The vocabulary-sized state is carried in the compute dtype."""
        num_blocks = blocks.anchor_positions.numel()
        taus, times = self.time_schedule.sample((num_blocks,), blocks.labels.device)
        slot_times = times.repeat_interleave(self.block_size).unsqueeze(0)
        with torch.no_grad():
            state = self._interpolant(blocks.labels, slot_times, state_dtype)
        logits = self.denoise(state, slot_times, blocks.position_ids, ctx_kv, ctx_mask)
        del state
        return compute_flow_loss(
            logits, blocks.labels, blocks.label_mask, taus, self.block_size, gamma
        )

    def _semigroup_loss(
        self,
        blocks: AnchoredBlocks,
        ctx_kv: ContextKV,
        ctx_mask,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        gamma: float,
        loss_config: LossConfig | None,
        diagonal_fraction: float,
        boundary_prob: float,
        antithetic_sampling: bool,
    ):
        """Stage 2: the paper's semigroup distillation (eq. 23, Algorithm 7).

            L = E KL( sg(gamma delta_{s,u}(I_s) + (1 - gamma) delta_{u,t}(X_{s,u}))
                      || delta_{s,t}(I_s) )                 off the diagonal,
              + E KL( D_t(I_t) || delta_{t,t}(I_t) )         on the diagonal,

        with ``gamma = (1 - t)(u - s) / ((1 - u)(t - s))``, ``D`` the frozen flow
        and ``X_{s,u}`` the map's jump recovered from ``delta_{s,u}`` (eq. 19). The
        semigroup target is the student's own two half-jumps under stop-gradient.
        """
        if not self._teacher:
            raise RuntimeError(
                "Training a FlashMap flow map needs its frozen flow: pass "
                "--flashmap-teacher <stage-1 checkpoint>."
            )
        teacher = self._teacher[0]
        if teacher.fc.weight.device != input_ids.device:
            teacher.to(input_ids.device)  # type: ignore[arg-type]
        num_blocks = blocks.anchor_positions.numel()
        kind, s, u, t = self._sample_semigroup_times(
            num_blocks,
            input_ids.device,
            diagonal_fraction=diagonal_fraction,
            boundary_prob=boundary_prob,
            antithetic_sampling=antithetic_sampling,
        )

        def per_slot(x: torch.Tensor) -> torch.Tensor:
            return x.repeat_interleave(self.block_size).unsqueeze(0)

        s_slot, u_slot, t_slot = (per_slot(x).float() for x in (s, u, t))
        positions = blocks.position_ids
        with torch.no_grad():
            # One interpolant per block, at its start time.
            state = self._interpolant(blocks.labels, s_slot, torch.float32)

            # Diagonal target: the frozen flow's denoiser at the same state.
            teacher_kv = teacher.project_context(
                hidden_states, input_ids, position_ids, blocks.anchor_positions
            )
            target = teacher.denoise(state, s_slot, positions, teacher_kv, ctx_mask)
            target = target.float().log_softmax(dim=-1)
            del teacher_kv

            # Off-diagonal target: two half-jumps of the current map (eq. 87).
            log_su = (
                self.denoise(state, s_slot, positions, ctx_kv, ctx_mask, end=u_slot)
                .float()
                .log_softmax(dim=-1)
            )
            coef_state = per_slot((1.0 - u) / (1.0 - s + 1e-8)).float().unsqueeze(-1)
            coef_delta = per_slot((u - s) / (1.0 - s + 1e-8)).float().unsqueeze(-1)
            x_su = coef_state * state + coef_delta * log_su.exp()
            log_ut = (
                self.denoise(x_su, u_slot, positions, ctx_kv, ctx_mask, end=t_slot)
                .float()
                .log_softmax(dim=-1)
            )
            del x_su
            # The 1e-8 (the official code's) also makes gamma 0 on the diagonal.
            mix = (1.0 - t) * (u - s) / ((1.0 - u) * (t - s) + 1e-8)
            mix = per_slot(mix).float().unsqueeze(-1)
            semigroup = torch.logaddexp(log_su + mix.log(), log_ut + torch.log1p(-mix))
            del log_su, log_ut
            on_diagonal = per_slot(kind == DIAGONAL).unsqueeze(-1)
            target = torch.where(on_diagonal, target, semigroup)
            del semigroup

        # One evaluation with gradients: delta_{s,t}(I_s) for every block.
        prediction = self.denoise(
            state, s_slot, positions, ctx_kv, ctx_mask, end=t_slot
        )
        groups = {
            "diag": per_slot(kind == DIAGONAL),
            # As in the official code, the off-diagonal term includes the boundary
            # blocks; the boundary, the jump drafting takes, is also its own group.
            "off": per_slot(kind != DIAGONAL),
            "boundary": per_slot(kind == BOUNDARY),
        }
        return compute_map_loss(
            prediction,
            target,
            blocks.labels,
            blocks.label_mask,
            groups,
            self.block_size,
            gamma,
            loss_config,
        )

    def _sample_semigroup_times(
        self,
        num_blocks: int,
        device: torch.device,
        diagonal_fraction: float,
        boundary_prob: float,
        antithetic_sampling: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-block category and float64 times ``s <= u <= t``.

        As in the paper's section 4 and the official implementation, in the
        schedule's ``tau``: a ``diagonal_fraction`` of the blocks (exact count,
        random placement) get ``tau_s = tau_t ~ U[0, 1]``; of the rest, a
        ``boundary_prob`` get ``(s, t) = (0, 1)``, the jump drafting takes, and
        the others ``h ~ U[0, 1]``, ``tau_s ~ U[0, 1 - h]``, ``tau_t = tau_s + h``;
        ``tau_u`` is the midpoint. Uniform draws are stratified over the batch
        when ``antithetic_sampling`` is set.
        """

        def uniform() -> torch.Tensor:
            draws = torch.rand(num_blocks, dtype=torch.float64, device=device)
            if antithetic_sampling:
                offset = torch.arange(num_blocks, dtype=torch.float64, device=device)
                perm = torch.randperm(num_blocks, device=device)
                draws = ((draws + offset) / num_blocks)[perm]
            return draws

        num_diagonal = int(num_blocks * diagonal_fraction)
        num_boundary = (
            int(
                num_blocks
                * (diagonal_fraction + (1.0 - diagonal_fraction) * boundary_prob)
            )
            - num_diagonal
        )
        kind = torch.full((num_blocks,), OFF_DIAGONAL, dtype=torch.long, device=device)
        kind[:num_diagonal] = DIAGONAL
        kind[num_diagonal : num_diagonal + num_boundary] = BOUNDARY
        kind = kind[torch.randperm(num_blocks, device=device)]
        diagonal, boundary = kind == DIAGONAL, kind == BOUNDARY

        tau_diagonal = uniform()
        h = uniform()
        tau_s_off = uniform() * (1.0 - h)
        zero, one = torch.zeros_like(h), torch.ones_like(h)
        tau_s = torch.where(
            diagonal, tau_diagonal, torch.where(boundary, zero, tau_s_off)
        )
        tau_t = torch.where(
            diagonal, tau_diagonal, torch.where(boundary, one, tau_s_off + h)
        )
        tau_s, tau_t = tau_s.clamp(0.0, 1.0), tau_t.clamp(0.0, 1.0)
        tau_u = 0.5 * (tau_s + tau_t)
        s, u, t = (self.time_schedule.t(tau).double() for tau in (tau_s, tau_u, tau_t))
        # The schedule's inverse stops just short of t = 1, so set the boundary
        # pair exactly.
        s = torch.where(boundary, zero, s)
        t = torch.where(boundary, one, t)
        return kind, s, u, t

    @torch.no_grad()
    def _draft_metrics(self, blocks, ctx_kv, ctx_mask, verifier_last_hidden_states):
        """One-step drafts of every block (the flow's denoiser at ``t = 0``, or the
        map's jump ``delta_{0,1}``) scored against the verifier's greedy tokens."""
        shape = blocks.labels.shape
        noise = torch.randn(*shape, self.draft_vocab_size, device=blocks.labels.device)
        start = torch.zeros(shape, device=noise.device)
        end = torch.ones(shape, device=noise.device) if self.config.flow_map else None
        draft_logits = self.denoise(
            noise, start, blocks.position_ids, ctx_kv, ctx_mask, end=end
        )
        # The verifier's distribution for a slot is read at the position before it.
        target_ids = self.verifier_lm_head(
            self.verifier_norm(verifier_last_hidden_states[:, blocks.label_indices - 1])
        ).argmax(dim=-1)
        return compute_draft_metrics(
            draft_logits, target_ids, blocks.label_mask, self.block_size
        )
