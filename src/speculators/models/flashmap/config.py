from typing import Literal

from pydantic import Field

from speculators import SpeculatorModelConfig
from speculators.models.dflash.config import DFlashSpeculatorConfig

__all__ = [
    "FlashMapSpeculatorConfig",
]


@SpeculatorModelConfig.register("flashmap")
class FlashMapSpeculatorConfig(DFlashSpeculatorConfig):
    """
    Configuration for the FlashMap block drafter.

    FlashMap conditions on the verifier like DFlash (auxiliary hidden states and
    the anchor token), but drafts a whole block from Gaussian noise with a DiT
    denoiser over the vocabulary simplex: a flow (stage 1) or a flow map
    distilled from it (stage 2, the released drafter), which jumps from noise to
    the block in one network evaluation.

    ``transformer_layer_config`` carries the verifier's hidden/vocab sizes and
    ``num_hidden_layers`` (the DiT depth); the DiT's own width is set by the
    ``dit_*`` fields. Every one of the ``block_size`` slots is a draft token at
    the positions after the anchor.
    """

    speculators_model_type: Literal["flashmap"] = "flashmap"  # type: ignore[assignment]
    architectures: list[str] = Field(
        default_factory=lambda: ["FlashMapDraftModel"],
        description="Model architectures that can load these weights",
    )

    block_size: int = Field(
        default=16,
        description="Number of draft tokens proposed per block.",
    )
    sample_from_anchor: bool = Field(
        default=True,
        description="Always True: every one of the block_size slots is a draft "
        "token (the anchor enters through cross-attention, not as a slot).",
    )
    dit_hidden_size: int = Field(
        default=2048,
        description="Width of the DiT denoiser (and of its cross-attention context).",
    )
    dit_num_heads: int = Field(
        default=16,
        description="Attention heads of the DiT's self- and cross-attention.",
    )
    dit_cond_dim: int = Field(
        default=256,
        description="Width of the time embeddings that drive the AdaLN modulation.",
    )
    dit_mlp_ratio: float = Field(
        default=4.0,
        description="DiT MLP hidden width as a multiple of dit_hidden_size.",
    )
    rope_theta: float = Field(
        default=10000.0,
        description="RoPE base; rotary pairs are interleaved (GPT-J style).",
    )
    norm_eps: float = Field(
        default=1e-6,
        description="Epsilon of the context/state RMSNorms and the AdaLN LayerNorms.",
    )
    qk_norm_eps: float = Field(
        default=1e-5,
        description="Epsilon of the per-head query/key RMSNorms.",
    )
    time_scale: float = Field(
        default=1000.0,
        description="Times in [0, 1] are multiplied by this before the sinusoidal "
        "embedding.",
    )
    time_schedule: Literal["decoding_error"] = Field(
        default="decoding_error",
        description="Time reparameterization used to sample training times and to "
        "grid multi-step drafting (Flow Map Language Models, eq. 25).",
    )
    flow_map: bool = Field(
        default=True,
        description="True for the distilled flow map (stage 2), which is conditioned "
        "on its end time and reads its noise through a gated linear term; False for "
        "the flow it is distilled from (stage 1).",
    )
    noise_gate: Literal["jump"] = Field(
        default="jump",
        description="How the flow map weights its noise term: by the jump t - s, so "
        "it vanishes on the diagonal where the map is the flow's denoiser.",
    )
    num_steps: int = Field(
        default=1,
        description="Default number of map jumps (network evaluations) per draft.",
    )
