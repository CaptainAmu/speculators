from typing import Literal

from pydantic import Field, model_validator

from speculators import SpeculatorModelConfig
from speculators.models.dflash.config import DFlashSpeculatorConfig

__all__ = [
    "DEFAULT_TIME_GRIDS",
    "FlashMapSpeculatorConfig",
]

# Jump grids t_0 = 0 < ... < t_n = 1 for n = 1..4 jumps under the decoding-error
# time reparameterization of the Qwen3 vocabulary (151936 tokens). Inner points
# are t(k / n); the endpoints are forced to exactly 0 and 1, the times the map
# was trained and drafts at.
DEFAULT_TIME_GRIDS: list[list[float]] = [
    [0.0, 1.0],
    [0.0, 0.8172569274902344, 1.0],
    [0.0, 0.8010796308517456, 0.8310279846191406, 1.0],
    [0.0, 0.7906050682067871, 0.8172569274902344, 0.8379518389701843, 1.0],
]


@SpeculatorModelConfig.register("flashmap")
class FlashMapSpeculatorConfig(DFlashSpeculatorConfig):
    """Configuration for FlashMap, a block drafter distilled from a discrete flow.

    FlashMap shares DFlash's conditioning: the fused target hidden states at
    ``aux_hidden_state_layer_ids`` (projected by ``fc``) and the anchor token
    (the last verified token, embedded at its own position) form the context that
    a block of ``block_size`` draft slots cross-attends to. The slots themselves
    start from Gaussian noise over the vocabulary and are mapped to token
    distributions by a DiT in ``num_steps`` jumps along ``time_grids``.

    ``transformer_layer_config`` describes the DiT: ``hidden_size`` is its width,
    ``num_hidden_layers`` its blocks, ``intermediate_size`` its MLP width and
    ``rope_parameters`` its rotary base. ``vocab_size`` is the verifier's and
    ``target_hidden_size`` the verifier's hidden size. Every slot is a draft
    token (the anchor is context, not a query), so ``sample_from_anchor`` is True
    and a block proposes ``block_size`` tokens.
    """

    speculators_model_type: Literal["flashmap"] = "flashmap"  # type: ignore[assignment]
    architectures: list[str] = Field(
        default_factory=lambda: ["FlashMapDraftModel"],
        description="Model architectures that can load these weights",
    )

    sample_from_anchor: bool = Field(
        default=True,
        description=(
            "All block_size slots are draft tokens at positions anchor + 1 .. "
            "anchor + block_size; the anchor itself is a context entry."
        ),
    )

    timestep_embed_dim: int = Field(
        default=256,
        description="Width of the sinusoidal timestep embedding and AdaLN condition.",
    )
    time_scale: float = Field(
        default=1000.0,
        description="Times are multiplied by this before the sinusoidal embedding.",
    )
    qk_norm_eps: float = Field(
        default=1e-5,
        description="Epsilon of the per-head query/key RMSNorms.",
    )
    adaln_eps: float = Field(
        default=1e-6,
        description="Epsilon of the (non-affine) LayerNorms inside AdaLN-zero.",
    )
    noise_gate: Literal["jump", "constant"] = Field(
        default="jump",
        description=(
            "Weight of the linear noise read-out of the state: the learned "
            "``noise_gate`` times the jump t - s ('jump') or alone ('constant')."
        ),
    )
    time_schedule: Literal["decoding_error"] = Field(
        default="decoding_error",
        description="Time reparameterization the jump grids are taken from.",
    )
    num_steps: int = Field(
        default=1,
        ge=1,
        description="Default number of jumps (network evaluations) per draft block.",
    )
    time_grids: list[list[float]] = Field(
        default_factory=lambda: [list(grid) for grid in DEFAULT_TIME_GRIDS],
        description="time_grids[n - 1] is the jump grid used for n jumps.",
    )

    @model_validator(mode="after")
    def check_time_grids(self) -> "FlashMapSpeculatorConfig":
        for num_jumps, grid in enumerate(self.time_grids, start=1):
            if len(grid) != num_jumps + 1 or grid[0] != 0.0 or grid[-1] != 1.0:
                raise ValueError(
                    f"time_grids[{num_jumps - 1}] must hold {num_jumps + 1} times "
                    f"from 0 to 1, got {grid}"
                )
        if self.num_steps > len(self.time_grids):
            raise ValueError(
                f"num_steps={self.num_steps} has no grid in time_grids "
                f"({len(self.time_grids)} given)"
            )
        return self

    def time_grid(self, num_steps: int | None = None) -> list[float]:
        """The jump grid for ``num_steps`` jumps (default ``self.num_steps``)."""
        return self.time_grids[(num_steps or self.num_steps) - 1]
