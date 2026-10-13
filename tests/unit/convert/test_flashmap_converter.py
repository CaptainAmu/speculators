"""Unit tests for FlashMapConverter weight renaming and config building."""

from unittest.mock import patch

import torch

from speculators import SpeculatorModelConfig
from speculators.convert.flashmap.converter import (
    FlashMapConverter,
    deepspec_to_canonical,
)
from speculators.models.flashmap import FlashMapSpeculatorConfig
from speculators.models.flashmap.config import DEFAULT_TIME_GRIDS

_DIM = 8
_HIDDEN = 12
_COND = 4


def _deepspec_state(num_layers: int = 2) -> dict[str, torch.Tensor]:
    """A tiny DeepSpec flow-map state dict (bf16, random values)."""
    shapes = {
        "fc.weight": (_DIM, 2 * _HIDDEN),
        "hidden_norm.scale": (_DIM,),
        "anchor_proj.weight": (_DIM, _HIDDEN),
        "anchor_proj.bias": (_DIM,),
        "anchor_norm.scale": (_DIM,),
        "state_norm.scale": (_HIDDEN,),
        "noise_norm.scale": (_HIDDEN,),
        "noise_gate": (1,),
        "embed_tokens.weight": (32, _HIDDEN),
        "lm_head.weight": (32, _HIDDEN),
        "denoiser.x_embed.weight": (_DIM, _HIDDEN),
        "denoiser.x_embed.bias": (_DIM,),
        "denoiser.head.weight": (_HIDDEN, _DIM),
        "denoiser.head.bias": (_HIDDEN,),
        "denoiser.final_norm.cond_proj.1.weight": (3 * _DIM, _COND),
        "denoiser.final_norm.cond_proj.1.bias": (3 * _DIM,),
    }
    for temb in ("temb", "temb_jump"):
        for i in (0, 2):
            shapes[f"denoiser.{temb}.mlp.{i}.weight"] = (_COND, _COND)
            shapes[f"denoiser.{temb}.mlp.{i}.bias"] = (_COND,)
    for n in range(num_layers):
        block = f"denoiser.blocks.{n}"
        for adaln in ("adaln_self", "adaln_cross", "adaln_mlp"):
            shapes[f"{block}.{adaln}.cond_proj.1.weight"] = (3 * _DIM, _COND)
            shapes[f"{block}.{adaln}.cond_proj.1.bias"] = (3 * _DIM,)
        for attn in ("self_attn", "cross_attn"):
            shapes[f"{block}.{attn}.wq.weight"] = (_DIM, _DIM)
            shapes[f"{block}.{attn}.wkv.weight"] = (2 * _DIM, _DIM)
            shapes[f"{block}.{attn}.wo.weight"] = (_DIM, _DIM)
            shapes[f"{block}.{attn}.qnorm.scale"] = (4,)
            shapes[f"{block}.{attn}.knorm.scale"] = (4,)
        shapes[f"{block}.mlp.0.weight"] = (4 * _DIM, _DIM)
        shapes[f"{block}.mlp.0.bias"] = (4 * _DIM,)
        shapes[f"{block}.mlp.3.weight"] = (_DIM, 4 * _DIM)
        shapes[f"{block}.mlp.3.bias"] = (_DIM,)
    generator = torch.Generator().manual_seed(0)
    return {
        name: torch.randn(shape, generator=generator).to(torch.bfloat16)
        for name, shape in shapes.items()
    }


def test_deepspec_to_canonical_names_and_values():
    source = _deepspec_state()
    weights = deepspec_to_canonical(source)

    assert "embed_tokens.weight" not in weights
    assert "lm_head.weight" not in weights
    # RMSNorm scales become multiplicative weights, formed in bf16.
    assert torch.equal(weights["hidden_norm.weight"], 1 + source["hidden_norm.scale"])
    assert torch.equal(
        weights["layers.1.cross_attn.k_norm.weight"],
        1 + source["denoiser.blocks.1.cross_attn.knorm.scale"],
    )
    # Fused key/value projections are split, keys first.
    keys, values = source["denoiser.blocks.0.self_attn.wkv.weight"].chunk(2)
    assert torch.equal(weights["layers.0.self_attn.k_proj.weight"], keys)
    assert torch.equal(weights["layers.0.self_attn.v_proj.weight"], values)
    assert torch.equal(
        weights["layers.1.mlp_adaln.linear.bias"],
        source["denoiser.blocks.1.adaln_mlp.cond_proj.1.bias"],
    )
    assert torch.equal(
        weights["jump_embed.linear_2.weight"],
        source["denoiser.temb_jump.mlp.2.weight"],
    )
    assert torch.equal(
        weights["final_adaln.linear.weight"],
        source["denoiser.final_norm.cond_proj.1.weight"],
    )
    assert torch.equal(weights["noise_gate"], source["noise_gate"])
    # Every source tensor but the verifier's lands somewhere (wkv as two).
    assert len(weights) == len(source) - 2 + 2 * 2


def test_build_config_from_deepspec():
    weights = deepspec_to_canonical(_deepspec_state())
    source = {"block_size": 4, "target_layer_ids": [0, 1], "noise_gate": "jump"}
    verifier = {"hidden_size": _HIDDEN, "vocab_size": 32, "architectures": ["X"]}
    with patch(
        "speculators.convert.flashmap.converter.PretrainedConfig.get_config_dict",
        return_value=(verifier, {}),
    ):
        config = FlashMapConverter()._build_config(source, weights, "verifier", 1)

    assert isinstance(config, FlashMapSpeculatorConfig)
    assert config.aux_hidden_state_layer_ids == [1, 2]
    assert config.block_size == 4
    assert config.target_hidden_size == _HIDDEN
    assert config.transformer_layer_config.hidden_size == _DIM
    assert config.transformer_layer_config.num_hidden_layers == 2
    assert config.timestep_embed_dim == _COND
    assert config.time_grids == DEFAULT_TIME_GRIDS
    spec = config.speculators_config
    assert spec.proposal_methods[0].speculative_tokens == 4

    reloaded = SpeculatorModelConfig.from_dict(config.to_dict())
    assert isinstance(reloaded, FlashMapSpeculatorConfig)
    assert reloaded.time_grid(2) == DEFAULT_TIME_GRIDS[1]
