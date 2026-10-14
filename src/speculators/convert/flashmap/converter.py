"""FlashMap checkpoint converter.

Converts a FlashMap checkpoint to a Speculators checkpoint that vLLM serves
(``vllm serve <output>``, see ``speculators.models.flashmap.vllm``). Three sources
are accepted:

* a speculators FlashMap checkpoint (``speculators_model_type: "flashmap"``,
  e.g. one ``speculators train`` saved), which vLLM serves as it is; converting
  it only adds a restricted draft vocabulary;
* a released FlashMap checkpoint (``model_type: "flashmap"``), whose weights
  already carry the canonical names and are copied as they are;
* a DeepSpec training checkpoint (a ``Qwen3DFlashier*FlowMapModel``), whose
  weights are renamed to the canonical names: RMSNorm ``scale`` becomes the
  multiplicative ``weight = 1 + scale`` (exact in bf16), fused ``wkv``
  projections are split into ``k_proj``/``v_proj``, and the target's frozen
  ``embed_tokens``/``lm_head`` copies are dropped (they are the verifier's).

Optionally the draft head is restricted to the verifier's most frequent tokens:
``draft_vocab_file`` holds a ranking (safetensors key ``ids``) or speculators'
token frequencies (``token_freq.pt``, as ``prepare-data`` writes them), and the
first ``draft_vocab_size`` tokens become the draft vocabulary, stored as the
speculators ``d2t``/``t2d`` mappings.
"""

import json
from pathlib import Path

import torch
from loguru import logger
from safetensors.torch import load_file, save_file
from transformers import PretrainedConfig

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.convert.utils import (
    ensure_checkpoint_is_local,
    load_checkpoint_config,
    load_checkpoint_weights,
)
from speculators.models.flashmap import FlashMapSpeculatorConfig
from speculators.models.flashmap.config import DEFAULT_TIME_GRIDS
from speculators.proposals.greedy import GreedyTokenProposalConfig
from speculators.train.vocab_mapping import build_vocab_mappings_from_distribution

__all__ = ["FlashMapConverter", "deepspec_to_canonical"]

# Weights the verifier provides at runtime.
_VERIFIER_KEYS = frozenset({"embed_tokens.weight", "lm_head.weight"})


def deepspec_to_canonical(  # noqa: C901
    weights: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Rename a DeepSpec flow-map state dict to the canonical FlashMap names."""
    out: dict[str, torch.Tensor] = {}
    sub = {"self_attn": "self_attn", "cross_attn": "cross_attn"}
    adaln = {
        "adaln_self": "self_attn_adaln",
        "adaln_cross": "cross_attn_adaln",
        "adaln_mlp": "mlp_adaln",
    }
    for source_name, source_tensor in weights.items():
        if source_name in _VERIFIER_KEYS:
            continue
        name, tensor = source_name, source_tensor
        if name.endswith(".scale"):
            # DeepSpec RMSNorm multiplies by (1 + scale), formed in bf16.
            tensor = 1 + tensor
            name = name.removesuffix(".scale") + ".weight"
        parts = name.split(".")
        if parts[0] != "denoiser":
            out[name] = tensor
            continue
        parts = parts[1:]
        if parts[0] in ("temb", "temb_jump"):
            # mlp.{0,2}.{weight,bias}
            target = "time_embed" if parts[0] == "temb" else "jump_embed"
            linear = {"0": "linear_1", "2": "linear_2"}[parts[2]]
            out[f"{target}.{linear}.{parts[3]}"] = tensor
        elif parts[0] == "final_norm":
            out[f"final_adaln.linear.{parts[-1]}"] = tensor
        elif parts[0] in ("x_embed", "head"):
            out[".".join(parts)] = tensor
        elif parts[0] == "blocks":
            layer, module = parts[1], parts[2]
            prefix = f"layers.{layer}"
            if module in adaln:
                out[f"{prefix}.{adaln[module]}.linear.{parts[-1]}"] = tensor
            elif module in sub:
                proj = parts[3]
                if proj == "wkv":
                    keys, values = tensor.chunk(2, dim=0)
                    out[f"{prefix}.{module}.k_proj.weight"] = keys
                    out[f"{prefix}.{module}.v_proj.weight"] = values
                elif proj in ("wq", "wo"):
                    out[f"{prefix}.{module}.{proj[1]}_proj.weight"] = tensor
                else:  # qnorm / knorm
                    out[f"{prefix}.{module}.{proj[0]}_norm.weight"] = tensor
            elif module == "mlp":
                linear = {"0": "fc1", "3": "fc2"}[parts[3]]
                out[f"{prefix}.mlp.{linear}.{parts[-1]}"] = tensor
            else:
                raise ValueError(f"unexpected DeepSpec weight {name!r}")
        else:
            raise ValueError(f"unexpected DeepSpec weight {name!r}")
    return {name: tensor.contiguous() for name, tensor in out.items()}


class FlashMapConverter:
    """Convert a FlashMap or DeepSpec flow-map checkpoint to speculators format."""

    def convert(
        self,
        input_path: str | Path,
        output_path: str | Path,
        base_model: str,
        validate: bool = True,
        num_steps: int = 1,
        draft_vocab_size: int | None = None,
        draft_vocab_file: str | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        logger.info(f"Converting FlashMap checkpoint: {input_path}")
        local_path = ensure_checkpoint_is_local(input_path, cache_dir)
        source_config = load_checkpoint_config(local_path)
        weights = load_checkpoint_weights(local_path)
        if "denoiser.x_embed.weight" in weights:
            weights = deepspec_to_canonical(weights)
        else:
            weights = {k: v for k, v in weights.items() if k not in _VERIFIER_KEYS}
        logger.info(f"Loaded {len(weights)} weights")

        config = self._build_config(source_config, weights, base_model, num_steps)
        if draft_vocab_size is not None and draft_vocab_size < config.target_vocab_size:
            if draft_vocab_file is None:
                raise ValueError("draft_vocab_size needs a draft_vocab_file ranking")
            weights.update(
                self._draft_vocab(draft_vocab_file, draft_vocab_size, config)
            )
            config.draft_vocab_size = draft_vocab_size

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        config.save_pretrained(str(output_path))
        save_file(weights, str(output_path / "model.safetensors"))
        logger.success(f"Saved to: {output_path}")
        if validate:
            self._validate(output_path, weights)

    def _build_config(
        self,
        source: dict,
        weights: dict[str, torch.Tensor],
        base_model: str,
        num_steps: int,
    ) -> FlashMapSpeculatorConfig:
        verifier_config, _ = PretrainedConfig.get_config_dict(base_model)
        released = source.get("model_type") == "flashmap"
        dim = weights["x_embed.weight"].shape[0]
        target_hidden = weights["x_embed.weight"].shape[1]
        if target_hidden != verifier_config["hidden_size"]:
            raise ValueError(
                f"checkpoint was trained on hidden size {target_hidden}, but "
                f"{base_model} has {verifier_config['hidden_size']}"
            )
        if source.get("speculators_model_type") == "flashmap":
            # Already this schema (e.g. saved by speculators train).
            if source.get("flow_map") is False:
                raise ValueError(
                    "this is a stage-1 FlashMap flow (flow_map: false); convert "
                    "the flow map distilled from it"
                )
            config = FlashMapSpeculatorConfig.model_validate(dict(source))
            config.num_steps = num_steps
            return config
        num_layers = 1 + max(
            int(k.split(".")[1]) for k in weights if k.startswith("layers.")
        )
        head_dim = weights["layers.0.self_attn.q_norm.weight"].shape[0]
        intermediate = weights["layers.0.mlp.fc1.weight"].shape[0]
        cond_dim = weights["time_embed.linear_1.weight"].shape[0]
        frequency_dim = weights["time_embed.linear_1.weight"].shape[1]

        # DeepSpec's target_layer_ids index decoder outputs; speculators' aux ids
        # index the decoder layer whose input vLLM captures (one later).
        if "aux_hidden_state_layer_ids" in source:
            aux_layer_ids = list(source["aux_hidden_state_layer_ids"])
        else:
            aux_layer_ids = [i + 1 for i in source["target_layer_ids"]]
        block_size = int(source["block_size"])
        transformer_layer_config = {
            "model_type": "qwen3",
            "hidden_size": dim,
            "num_hidden_layers": num_layers,
            "num_attention_heads": dim // head_dim,
            "num_key_value_heads": dim // head_dim,
            "head_dim": head_dim,
            "intermediate_size": intermediate,
            "hidden_act": "gelu",
            "rms_norm_eps": float(source.get("rms_norm_eps", 1e-6)),
            "rope_parameters": {
                "rope_type": "default",
                # The released config's rope_theta is the DiT's; a DeepSpec
                # config's is the target's, its DiT's being dit_rope_base.
                "rope_theta": float(
                    source.get("dit_rope_base")
                    or (source.get("rope_theta") if released else None)
                    or 10000.0
                ),
            },
            "vocab_size": int(verifier_config["vocab_size"]),
            "max_position_embeddings": int(
                verifier_config.get("max_position_embeddings", 40960)
            ),
            "tie_word_embeddings": False,
        }
        speculators_config = SpeculatorsConfig(
            algorithm="flashmap",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=block_size)],
            default_proposal_method="greedy",
            verifier=VerifierConfig(
                name_or_path=base_model,
                architectures=verifier_config.get("architectures", []),
            ),
        )
        time_grids = (
            source.get("time_grids")
            or source.get("jump_grids")
            or [list(g) for g in DEFAULT_TIME_GRIDS]
        )
        if isinstance(time_grids, dict):  # {"1": [...], "2": [...]}
            time_grids = [time_grids[str(n)] for n in range(1, len(time_grids) + 1)]
        return FlashMapSpeculatorConfig(
            transformer_layer_config=transformer_layer_config,  # type: ignore[arg-type]
            draft_vocab_size=int(verifier_config["vocab_size"]),
            block_size=block_size,
            target_hidden_size=target_hidden,
            aux_hidden_state_layer_ids=aux_layer_ids,
            timestep_embed_dim=cond_dim,
            timestep_frequency_dim=frequency_dim if frequency_dim != cond_dim else None,
            time_scale=float(
                source.get("dit_time_scale", source.get("time_scale", 1000.0))
            ),
            qk_norm_eps=float(source.get("qk_norm_eps", 1e-5)),
            adaln_eps=float(source.get("layer_norm_eps", 1e-6)),
            noise_gate=source.get("noise_gate", "jump"),
            time_schedule=source.get("time_schedule", "decoding_error"),
            num_steps=num_steps,
            time_grids=time_grids,
            speculators_config=speculators_config,
        )

    @staticmethod
    def _draft_vocab(
        path: str, size: int, config: FlashMapSpeculatorConfig
    ) -> dict[str, torch.Tensor]:
        """speculators ``d2t`` (draft id -> target id offset) and ``t2d`` (target
        id in the draft vocabulary) for the first ``size`` ids of a ranking, or
        the ``size`` most frequent tokens of a ``token_freq.pt``."""
        if path.endswith(".pt"):
            d2t, t2d = build_vocab_mappings_from_distribution(
                torch.load(path, weights_only=True), size, config.target_vocab_size
            )
            return {"d2t": d2t, "t2d": t2d}
        ranking = load_file(path)["ids"]
        ids = ranking[:size].to(torch.int64).sort().values
        if ids.numel() != size:
            raise ValueError(f"{path} ranks {ids.numel()} tokens, fewer than {size}")
        t2d = torch.zeros(config.target_vocab_size, dtype=torch.bool)
        t2d[ids] = True
        return {"d2t": ids - torch.arange(size), "t2d": t2d}

    @staticmethod
    def _validate(output_path: Path, weights: dict[str, torch.Tensor]) -> None:
        logger.info("Validating converted FlashMap checkpoint...")
        config = FlashMapSpeculatorConfig.from_pretrained(str(output_path))
        num_layers = config.transformer_layer_config.num_hidden_layers
        expected = {
            "fc.weight",
            "hidden_norm.weight",
            "anchor_proj.weight",
            "anchor_proj.bias",
            "anchor_norm.weight",
            "state_norm.weight",
            "noise_norm.weight",
            "noise_gate",
            "x_embed.weight",
            "x_embed.bias",
            "head.weight",
            "head.bias",
            "final_adaln.linear.weight",
            "final_adaln.linear.bias",
        }
        for i in range(num_layers):
            for attn in ("self_attn", "cross_attn"):
                expected |= {
                    f"layers.{i}.{attn}.{p}.weight"
                    for p in (
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "o_proj",
                        "q_norm",
                        "k_norm",
                    )
                }
        missing = expected - set(weights)
        if missing:
            raise ValueError(f"Converted checkpoint is missing {sorted(missing)}")
        for name, tensor in weights.items():
            if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                raise ValueError(f"Converted checkpoint has non-finite {name}")
        logger.success(
            "Validation succeeded: "
            + json.dumps({"layers": num_layers, "weights": len(weights)})
        )
