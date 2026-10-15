"""FlashMap checkpoint converter: a trained checkpoint to a FlashMap release folder.

``speculators train --speculator-type flashmap`` saves speculators checkpoints. The
drafter, the flow map of stage 2, is published and served in the FlashMap release
format, which the ``flashmap`` package (https://github.com/al-jshen/flashmap) serves
in vLLM (``flashmap[vllm]``) and loads with Hugging Face ``transformers``:

* ``config.json``: ``model_type: "flashmap"`` with the drafter's fields, plus the
  speculators-style header (``speculators_model_type``, ``speculators_config``, an
  empty ``transformer_layer_config``) and ``draft_vocab_size`` that vLLM reads, and
  ``max_position_embeddings`` and ``dspark_config`` that SGLang reads;
* ``model.safetensors``: the trained weights under their own names, without the
  verifier's embedding and LM head, in bf16;
* optionally ``draft_vocab.safetensors``: the verifier's tokens ranked by frequency
  (``ids``, most frequent first, and their ``counts``), whose first
  ``draft_vocab_size`` the serving integrations draft over;
* with the ``flashmap`` package installed, its model code
  (``configuration_flashmap.py``, ``modeling_flashmap.py`` and ``auto_map``), so that
  ``AutoModel.from_pretrained(..., trust_remote_code=True)`` works without it.

The weights are not renamed: a speculators FlashMap checkpoint already uses the
release names. Only the configuration is rewritten.
"""

import importlib.util
import json
import shutil
from pathlib import Path

import torch
import transformers
from loguru import logger
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import PretrainedConfig

from speculators.convert.utils import (
    ensure_checkpoint_is_local,
    load_checkpoint_config,
    load_checkpoint_weights,
)
from speculators.models.flashmap.config import DEFAULT_TIME_GRIDS

__all__ = [
    "DEFAULT_DRAFT_VOCAB_SIZE",
    "FlashMapConverter",
    "release_config",
    "release_weight_names",
    "token_ranking",
]

# Draft vocabulary of the serving integrations when a ranking is given.
DEFAULT_DRAFT_VOCAB_SIZE = 65536
# Positions a served draft accepts: Qwen3's context with YaRN (and vLLM's default).
MAX_POSITION_EMBEDDINGS = 131072
RELEASE_DRAFT_VOCAB = "draft_vocab.safetensors"
# The flashmap package's model code, which a release folder ships for
# trust_remote_code.
REMOTE_CODE = ("configuration_flashmap.py", "modeling_flashmap.py")

# The verifier's weights and the speculators vocabulary maps are not part of a
# release folder.
_DROPPED = frozenset(
    {"embed_tokens.weight", "lm_head.weight", "verifier_lm_head.weight", "d2t", "t2d"}
)


def release_weight_names(num_layers: int) -> set[str]:
    """The weights of a FlashMap release folder with ``num_layers`` DiT blocks."""
    names = {
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
        "final_adaln.linear.weight",
        "final_adaln.linear.bias",
        "head.weight",
        "head.bias",
    }
    for embed in ("time_embed", "jump_embed"):
        names |= {f"{embed}.linear_{i}.{p}" for i in (1, 2) for p in ("weight", "bias")}
    for n in range(num_layers):
        for adaln in ("self_attn_adaln", "cross_attn_adaln", "mlp_adaln"):
            names |= {f"layers.{n}.{adaln}.linear.{p}" for p in ("weight", "bias")}
        for attn in ("self_attn", "cross_attn"):
            names |= {
                f"layers.{n}.{attn}.{p}.weight"
                for p in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm")
            }
        names |= {
            f"layers.{n}.mlp.fc{i}.{p}" for i in (1, 2) for p in ("weight", "bias")
        }
    return names


def release_config(
    source: dict,
    weights: dict[str, torch.Tensor],
    target_model: str,
    target_architectures: list[str],
    num_steps: int | None = None,
    draft_vocab_size: int | None = None,
) -> dict:
    """The release config.json of a speculators FlashMap checkpoint's config
    (``source``) and weights."""
    layer = source["transformer_layer_config"]
    rope = layer.get("rope_parameters") or {}
    hidden_size, target_hidden_size = weights["x_embed.weight"].shape
    cond_dim, time_embed_dim = weights["time_embed.linear_1.weight"].shape
    head_dim = weights["layers.0.self_attn.q_norm.weight"].shape[0]
    block_size = int(source["block_size"])
    grids = source.get("time_grids") or DEFAULT_TIME_GRIDS
    num_layers = 1 + max(
        int(k.split(".")[1]) for k in weights if k.startswith("layers.")
    )
    config = {
        "model_type": "flashmap",
        "architectures": ["FlashMapDraftModel"],
        "target_model": target_model,
        "target_hidden_size": int(target_hidden_size),
        "vocab_size": int(layer["vocab_size"]),
        # speculators counts the decoder layer whose input it captures, the
        # release format the decoder layer whose output it is: one earlier.
        "target_layer_ids": [int(i) - 1 for i in source["aux_hidden_state_layer_ids"]],
        "block_size": block_size,
        "hidden_size": int(hidden_size),
        "num_hidden_layers": num_layers,
        "num_attention_heads": int(hidden_size) // int(head_dim),
        "intermediate_size": int(weights["layers.0.mlp.fc1.weight"].shape[0]),
        "cond_dim": int(cond_dim),
        "time_embed_dim": int(time_embed_dim),
        "time_scale": float(source.get("time_scale", 1000.0)),
        "rope_theta": float(rope.get("rope_theta", layer.get("rope_theta", 10000.0))),
        "rms_norm_eps": float(layer.get("rms_norm_eps", 1e-6)),
        "qk_norm_eps": float(source.get("qk_norm_eps", 1e-5)),
        "layer_norm_eps": float(source.get("adaln_eps", 1e-6)),
        "noise_gate": source.get("noise_gate", "jump"),
        "time_schedule": source.get("time_schedule", "decoding_error"),
        "num_steps": int(num_steps or source.get("num_steps", 1)),
        "jump_grids": {str(n): list(grid) for n, grid in enumerate(grids, start=1)},
        "dtype": "bfloat16",
        "tie_word_embeddings": False,
        "transformers_version": transformers.__version__,
        # What vLLM reads (with flashmap[vllm]): a speculators-style header and
        # the restricted draft vocabulary.
        "speculators_model_type": "flashmap",
        "speculators_config": {
            "algorithm": "flashmap",
            "default_proposal_method": "greedy",
            "proposal_methods": [
                {
                    "proposal_type": "greedy",
                    "speculative_tokens": block_size,
                    "accept_tolerance": 0.0,
                    "verifier_accept_k": 1,
                }
            ],
            "verifier": {
                "name_or_path": target_model,
                "architectures": list(target_architectures),
            },
        },
        "transformer_layer_config": {},
        # What SGLang's DSPARK worker reads (with flashmap[sglang]): the longest
        # context it may serve, and DSPARK's own fields: a rank-1 "vanilla" Markov
        # head, which flashmap replaces by an identity, and the token DSPARK fills
        # the block's slots with, which FlashMap never reads.
        "max_position_embeddings": MAX_POSITION_EMBEDDINGS,
        "dspark_config": {
            "markov_rank": 1,
            "markov_head_type": "vanilla",
            "mask_token_id": _mask_token_id(target_model, int(layer["vocab_size"])),
        },
    }
    if draft_vocab_size:
        config["draft_vocab_size"] = int(draft_vocab_size)
    return config


def _mask_token_id(target_model: str, vocab_size: int) -> int:
    """A token id outside the target tokenizer's vocabulary (151669 for Qwen3): the
    first past it, or the last embedding row if the tokenizer is not reachable."""
    try:
        tokens = len(transformers.AutoTokenizer.from_pretrained(target_model))
    except Exception:
        return vocab_size - 1
    return tokens if tokens < vocab_size else vocab_size - 1


def token_ranking(path: str, vocab_size: int) -> dict[str, torch.Tensor]:
    """``ids`` (every token id, most frequent first) and their ``counts``, from a
    ranking (safetensors with ``ids`` and ``counts``) or from the ``token_freq.pt``
    of ``prepare-data --no-skip-token-freq`` (token id -> count; unseen tokens
    follow by id)."""
    if str(path).endswith(".safetensors"):
        ranking = load_file(path)
        return {
            "ids": ranking["ids"].to(torch.int64),
            "counts": ranking["counts"].to(torch.int64),
        }
    freq = torch.load(path, weights_only=True)
    counts = torch.zeros(vocab_size, dtype=torch.int64)
    for token, count in freq.items():
        counts[int(token)] = int(count)
    # Stable: equal counts keep increasing ids, as speculators' vocab mapping does.
    ids = torch.sort(-counts, stable=True).indices
    return {"ids": ids, "counts": counts[ids]}


def _target_architectures(source: dict, target_model: str) -> list[str]:
    verifier = (source.get("speculators_config") or {}).get("verifier") or {}
    if verifier.get("architectures"):
        return list(verifier["architectures"])
    try:
        config, _ = PretrainedConfig.get_config_dict(target_model)
    except OSError:
        return []
    return list(config.get("architectures") or [])


def _check_trained_flow_map(source: dict, input_path: str | Path) -> None:
    if source.get("model_type") == "flashmap":
        raise ValueError(f"{input_path} is already a FlashMap release folder")
    if source.get("speculators_model_type") != "flashmap":
        raise ValueError(
            f"{input_path} is not a FlashMap checkpoint saved by speculators"
        )
    if source.get("flow_map") is False:
        raise ValueError(
            "This is a stage-1 FlashMap flow (flow_map: false), which drafts only "
            "by integrating over many steps; convert the flow map distilled from it "
            "(speculators train --flashmap-teacher <flow>)."
        )


def _draft_vocab_size(
    ranking: dict[str, torch.Tensor] | None, size: int | None, vocab_size: int
) -> int | None:
    """The restricted draft vocabulary's size, or None for the full one."""
    if ranking is None:
        if size:
            raise ValueError("draft_vocab_size needs a draft_vocab_file ranking")
        return None
    size = size or DEFAULT_DRAFT_VOCAB_SIZE
    return size if size < vocab_size else None


class FlashMapConverter:
    """Convert a trained speculators FlashMap checkpoint to a release folder."""

    def convert(
        self,
        input_path: str | Path,
        output_path: str | Path,
        base_model: str,
        validate: bool = True,
        num_steps: int | None = None,
        draft_vocab_size: int | None = None,
        draft_vocab_file: str | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        """Write the flow map at ``input_path`` (a ``speculators train``
        checkpoint) as a FlashMap release folder at ``output_path``.

        :param base_model: the verifier (target) model the drafter was trained for.
        :param num_steps: default number of jumps (1 unless the checkpoint says).
        :param draft_vocab_file: a token ranking (``ids``/``counts`` safetensors)
            or ``prepare-data``'s ``token_freq.pt``, shipped as
            ``draft_vocab.safetensors``.
        :param draft_vocab_size: how many of its most frequent tokens vLLM drafts
            over (default 65536 with a ranking; without one, the full vocabulary).
        """
        logger.info(f"Converting FlashMap checkpoint: {input_path}")
        local_path = ensure_checkpoint_is_local(input_path, cache_dir)
        source = load_checkpoint_config(local_path)
        _check_trained_flow_map(source, input_path)
        weights = {
            name: tensor.to(torch.bfloat16) if tensor.is_floating_point() else tensor
            for name, tensor in load_checkpoint_weights(local_path).items()
            if name not in _DROPPED
        }
        vocab_size = int(source["transformer_layer_config"]["vocab_size"])
        ranking = (
            token_ranking(draft_vocab_file, vocab_size) if draft_vocab_file else None
        )
        draft_vocab_size = _draft_vocab_size(ranking, draft_vocab_size, vocab_size)
        config = release_config(
            source,
            weights,
            base_model,
            _target_architectures(source, base_model),
            num_steps=num_steps,
            draft_vocab_size=draft_vocab_size,
        )

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        code = importlib.util.find_spec("flashmap")
        if code is not None and code.origin is not None:
            for name in REMOTE_CODE:
                shutil.copy(Path(code.origin).parent / name, output_path / name)
            config["auto_map"] = {
                "AutoConfig": "configuration_flashmap.FlashMapConfig",
                "AutoModel": "modeling_flashmap.FlashMapDraftModel",
            }
        else:
            logger.info(
                "The flashmap package is not installed, so the folder ships without "
                "its model code: load it with transformers after `import flashmap`."
            )
        (output_path / "config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n"
        )
        save_file(
            weights, str(output_path / "model.safetensors"), metadata={"format": "pt"}
        )
        if ranking is not None:
            save_file(
                ranking,
                str(output_path / RELEASE_DRAFT_VOCAB),
                metadata={
                    "description": "target response-token frequency ranking: ids "
                    "= token ids, most frequent first; counts[i] = occurrences of "
                    "ids[i]",
                    "target": base_model,
                },
            )
        logger.success(f"Saved to: {output_path}")
        if validate:
            self._validate(output_path, weights, config)

    @staticmethod
    def _validate(
        output_path: Path, weights: dict[str, torch.Tensor], config: dict
    ) -> None:
        logger.info("Validating the FlashMap release folder...")
        expected = release_weight_names(config["num_hidden_layers"])
        if set(weights) != expected:
            raise ValueError(
                "Release weights differ from the FlashMap layout: missing "
                f"{sorted(expected - set(weights))}, unexpected "
                f"{sorted(set(weights) - expected)}"
            )
        with safe_open(str(output_path / "model.safetensors"), "pt") as handle:
            for name in handle.keys():  # noqa: SIM118
                saved = handle.get_tensor(name)
                if not torch.equal(saved, weights[name]):
                    raise ValueError(f"{name} was not saved as converted")
                if saved.is_floating_point() and not torch.isfinite(saved).all():
                    raise ValueError(f"The release folder has non-finite {name}")
        logger.success(
            "Validation succeeded: "
            + json.dumps(
                {"layers": config["num_hidden_layers"], "weights": len(weights)}
            )
        )
