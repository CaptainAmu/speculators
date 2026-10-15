"""FlashMap support for vLLM, as a vLLM general plugin.

Installing speculators registers ``register`` under the ``vllm.general_plugins``
entry point, which vLLM calls in every process before it parses a model. It
teaches vLLM three things without patching its source:

* the ``flashmap`` speculators format: a speculators checkpoint with
  ``speculators_model_type: "flashmap"`` maps onto vLLM's DFlash runtime
  (``method="dflash"``) with the ``FlashMapDraftModel`` architecture;
* the ``FlashMapDraftModel`` architecture (model.py);
* the speculator that drives it on the V2 model runner (speculator.py), chosen
  by architecture the way vLLM chooses its DFlash2 speculator.

Every hook delegates to vLLM's own code unless a FlashMap checkpoint is in use.
The plugin supports vLLM 0.31.x; on other versions, or if registering fails, it
logs a warning and leaves vLLM untouched.

Serve with ``vllm serve <speculators checkpoint>``, or pass the checkpoint as
``--speculative-config '{"model": "<checkpoint>", "method": "dflash"}'``.
"""

import logging
import sys

__all__ = ["FLASHMAP_ARCHITECTURE", "SUPPORTED_VLLM", "register", "update_flashmap"]

logger = logging.getLogger(__name__)

FLASHMAP_ARCHITECTURE = "FlashMapDraftModel"
# vLLM (major, minor) versions whose internals the plugin hooks into.
SUPPORTED_VLLM = ((0, 31),)

# Speculators config fields the vLLM draft config carries over.
_FLASHMAP_FIELDS = (
    "target_hidden_size",
    "draft_vocab_size",
    "block_size",
    "timestep_embed_dim",
    "timestep_frequency_dim",
    "time_scale",
    "qk_norm_eps",
    "adaln_eps",
    "noise_gate",
    "time_schedule",
    "num_steps",
    "time_grids",
)


def update_flashmap(config_dict: dict, pre_trained_config: dict) -> None:
    """Map a ``flashmap`` speculators config onto the dict vLLM builds the draft
    model's config from (cf. vLLM's ``update_dflash``)."""
    if config_dict.get("flow_map") is False:
        raise ValueError(
            "This FlashMap checkpoint is a stage-1 flow (flow_map: false), which "
            "drafts only by integrating over many steps; serve the flow map "
            "distilled from it (speculators train --flashmap-teacher <flow>)."
        )
    pre_trained_config["architectures"] = [FLASHMAP_ARCHITECTURE]
    aux_layer_ids = config_dict["aux_hidden_state_layer_ids"]
    pre_trained_config["eagle_aux_hidden_state_layer_ids"] = aux_layer_ids
    pre_trained_config["dflash_config"] = {
        # FlashMap's slots are not embedded token ids; any valid id fills them.
        "mask_token_id": config_dict.get("mask_token_id") or 0,
        "target_layer_ids": [i - 1 for i in aux_layer_ids],
        "causal": False,
    }
    for key in _FLASHMAP_FIELDS:
        if config_dict.get(key) is not None:
            pre_trained_config[key] = config_dict[key]


def _init_speculator(init_speculator):
    def init_flashmap_or_other(vllm_config, device):
        spec = vllm_config.speculative_config
        if (
            spec is not None
            and spec.method == "dflash"
            and FLASHMAP_ARCHITECTURE in (spec.draft_model_config.architectures or [])
        ):
            from speculators.models.flashmap.vllm.speculator import (  # noqa: PLC0415
                FlashMapSpeculator,
            )

            return FlashMapSpeculator(vllm_config, device)
        return init_speculator(vllm_config, device)

    init_flashmap_or_other.flashmap = True  # type: ignore[attr-defined]
    return init_flashmap_or_other


def _vllm_version() -> tuple[int, int] | None:
    from vllm import __version__  # noqa: PLC0415

    try:
        major, minor = __version__.split(".")[:2]
        return int(major), int(minor)
    except ValueError:
        return None


def register() -> None:
    """vLLM general plugin entry point (idempotent, never raises)."""
    try:
        version = _vllm_version()
        if version not in SUPPORTED_VLLM:
            logger.warning(
                "FlashMap's vLLM plugin supports vLLM %s, not %s; FlashMap "
                "checkpoints will not load.",
                " / ".join(f"{a}.{b}.x" for a, b in SUPPORTED_VLLM),
                version,
            )
            return
        _register()
    except Exception:  # noqa: BLE001
        logger.warning(
            "FlashMap's vLLM plugin failed to register; FlashMap checkpoints will "
            "not load.",
            exc_info=True,
        )


def _register() -> None:
    from vllm import ModelRegistry  # noqa: PLC0415
    from vllm.transformers_utils.configs.speculators import (  # noqa: PLC0415
        algos,
        base,
    )

    algos.SUPPORTED_SPECULATORS_TYPES["flashmap"] = update_flashmap

    build = base.SpeculatorsConfig.build_vllm_speculative_config
    if not getattr(build, "flashmap", False):
        # FlashMap runs on vLLM's DFlash runtime.
        def build_vllm_speculative_config(cls, config_dict):
            result = build.__func__(cls, config_dict)
            if result["method"] == "flashmap":
                result["method"] = "dflash"
            return result

        build_vllm_speculative_config.flashmap = True  # type: ignore[attr-defined]
        base.SpeculatorsConfig.build_vllm_speculative_config = classmethod(  # type: ignore[method-assign]
            build_vllm_speculative_config
        )

    if FLASHMAP_ARCHITECTURE not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            FLASHMAP_ARCHITECTURE,
            "speculators.models.flashmap.vllm.model:FlashMapDraftModel",
        )

    from vllm.v1.worker.gpu import spec_decode  # noqa: PLC0415

    if not getattr(spec_decode.init_speculator, "flashmap", False):
        spec_decode.init_speculator = _init_speculator(spec_decode.init_speculator)
    # The model runner binds init_speculator at import.
    runner = sys.modules.get("vllm.v1.worker.gpu.model_runner")
    if runner is not None and not getattr(runner.init_speculator, "flashmap", False):
        runner.init_speculator = spec_decode.init_speculator
