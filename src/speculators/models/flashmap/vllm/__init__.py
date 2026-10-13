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

Serve with ``vllm serve <speculators checkpoint>``, or pass the checkpoint as
``--speculative-config '{"model": "<checkpoint>", "method": "dflash"}'``.

``FLASHMAP_STEP_TIMING=<file>`` appends the GPU time of every draft step of any
DFlash-family speculator (FlashMap, DFlash, DSpark) to ``<file>`` as JSON lines
``{"reqs": <batch size>, "ms": <time>}``, timed with CUDA events.
"""

import json
import os
import sys
from pathlib import Path

__all__ = ["FLASHMAP_ARCHITECTURE", "register"]

FLASHMAP_ARCHITECTURE = "FlashMapDraftModel"

# Draft steps timed between flushes of FLASHMAP_STEP_TIMING.
_TIMING_FLUSH = 256

# Speculators config fields the vLLM draft config carries over.
_FLASHMAP_FIELDS = (
    "target_hidden_size",
    "draft_vocab_size",
    "block_size",
    "timestep_embed_dim",
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


def _time_draft_steps(path: str) -> None:
    import torch  # noqa: PLC0415
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (  # noqa: PLC0415
        DFlashSpeculator,
    )

    propose = DFlashSpeculator.propose
    if getattr(propose, "flashmap_timed", False):
        return
    pending: list = []

    def timed_propose(self, input_batch, *args, **kwargs):
        if kwargs.get("dummy_run") or torch.cuda.is_current_stream_capturing():
            return propose(self, input_batch, *args, **kwargs)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = propose(self, input_batch, *args, **kwargs)
        end.record()
        pending.append((input_batch.num_reqs, start, end))
        if len(pending) >= _TIMING_FLUSH:
            end.synchronize()
            with Path(path).open("a") as f:
                for reqs, s, e in pending:
                    f.write(json.dumps({"reqs": reqs, "ms": s.elapsed_time(e)}) + "\n")
            pending.clear()
        return result

    timed_propose.flashmap_timed = True  # type: ignore[attr-defined]
    DFlashSpeculator.propose = timed_propose  # type: ignore[method-assign]


def register() -> None:
    """vLLM general plugin entry point (idempotent)."""
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

    if os.environ.get("FLASHMAP_STEP_TIMING"):
        _time_draft_steps(os.environ["FLASHMAP_STEP_TIMING"])
