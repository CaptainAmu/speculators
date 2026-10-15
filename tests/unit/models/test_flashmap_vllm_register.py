"""The FlashMap vLLM plugin registers with an installed vLLM."""

import pytest

from speculators.models.flashmap.vllm import (
    FLASHMAP_ARCHITECTURE,
    SUPPORTED_VLLM,
    _vllm_version,
    register,
    update_flashmap,
)

vllm = pytest.importorskip("vllm")
algos = pytest.importorskip("vllm.transformers_utils.configs.speculators.algos")
base = pytest.importorskip("vllm.transformers_utils.configs.speculators.base")


def test_register_is_idempotent_and_hooks_vllm():
    if _vllm_version() not in SUPPORTED_VLLM:
        pytest.skip(f"vLLM {vllm.__version__} is not supported by the plugin")
    from vllm.config.speculative import SpeculativeConfig  # noqa: PLC0415

    register()
    post_init = SpeculativeConfig.__post_init__
    register()
    assert SpeculativeConfig.__post_init__ is post_init
    assert post_init.flashmap  # type: ignore[attr-defined]
    assert algos.SUPPORTED_SPECULATORS_TYPES["flashmap"] is update_flashmap
    # A release folder (model_type "flashmap") is read as a speculators config.
    from vllm.transformers_utils import config as config_utils  # noqa: PLC0415

    assert "flashmap" in config_utils._SPECULATIVE_DECODING_CONFIGS  # noqa: SLF001
    assert config_utils._CONFIG_REGISTRY["flashmap"] is base.SpeculatorsConfig  # noqa: SLF001
    assert FLASHMAP_ARCHITECTURE in vllm.ModelRegistry.get_supported_archs()
    config = {
        "speculators_model_type": "flashmap",
        "speculators_config": {"proposal_methods": [{"speculative_tokens": 16}]},
    }
    assert base.SpeculatorsConfig.build_vllm_speculative_config(config) == {
        "method": "dflash",
        "num_speculative_tokens": 16,
    }
