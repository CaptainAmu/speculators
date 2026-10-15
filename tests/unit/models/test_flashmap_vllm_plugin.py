"""Unit tests for the FlashMap vLLM plugin's speculators-config mapping."""

from speculators.models.flashmap.config import DEFAULT_TIME_GRIDS
from speculators.models.flashmap.vllm import FLASHMAP_ARCHITECTURE, update_flashmap


def test_update_flashmap_maps_onto_dflash_runtime():
    config_dict = {
        "speculators_model_type": "flashmap",
        "aux_hidden_state_layer_ids": [2, 10, 18, 26, 34],
        "target_hidden_size": 4096,
        "draft_vocab_size": 65536,
        "block_size": 16,
        "timestep_embed_dim": 256,
        "time_scale": 1000.0,
        "qk_norm_eps": 1e-5,
        "adaln_eps": 1e-6,
        "noise_gate": "jump",
        "time_schedule": "decoding_error",
        "num_steps": 1,
        "time_grids": DEFAULT_TIME_GRIDS,
        "mask_token_id": None,
    }
    pre_trained_config = {"hidden_size": 2048, "num_hidden_layers": 6}
    update_flashmap(config_dict, pre_trained_config)

    assert pre_trained_config["architectures"] == [FLASHMAP_ARCHITECTURE]
    assert pre_trained_config["eagle_aux_hidden_state_layer_ids"] == [
        2,
        10,
        18,
        26,
        34,
    ]
    # vLLM's DFlash runtime: non-causal draft attention, target layers as i - 1,
    # and a valid (unused) id for the query rows after the anchor.
    assert pre_trained_config["dflash_config"] == {
        "mask_token_id": 0,
        "target_layer_ids": [1, 9, 17, 25, 33],
        "causal": False,
    }
    for key in ("target_hidden_size", "draft_vocab_size", "time_grids", "noise_gate"):
        assert pre_trained_config[key] == config_dict[key]
    # The DiT dimensions come from transformer_layer_config untouched.
    assert pre_trained_config["hidden_size"] == 2048


def test_update_flashmap_reads_a_release_config():
    """A FlashMap release folder's config.json (model_type "flashmap" with the
    speculators header flashmap's convert_checkpoint.py writes) maps the same way."""
    grids = {str(n): grid for n, grid in enumerate(DEFAULT_TIME_GRIDS, start=1)}
    config_dict = {
        "model_type": "flashmap",
        "speculators_model_type": "flashmap",
        "transformer_layer_config": {},
        "hidden_size": 2048,
        "num_hidden_layers": 6,
        "num_attention_heads": 16,
        "intermediate_size": 8192,
        "vocab_size": 151936,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "target_layer_ids": [1, 9, 17, 25, 33],
        "target_hidden_size": 4096,
        "draft_vocab_size": 65536,
        "block_size": 16,
        "cond_dim": 256,
        "time_embed_dim": 256,
        "time_scale": 1000.0,
        "layer_norm_eps": 1e-6,
        "num_steps": 1,
        "jump_grids": grids,
    }
    pre_trained_config = dict(config_dict["transformer_layer_config"])
    update_flashmap(config_dict, pre_trained_config)
    assert pre_trained_config["eagle_aux_hidden_state_layer_ids"] == [2, 10, 18, 26, 34]
    assert pre_trained_config["dflash_config"]["target_layer_ids"] == [1, 9, 17, 25, 33]
    assert pre_trained_config["head_dim"] == 128
    assert pre_trained_config["num_key_value_heads"] == 16
    assert pre_trained_config["rope_parameters"]["rope_theta"] == 10000.0
    assert pre_trained_config["time_grids"] == DEFAULT_TIME_GRIDS
    assert pre_trained_config["timestep_embed_dim"] == 256
    assert pre_trained_config["draft_vocab_size"] == 65536
