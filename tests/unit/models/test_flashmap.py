from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from safetensors import safe_open
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    set_model_state_dict,
)
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.convert.flashmap.converter import FlashMapConverter
from speculators.losses import resolve_loss_config
from speculators.model import SpeculatorModel
from speculators.models.flashmap import FlashMapDraftModel, FlashMapSpeculatorConfig
from speculators.models.flashmap.config import DEFAULT_TIME_GRIDS
from speculators.models.flashmap.metrics import slot_loss_weights
from speculators.models.flashmap.schedule import DecodingErrorSchedule
from speculators.models.flashmap.vllm import FLASHMAP_ARCHITECTURE, update_flashmap
from speculators.proposals.greedy import GreedyTokenProposalConfig
from speculators.train.config.schema import TrainConfig
from speculators.train.distributed import apply_fully_sharded
from speculators.train.optimizers import build_optimizers
from speculators.train.trainer import TrainerConfig, make_scheduler

HIDDEN, VOCAB, BLOCK = 16, 64, 4
# Fused losses need a GPU.
EAGER_KL = resolve_loss_config("kl_div", "eager")


def _tiny_flow(attn_impl: str = "sdpa") -> FlashMapDraftModel:
    # transformer_layer_config describes the DiT; target_hidden_size the verifier.
    dit_config = Qwen3Config(
        hidden_size=32,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        hidden_act="gelu",
        vocab_size=VOCAB,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        _attn_implementation=attn_impl,  # type: ignore[call-arg]
    )
    config = FlashMapSpeculatorConfig(
        transformer_layer_config=dit_config,
        target_hidden_size=HIDDEN,
        draft_vocab_size=VOCAB,
        block_size=BLOCK,
        aux_hidden_state_layer_ids=[0, 1],
        timestep_embed_dim=16,
        flow_map=False,
        speculators_config=SpeculatorsConfig(
            algorithm="flashmap",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=BLOCK)],
            default_proposal_method="greedy",
            verifier=VerifierConfig(name_or_path=None, architectures=[]),
        ),
    )
    model = FlashMapDraftModel(config)
    torch.nn.init.normal_(model.embed_tokens.weight)
    torch.nn.init.normal_(model.lm_head.weight)
    torch.nn.init.ones_(model.verifier_norm.weight)
    # Leave the zero-initialized output layers random so outputs depend on inputs.
    torch.nn.init.normal_(model.head.weight, std=0.1)
    for layer in model.layers:
        for adaln in (layer.self_attn_adaln, layer.cross_attn_adaln, layer.mlp_adaln):
            torch.nn.init.normal_(adaln.linear.weight, std=0.1)
    return model


def _batch(seq_len: int = 40, num_docs: int = 2):
    torch.manual_seed(0)
    lengths = [seq_len // num_docs] * num_docs
    document_ids = torch.repeat_interleave(
        torch.arange(num_docs), torch.tensor(lengths)
    ).unsqueeze(0)
    position_ids = torch.cat([torch.arange(n) for n in lengths]).unsqueeze(0)
    loss_mask = torch.ones(1, seq_len)
    loss_mask[:, :3] = 0
    return {
        "hidden_states": torch.randn(1, seq_len, 2 * HIDDEN),
        "input_ids": torch.randint(0, VOCAB, (1, seq_len)),
        "loss_mask": loss_mask,
        "document_ids": document_ids,
        "position_ids": position_ids,
        "verifier_last_hidden_states": torch.randn(1, seq_len, HIDDEN),
    }


def _vllm_plugin_weight_names(num_layers: int) -> set[str]:
    """The weights the vLLM plugin's FlashMapDraftModel loads (every parameter
    of speculators/models/flashmap/vllm/model.py but the target's own and the
    attention layers' cache scales), for a full-vocabulary draft."""
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


def _toy_verifier_config() -> Qwen3Config:
    return Qwen3Config(
        hidden_size=HIDDEN,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=VOCAB,
        max_position_embeddings=128,
    )


def _training_args(**overrides) -> dict:
    """CLI-shaped kwargs for from_training_args with a toy verifier."""
    flat = TrainConfig(speculator_type="flashmap").flatten()
    flat.update(
        verifier_name_or_path="toy-verifier",
        target_layer_ids=[0, 1],
        draft_vocab_size=VOCAB,
        block_size=BLOCK,
        dit_hidden_size=32,
        dit_num_heads=4,
        timestep_embed_dim=16,
        draft_attn_impl="sdpa",
    )
    flat.update(overrides)
    return flat


@pytest.fixture
def offline_verifier(monkeypatch):
    """from_training_args without the hub: toy verifier ids, config and weights."""
    monkeypatch.setattr(
        "speculators.models.dflash.core.resolve_target_layer_ids",
        lambda ids, *args, **kwargs: list(ids),
    )
    monkeypatch.setattr(
        VerifierConfig,
        "from_pretrained",
        classmethod(lambda cls, name, **kw: cls(name_or_path=name, architectures=[])),
    )
    monkeypatch.setattr(FlashMapDraftModel, "load_verifier_weights", lambda self: None)


def test_flashmap_is_registered():
    assert SpeculatorModel.registry["flashmap"] is FlashMapDraftModel  # type: ignore[index]


def test_flashmap_cli_defaults():
    cfg = TrainConfig(speculator_type="flashmap")
    assert cfg.dflash.block_size == 16
    assert cfg.draft.num_layers == 6
    assert cfg.dflash.dflash_decay_gamma == 7.0
    assert cfg.flashmap.flashmap_teacher is None
    # Other algorithms keep their defaults.
    assert TrainConfig(speculator_type="dflash").dflash.dflash_decay_gamma == 4.0


def test_flow_map_cli_defaults_and_checks():
    cfg = TrainConfig(speculator_type="flashmap", flashmap={"flashmap_teacher": "f"})
    assert cfg.data.max_anchors == 128
    assert TrainConfig(speculator_type="flashmap").data.max_anchors == 512
    assert TrainConfig(speculator_type="dflash").data.max_anchors == 512
    flat = cfg.flatten()
    assert FlashMapDraftModel.get_trainer_kwargs(**flat)[0]["max_anchors"] == 128
    flat["dflash_decay_gamma"] = 0.0
    assert FlashMapDraftModel.get_trainer_kwargs(**flat)[0]["gamma"] == 0.0
    with pytest.raises(ValueError, match="mutually exclusive"):
        TrainConfig(
            speculator_type="flashmap",
            flashmap={"flashmap_teacher": "f"},
            draft={"from_pretrained": "d"},
        )


def test_flashmap_warns_on_dflash_flags_it_ignores():
    with pytest.warns(UserWarning, match="flashmap ignores: --dpace-alpha"):
        TrainConfig.from_sources(
            cli={
                "speculator_type": "flashmap",
                "verifier_name_or_path": "v",
                "dpace_alpha": 0.7,
            },
            argv=[],
        )


def test_slot_decay_off_at_gamma_zero():
    mask = torch.ones(1, 2 * BLOCK, dtype=torch.bool)
    weights, _ = slot_loss_weights(mask, BLOCK, 0.0)
    assert torch.equal(weights, mask.float())
    weights, _ = slot_loss_weights(mask, BLOCK, 7.0)
    assert weights[0, 1] == pytest.approx(torch.exp(torch.tensor(-1 / 7.0)).item())


def test_flashmap_trainer_kwargs_follow_the_cli():
    flat = TrainConfig(speculator_type="flashmap").flatten()
    train_kwargs, val_kwargs = FlashMapDraftModel.get_trainer_kwargs(**flat)
    assert train_kwargs["gamma"] == 7.0
    assert "kl_div" in train_kwargs["loss_config"]
    assert train_kwargs["max_anchors"] == 512
    assert (train_kwargs["diagonal_fraction"], train_kwargs["boundary_prob"]) == (
        0.5,
        0.0625,
    )
    assert val_kwargs.keys() == train_kwargs.keys()


def test_decoding_error_grids_match_the_release():
    schedule = DecodingErrorSchedule(151936)
    for num_steps, expected in enumerate(DEFAULT_TIME_GRIDS, start=1):
        assert schedule.grid(num_steps, torch.device("cpu")).tolist() == expected


@pytest.mark.parametrize("attn_impl", ["sdpa", "eager"])
def test_flow_forward_backward(attn_impl):
    model = _tiny_flow(attn_impl).train()
    _, loss, metrics = model(**_batch(), max_anchors=8, gamma=7.0)
    assert torch.isfinite(loss)
    assert {"loss_sum", "loss_tau0_sum", "denoise_acc_sum"} <= set(metrics)
    loss.backward()
    assert model.layers[0].cross_attn.q_proj.weight.grad is not None
    assert model.embed_tokens.weight.grad is None


def test_context_mask_sees_document_prefix_and_own_anchor():
    model = _tiny_flow("sdpa")
    batch = _batch()
    seq_len = batch["input_ids"].shape[1]
    anchors = torch.tensor([5, 25])
    mask = model._context_mask(batch["document_ids"], anchors)[0, 0]
    assert mask.shape == (2 * BLOCK, seq_len + 2)
    # Block 0 (anchor 5, document 0): context 0..4 and anchor entry 0.
    expected0 = torch.zeros(seq_len + 2, dtype=torch.bool)
    expected0[:5] = True
    expected0[seq_len] = True
    # Block 1 (anchor 25, document 1 = positions 20..39): 20..24 and entry 1.
    expected1 = torch.zeros(seq_len + 2, dtype=torch.bool)
    expected1[20:25] = True
    expected1[seq_len + 1] = True
    assert torch.equal(mask[:BLOCK], expected0.expand(BLOCK, -1))
    assert torch.equal(mask[BLOCK:], expected1.expand(BLOCK, -1))


def test_blocks_stay_in_their_document():
    model = _tiny_flow()
    batch = _batch()
    blocks = model._sample_blocks(
        batch["input_ids"],
        batch["loss_mask"],
        batch["document_ids"],
        batch["position_ids"],
        num_anchors=64,
    )
    docs = batch["document_ids"][0]
    label_docs = docs[blocks.label_indices].view(-1, BLOCK)
    anchor_docs = docs[blocks.anchor_positions].unsqueeze(-1)
    supervised = blocks.label_mask.view(-1, BLOCK)
    assert torch.all(
        label_docs[supervised] == anchor_docs.expand_as(label_docs)[supervised]
    )
    # A supervised prefix per block: never a gap followed by a supervised slot.
    assert torch.all(supervised[:, 1:] <= supervised[:, :-1])
    # Slot positions continue from the anchor's position within its document.
    expected = batch["position_ids"][0, blocks.anchor_positions].unsqueeze(-1) + 1
    expected = expected + torch.arange(BLOCK)
    assert torch.equal(blocks.position_ids.view(-1, BLOCK), expected)


def test_map_starts_as_its_flow_on_the_diagonal():
    torch.manual_seed(0)
    flow = _tiny_flow().eval()
    flow_map = FlashMapDraftModel.from_teacher(flow).eval()
    assert flow_map.config.flow_map
    assert not any(
        name.startswith("_teacher") for name, _ in flow_map.named_parameters()
    )

    batch = _batch()
    blocks = flow_map._sample_blocks(
        batch["input_ids"],
        batch["loss_mask"],
        batch["document_ids"],
        batch["position_ids"],
        num_anchors=4,
    )
    mask = flow_map._context_mask(batch["document_ids"], blocks.anchor_positions)
    args = (
        batch["hidden_states"],
        batch["input_ids"],
        batch["position_ids"],
        blocks.anchor_positions,
    )
    state = torch.randn(1, blocks.labels.shape[1], VOCAB)
    times = torch.rand(1, blocks.labels.shape[1])
    with torch.no_grad():
        expected = flow.denoise(
            state, times, blocks.position_ids, flow.project_context(*args), mask
        )
        actual = flow_map.denoise(
            state,
            times,
            blocks.position_ids,
            flow_map.project_context(*args),
            mask,
            end=times,
        )
    torch.testing.assert_close(actual, expected)


def test_semigroup_loss_vanishes_on_the_diagonal_at_init():
    flow_map = FlashMapDraftModel.from_teacher(_tiny_flow()).train()
    _, loss, _ = flow_map(
        **_batch(), max_anchors=8, diagonal_fraction=1.0, loss_config=EAGER_KL
    )
    assert loss.abs().item() < 1e-5

    _, loss, metrics = flow_map(
        **_batch(), max_anchors=8, boundary_prob=0.25, loss_config=EAGER_KL
    )
    assert torch.isfinite(loss)
    assert {"loss_diag_sum", "loss_off_sum", "loss_boundary_sum"} <= set(metrics)
    loss.backward()
    assert flow_map.jump_embed.linear_2.weight.grad is not None


def test_eval_reports_one_step_draft_metrics():
    flow_map = FlashMapDraftModel.from_teacher(_tiny_flow()).eval()
    with torch.no_grad():
        _, _, metrics = flow_map(**_batch(), max_anchors=8, loss_config=EAGER_KL)
    assert metrics["eal_total"] > 0
    assert 1.0 <= metrics["eal_sum"] / metrics["eal_total"] <= BLOCK + 1


def test_checkpoint_layout(tmp_path):
    flow_map = FlashMapDraftModel.from_teacher(_tiny_flow())
    flow_map.save_pretrained(tmp_path)
    with safe_open(tmp_path / "model.safetensors", framework="pt") as f:
        keys = set(f.keys())
    # Exactly what the vLLM plugin loads: no verifier weights, nothing missing.
    assert keys == _vllm_plugin_weight_names(num_layers=2)

    loaded = FlashMapDraftModel.from_pretrained(tmp_path)
    assert loaded.config.flow_map
    times = loaded.time_schedule.grid(2, torch.device("cpu"))
    assert torch.isfinite(times).all()
    assert times.device.type == "cpu"
    for name, tensor in flow_map.state_dict().items():
        if name.split(".")[0] in {"embed_tokens", "lm_head", "verifier_lm_head"}:
            continue
        torch.testing.assert_close(loaded.state_dict()[name], tensor)


def test_semigroup_times_split_and_order():
    flow_map = FlashMapDraftModel.from_teacher(_tiny_flow())
    kind, s, u, t = flow_map._sample_semigroup_times(
        64,
        torch.device("cpu"),
        diagonal_fraction=0.5,
        boundary_prob=0.25,
        antithetic_sampling=True,
    )
    diagonal, boundary = kind == 0, kind == 1
    assert diagonal.sum() == 32
    assert boundary.sum() == 8  # 64 * 0.5 * 0.25, a whole number
    assert torch.all(s <= u)
    assert torch.all(u <= t)
    assert torch.equal(s[diagonal], t[diagonal])
    assert torch.all(s[boundary] == 0.0)
    assert torch.all(t[boundary] == 1.0)


def test_boundary_count_is_rounded_stochastically():
    flow_map = FlashMapDraftModel.from_teacher(_tiny_flow())
    torch.manual_seed(0)
    counts = [
        int(
            (
                flow_map._sample_semigroup_times(
                    8,
                    torch.device("cpu"),
                    diagonal_fraction=0.5,
                    boundary_prob=0.0625,
                    antithetic_sampling=True,
                )[0]
                == 1
            ).sum()
        )
        for _ in range(400)
    ]
    # 8 * 0.5 * 0.0625 = 0.25 boundary blocks per batch on average, not 0.
    assert set(counts) <= {0, 1}
    assert 0.15 < sum(counts) / len(counts) < 0.35


def test_teacher_shares_the_frozen_vocabulary_matrices():
    flow = _tiny_flow()
    flow_map = FlashMapDraftModel.from_teacher(flow)
    assert flow.embed_tokens is flow_map.embed_tokens
    assert flow.lm_head is flow_map.lm_head
    # The end-time embedding: PyTorch's default init in, zero out (as DeepSpec).
    assert flow_map.jump_embed.linear_1.weight.std() > 0.03
    assert torch.all(flow_map.jump_embed.linear_2.weight == 0)


def test_weight_decay_skips_vectors_for_flashmap_only():
    config = SimpleNamespace(
        optimizer="adamw", lr=1e-3, adam_beta2=0.95, weight_decay=0.1
    )
    (optimizer,) = build_optimizers(_tiny_flow(), config)
    decayed, plain = optimizer.param_groups
    assert decayed["weight_decay"] == 0.1
    assert all(p.ndim >= 2 for p in decayed["params"])
    assert plain["weight_decay"] == 0.0
    assert all(p.ndim < 2 for p in plain["params"])
    assert "hidden_norm.weight" in plain["param_names"]
    # Other models keep a single decayed group.
    (optimizer,) = build_optimizers(torch.nn.Linear(4, 4), config)
    assert len(optimizer.param_groups) == 1


def test_wsd_scheduler_holds_then_anneals():
    config = TrainerConfig(
        lr=1.0,
        num_epochs=1,
        save_path="unused",
        scheduler_type="wsd",
        scheduler_decay_ratio=0.2,
    )
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    scheduler = make_scheduler(optimizer, config, warmup_steps=4, total_steps=100)
    lrs = []
    for _ in range(100):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    assert lrs[0] == 0.0
    assert lrs[4] == lrs[79] == 1.0
    assert lrs[90] == pytest.approx(0.5, abs=0.05)  # halfway through the anneal
    assert lrs[99] < 0.01


def test_training_config_feeds_the_vllm_plugin(offline_verifier):
    flow = FlashMapDraftModel.from_training_args(
        _toy_verifier_config(), **_training_args()
    )
    assert not flow.config.flow_map
    with pytest.raises(ValueError, match="stage-1 flow"):
        update_flashmap(flow.config.to_dict(), {})

    flow_map = FlashMapDraftModel.from_training_args(
        _toy_verifier_config(), **_training_args(flashmap_teacher=flow)
    )
    config_dict = flow_map.config.to_dict()
    pre_trained_config = dict(config_dict["transformer_layer_config"])
    update_flashmap(config_dict, pre_trained_config)
    assert pre_trained_config["architectures"] == [FLASHMAP_ARCHITECTURE]
    assert pre_trained_config["eagle_aux_hidden_state_layer_ids"] == [0, 1]
    assert pre_trained_config["dflash_config"]["target_layer_ids"] == [-1, 0]
    assert pre_trained_config["target_hidden_size"] == HIDDEN
    assert pre_trained_config["timestep_embed_dim"] == 16
    assert pre_trained_config["time_grids"] == flow_map.config.time_grids
    assert pre_trained_config["hidden_size"] == 32  # the DiT's


def test_flow_map_rejects_other_target_layers(offline_verifier):
    with pytest.raises(ValueError, match="differ from the teacher"):
        FlashMapDraftModel.from_training_args(
            _toy_verifier_config(),
            **_training_args(flashmap_teacher=_tiny_flow(), target_layer_ids=[1, 0]),
        )


def test_converter_keeps_a_trained_config(tmp_path):
    flow_map = FlashMapDraftModel.from_teacher(_tiny_flow())
    flow_map.config.adaln_eps = 1e-5  # not the default: must survive conversion
    flow_map.save_pretrained(tmp_path / "trained")
    _toy_verifier_config().save_pretrained(tmp_path / "verifier")
    token_freq = {token: VOCAB - token for token in range(VOCAB)}
    torch.save(token_freq, tmp_path / "token_freq.pt")

    FlashMapConverter().convert(
        tmp_path / "trained",
        tmp_path / "served",
        base_model=str(tmp_path / "verifier"),
        draft_vocab_size=16,
        draft_vocab_file=str(tmp_path / "token_freq.pt"),
    )
    served = FlashMapSpeculatorConfig.from_pretrained(tmp_path / "served")
    assert served.adaln_eps == 1e-5
    assert served.draft_vocab_size == 16
    assert served.aux_hidden_state_layer_ids == [0, 1]
    with safe_open(tmp_path / "served" / "model.safetensors", framework="pt") as f:
        assert torch.equal(f.get_tensor("t2d").nonzero().squeeze(-1), torch.arange(16))
        assert set(f.keys()) == _vllm_plugin_weight_names(num_layers=2) | {"d2t", "t2d"}


@pytest.mark.parametrize("stage", ["flow", "map"])
def test_fully_sharded_on_one_rank(tmp_path, stage):
    """--fsdp-shard: the context projection runs as an FSDP forward method."""
    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'pg'}", rank=0, world_size=1
    )
    try:
        model = _tiny_flow()
        if stage == "map":
            model = FlashMapDraftModel.from_teacher(model)
        # As Trainer._setup_model_fsdp: shard, then broadcast rank 0's weights.
        full_state_dict = model.state_dict()
        apply_fully_sharded(model.train(), param_dtype=torch.float32)
        set_model_state_dict(
            model,
            full_state_dict,
            options=StateDictOptions(
                full_state_dict=True, broadcast_from_rank0=True, strict=False
            ),
        )
        _, loss, _ = model(**_batch(), max_anchors=8, loss_config=EAGER_KL)
        loss.backward()
        grad = model.layers[0].cross_attn.k_proj.weight.grad
        assert grad is not None
        assert grad.full_tensor().abs().sum() > 0
    finally:
        dist.destroy_process_group()
