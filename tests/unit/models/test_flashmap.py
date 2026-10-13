import pytest
import torch
from safetensors import safe_open
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.losses import resolve_loss_config
from speculators.model import SpeculatorModel
from speculators.models.flashmap import FlashMapDraftModel, FlashMapSpeculatorConfig
from speculators.models.flashmap.config import DEFAULT_TIME_GRIDS
from speculators.models.flashmap.schedule import DecodingErrorSchedule
from speculators.proposals.greedy import GreedyTokenProposalConfig
from speculators.train.config.schema import TrainConfig

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
    assert "embed_tokens.weight" not in keys
    assert "lm_head.weight" not in keys
    assert {
        "fc.weight",
        "hidden_norm.weight",
        "anchor_proj.bias",
        "state_norm.weight",
        "noise_norm.weight",
        "noise_gate",
        "x_embed.weight",
        "time_embed.linear_1.weight",
        "jump_embed.linear_2.bias",
        "layers.0.self_attn_adaln.linear.weight",
        "layers.0.cross_attn.k_proj.weight",
        "layers.0.cross_attn.v_proj.weight",
        "layers.1.self_attn.q_norm.weight",
        "layers.1.mlp.fc1.weight",
        "final_adaln.linear.weight",
        "head.bias",
    } <= keys

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
    assert boundary.sum() == 8  # int(64 * (0.5 + 0.5 * 0.25)) - 32
    assert torch.all(s <= u)
    assert torch.all(u <= t)
    assert torch.equal(s[diagonal], t[diagonal])
    assert torch.all(s[boundary] == 0.0)
    assert torch.all(t[boundary] == 1.0)
