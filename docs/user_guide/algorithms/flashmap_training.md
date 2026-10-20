# Training FlashMap

[FlashMap](flashmap.md) is a block-parallel drafter, like [DFlash](dflash.md), whose block is drawn from a distilled discrete flow map: one network evaluation turns Gaussian noise into a whole block of `block_size` draft tokens. It is conditioned like DFlash -- the fused hidden states of the verifier's auxiliary layers before the anchor, plus the anchor token -- but drafts with a DiT denoiser over the vocabulary simplex instead of mask tokens. Taking two jumps instead of one is slower but more accurate, from the same weights.

## How It Works

### The Flow

A block of draft tokens `y` is represented as stacked one-hot vectors, and the probability path is the linear interpolant `I_t = (1 - t) z + t onehot(y)` with Gaussian `z` over the vocabulary ([Flow Map Language Models](https://arxiv.org/abs/2602.16813)). The network is the denoiser `D_t(x) = E[onehot(y) | I_t = x]`: a DiT whose AdaLN-zero layers are modulated by the time, with self-attention within the block and cross-attention onto the verifier context. It reads the state through the verifier's frozen embedding (as the posterior `softmax(t x / (1 - t)^2)`) and writes logits through the verifier's frozen LM head, so its trained parameters do not grow with the vocabulary.

### The Flow Map

The flow map `delta_{s,t}(x)` jumps the path from time `s` to time `t` in one evaluation; the drafter takes `delta_{0,1}(z)`. It is the same DiT conditioned on its end time as well, initialized from the flow (on the diagonal `s = t` it *is* the flow's denoiser) and distilled from it with the paper's semigroup objective: on half the blocks it matches the frozen flow on the diagonal, on the rest it matches two half-jumps of itself, and a fraction of those are the drafting jump `(0, 1)` itself.

### Training Stages

FlashMap trains in two stages, both through the standard `speculators train` command with `--speculator-type flashmap`:

1. **Flow** (default): the denoiser is trained with cross-entropy against the data tokens of anchor-sampled blocks, each noised to its own time from the decoding-error schedule. Use target-generated data ([response regeneration](../tutorials/response_regeneration.md)): the flow models the distribution the data was drawn from.
2. **Flow map**: `--flashmap-teacher <stage-1 checkpoint>` builds the flow map from a trained flow (its architecture is taken from the teacher) and distills it. The teacher stays frozen in memory and is not saved. The resulting checkpoints are the drafter, which is [converted for serving](#serving-the-result).

Each block's slots are weighted by `exp(-k / gamma)` for slot `k` (`--dflash-decay-gamma`, default 7; 0 turns the decay off). Validation additionally reports the one-step draft's greedy agreement with the verifier per slot (`position_k_acc`) and its expected accepted length (`eal`). The validation loss is itself random (fresh times and noise each pass), so compare checkpoints by `eal` rather than with `--save-best`, which ranks them by that loss.

## Training

```bash
# Stage 1: the flow
torchrun --standalone --nproc_per_node 8 -m speculators.train \
    --speculator-type flashmap \
    --verifier-name-or-path Qwen/Qwen3-8B \
    --target-layer-ids 2 10 18 26 34 \
    --data-path ./output/qwen3_8b_regen \
    --save-path ./output/flashmap/flow \
    --optimizer adamw --lr 6e-4 --weight-decay 0.1 --adam-beta2 0.95 \
    --scheduler-type cosine --scheduler-warmup-ratio 0.04 \
    --max-anchors 512 --epochs 20

# Stage 2: the flow map, distilled from the last flow checkpoint
torchrun --standalone --nproc_per_node 8 -m speculators.train \
    --speculator-type flashmap \
    --verifier-name-or-path Qwen/Qwen3-8B \
    --target-layer-ids 2 10 18 26 34 \
    --flashmap-teacher ./output/flashmap/flow/19 \
    --data-path ./output/qwen3_8b_regen \
    --save-path ./output/flashmap/map \
    --optimizer adamw --lr 5e-5 --weight-decay 0 --adam-beta2 0.95 \
    --scheduler-type wsd --scheduler-warmup-ratio 0.04 --scheduler-decay-ratio 0.2 \
    --max-anchors 128 --epochs 12
```

`--target-layer-ids` follows the speculators/vLLM convention used by `launch_vllm.py` (vLLM's `eagle_aux_hidden_state_layer_ids`): id `i` is the hidden state entering decoder layer `i`, i.e. `hidden_states[i]` of a Hugging Face forward. The released Qwen3-8B and Qwen3-4B drafters read `2 10 18 26 34`; the release format lists the same layers as the decoder-layer outputs `target_layer_ids = [1, 9, 17, 25, 33]` (`hidden_states[i + 1]`), and the conversion below shifts them by one. Use the same ids for data generation and both training stages; stage 2 takes them from the teacher and rejects a different `--target-layer-ids`.

The released recipe trains both stages with AdamW and `beta2 = 0.95` (the default 0.999 let the flow's loss spike on a 500k-sample run), decaying weights but not biases and norm gains, with a cosine schedule for the flow and warmup-stable-decay (`wsd`: hold the peak rate, anneal over the last 20% of the steps) for the map. The flow map holds several vocabulary-sized states per draft slot during a step (the teacher's target and two half-jumps), so it is trained with fewer anchors than the flow: `--max-anchors` defaults to 128 with `--flashmap-teacher` and 512 otherwise. A flow-map run starts from its teacher, so `--from-pretrained` is rejected with `--flashmap-teacher`; rerun with the same `--save-path` to resume it. See `examples/train/flashmap_qwen3_8b_two_stage_offline.sh` for the full pipeline.

### Batch Size

An optimizer step takes one packed row of `--total-seq-len` tokens per GPU, with `--max-anchors` blocks drawn from it, and no gradient accumulation: the global batch is `world_size` rows and `world_size * max_anchors` blocks. The released drafters were trained (in DeepSpec) on 256 sequences of up to 4096 tokens per step with 512 (flow) or 128 (map) anchors each, about 131k (flow) and 33k (map) blocks per step. With fewer blocks per step, lower the learning rate (roughly in proportion, or with its square root) and expect more optimizer steps per epoch; because a packed row holds several sequences, fewer anchors are drawn per sequence, so raise `--max-anchors` (as memory allows) or the epochs to train on a comparable number of blocks. Weight decay acts per step, so scale it with the learning rate. Each rank draws its own anchors, times and noise (the draws are seeded with seed + rank).

### Differences from DeepSpec

- Sequences are packed: blocks stay within their document (a slot past the document's end is not supervised), and anchors come from DFlash's sampler, which skips the last `block_size` positions of a packed row where DeepSpec let a sequence's last blocks run off its end.
- The loss is averaged per rank, then across ranks by the data-parallel all-reduce, where DeepSpec normalized by the global count of supervised slots.

### Serving the Result

The flow map's checkpoints are speculators checkpoints. `speculators convert` writes one in the FlashMap release format, the format of the published drafters, which the [flashmap](https://github.com/al-jshen/flashmap) package serves in vLLM and SGLang and loads with `transformers`:

```bash
speculators convert ./output/flashmap/map/11 --algorithm flashmap --verifier Qwen/Qwen3-8B \
    --output-path ./FlashMap-Qwen3-8B-mine \
    --algorithm-kwargs '{"draft_vocab_file": "./output/qwen3_8b_regen/token_freq.pt"}'
```

The folder holds `config.json`, `model.safetensors` (bf16, without the verifier's embedding and LM head) and, with a `draft_vocab_file` (the token frequencies `prepare-data --no-skip-token-freq` writes to its output, or a ranking), `draft_vocab.safetensors`, whose 65536 most frequent tokens the serving integrations draft over (`"draft_vocab_size"` picks another number; without a `draft_vocab_file` they draft over the full vocabulary). When the flashmap package is installed, the converter also copies its model code into the folder, so that `AutoModel.from_pretrained(..., trust_remote_code=True)` loads it without the package. Stage-1 flows are not drafters and are refused.

Serve the folder with the flashmap package:

```bash
uv pip install "vllm==0.31.*" "flashmap[vllm]"
vllm serve Qwen/Qwen3-8B --speculative-config '{"model": "./FlashMap-Qwen3-8B-mine", "num_speculative_tokens": 16}'
```

## Key Parameters

| Parameter               | Default | Description                                                                   |
| ----------------------- | ------- | ----------------------------------------------------------------------------- |
| `--flashmap-teacher`    | unset   | Stage 2: the stage-1 flow checkpoint to initialize and distill the map from   |
| `--num-layers`          | 6       | DiT blocks (stage 1; stage 2 copies the teacher)                              |
| `--dit-hidden-size`     | 2048    | DiT width (stage 1)                                                           |
| `--dit-num-heads`       | 16      | DiT attention heads (stage 1)                                                 |
| `--timestep-embed-dim`  | 256     | Width of the time embeddings that drive AdaLN (stage 1)                       |
| `--block-size`          | 16      | Draft tokens per block                                                        |
| `--dflash-decay-gamma`  | 7       | Per-slot loss decay (0: none)                                                 |
| `--diagonal-fraction`   | 0.5     | Stage 2: fraction of blocks matched to the frozen flow on the diagonal        |
| `--boundary-prob`       | 0.0625  | Stage 2: fraction of the off-diagonal blocks trained on the drafting jump     |
| `--antithetic-sampling` | enabled | Stage 2: stratify the time draws over the batch                               |
| `--max-anchors`         | 512/128 | Blocks per packed row (128 with `--flashmap-teacher`)                         |
| `--scheduler-type wsd`  |         | Warmup, hold, cosine anneal over the last `--scheduler-decay-ratio` (0.2)     |
| `--loss-fn`             | kl_div  | Stage 2 divergence to the map's target (stage 1 is cross-entropy to the data) |

FlashMap trains over the verifier's full vocabulary (`--draft-vocab-size` is not supported); a restricted vocabulary is a serving-time option, set at conversion (above). Flags of the DFlash group that FlashMap ignores (`--sample-from-anchor`, `--per-position-loss-weight`, `--dpace-alpha`) warn when set.

## Research & Citation

FlashMap is described in [FlashMap: Flow Map Speculative Decoding](https://arxiv.org/abs/2610.24365) ([project page](https://jshen.net/flashmap); BibTeX in [FlashMap](flashmap.md)). It distills the flow maps of [Flow Map Language Models](https://arxiv.org/abs/2602.16813) into a speculative drafter conditioned like [DFlash](https://arxiv.org/abs/2602.06036).

## See Also

- [FlashMap](flashmap.md) -- What FlashMap is, and serving it with the flashmap package
- [DFlash](dflash.md) -- The block drafter whose conditioning and anchor sampling FlashMap shares
- [Train a Speculator](../tutorials/train.md) -- Data generation and the online/offline training modes
