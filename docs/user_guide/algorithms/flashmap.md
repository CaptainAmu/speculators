# FlashMap

FlashMap is a block-parallel drafter like [DFlash](dflash.md): one draft-model forward pass proposes a whole block of `block_size` tokens, conditioned on the target model's hidden states. Its block, however, is a sample from a distilled discrete *flow map*: each draft slot starts as Gaussian noise over the vocabulary and a DiT maps it to a token distribution in one network evaluation (one "jump"), or in a few for higher acceptance. The proposal for each slot is the softmax of its logits, so verification by rejection sampling is exact at any temperature.

## How It Works

### Conditioning

The context is the same as DFlash's: the target's hidden states at `aux_hidden_state_layer_ids` are concatenated, projected by `fc` and normalized, and each committed position becomes a cross-attention key/value entry at its own position. The anchor (the last verified token, which the target has not processed yet) adds one more entry at its position, from its token embedding through `anchor_proj`. All `block_size` slots sit at the positions after the anchor and are draft tokens (the anchor is context, not a query), so a block proposes `block_size` tokens.

### Draft Model

The draft model is a DiT: each of its layers is AdaLN-zero modulated self-attention over the block's slots (non-causal, QK-normed, rotary), cross-attention onto the context, and an MLP. The modulations depend only on the jump's start and end times. Its output head projects back to the target's hidden size and the target's LM head gives the logits.

### Jumps

With one jump (the default) a slot's input is the embedding of its noise at time 0: a constant plus `x0 @ E` for the target embedding `E`. With more jumps, the state between jumps mixes the noise with the previous jump's prediction, following the flow map's update rule, on a fixed time grid (`time_grids`).

## Serving with vLLM

Speculators ships FlashMap's vLLM support as a vLLM plugin, registered under the `vllm.general_plugins` entry point when speculators is installed, so no vLLM changes are needed. It requires vLLM's V2 model runner (the default) and runs on vLLM's DFlash runtime: context K/V precompute, attention metadata and full CUDA graphs of the draft step are DFlash's own.

Convert a released FlashMap checkpoint (or a DeepSpec flow-map checkpoint) to the speculators format:

```bash
speculators convert ./FlashMap-Qwen3-8B --algorithm flashmap --verifier Qwen/Qwen3-8B \
    --output-path ./FlashMap-Qwen3-8B-speculators
```

Then serve it, with the verifier read from the checkpoint:

```bash
vllm serve ./FlashMap-Qwen3-8B-speculators
```

or explicitly:

```bash
vllm serve Qwen/Qwen3-8B \
    --speculative-config '{"model": "./FlashMap-Qwen3-8B-speculators", "method": "dflash"}'
```

For sampling at temperature > 0, add `"draft_sample_method": "probabilistic"` to the speculative config so that drafts are sampled from FlashMap's distribution and verified against it (with the default `greedy` draft sampling verification is still exact, but accepts less).

| Environment variable | Default            | Description                                          |
| -------------------- | ------------------ | ---------------------------------------------------- |
| `FLASHMAP_NUM_STEPS` | config `num_steps` | Jumps per draft block (1 is fastest)                 |
| `FLASHMAP_COMPILE`   | `1`                | torch.compile the DiT (fuses its elementwise chains) |

### Restricted Draft Vocabulary

The largest weight a draft step reads is the output head over the vocabulary. `speculators convert` can restrict it to the target's most frequent tokens (a ranking file, safetensors key `ids`), stored as the usual `d2t`/`t2d` mappings:

```bash
speculators convert ./FlashMap-Qwen3-8B --algorithm flashmap --verifier Qwen/Qwen3-8B \
    --output-path ./FlashMap-Qwen3-8B-v64k \
    --algorithm-kwargs '{"draft_vocab_size": 65536, "draft_vocab_file": "qwen3_8b_ranking.safetensors"}'
```

A token outside the draft vocabulary is never drafted; verification stays exact.

### Implementation Notes

The draft step (`speculators/models/flashmap/vllm/`) is built for latency:

- everything that depends only on the jump times (every AdaLN modulation, the constant embedding of the uniform posterior at time 0) is computed once at load;
- the one-jump noise is drawn in the embedding's space: `x0 @ E ~ N(0, E^T E)` is drawn as `z @ A` with `A = sqrt(Lambda) Q^T` from `E^T E = Q Lambda Q^T`, so no vocabulary-sized noise is formed, the noise norm's RMS is elementwise in `z`, and `A`, the noise norm and `x_embed` fold into one matrix;
- every layer's context keys/values come from one GEMM and are written to the paged KV cache, k-normed and rotated, by one Triton kernel, for committed tokens only; the anchor's entry is written at the anchor position and overwritten once the target verifies it;
- cross-attention reads the paged cache with FlashAttention (non-causal, `block_size` queries per request); self-attention within the block, with its QK norms and RoPE, is one Triton kernel;
- the DiT's output head is folded into the LM head;
- the DiT is compiled with torch.compile and the whole draft step runs in vLLM's full CUDA graph, with no host synchronization.

### Performance

Qwen3-8B on one A100-SXM4-80GB (power-capped to 225 W, so absolute numbers are low), vLLM 0.31.0, temperature 0, at most 512 output tokens, `examples/evaluate/flashmap/bench_flashmap_vllm.sh` with 8 / 16 / 64 prompts at concurrency 1 / 4 / 16. Output throughput in tokens/s, with the mean acceptance length (accepted drafts plus one) in parentheses. FlashMap uses one jump and a 64k-token draft vocabulary; DFlash is `z-lab/Qwen3-8B-DFlash-b16` (15 drafts).

| Dataset   | Concurrency | Target only | DFlash        | FlashMap      |
| --------- | ----------- | ----------- | ------------- | ------------- |
| GSM8K     | 1           | 84.5        | 298.9 (5.55)  | 393.0 (7.25)  |
| GSM8K     | 4           | 288.8       | 1021.5 (5.90) | 1297.9 (7.33) |
| GSM8K     | 16          | 1030.9      | 2048.9 (6.23) | 2482.5 (7.84) |
| HumanEval | 1           | 84.4        | 384.7 (6.09)  | 383.6 (5.35)  |
| HumanEval | 4           | 331.5       | 1247.2 (6.47) | 1107.8 (5.62) |
| HumanEval | 16          | 1050.9      | 2053.3 (6.21) | 1827.4 (5.56) |
| MT-Bench  | 1           | 84.4        | 247.8 (3.88)  | 296.0 (4.11)  |
| MT-Bench  | 4           | 324.3       | 694.8 (3.59)  | 692.7 (3.56)  |
| MT-Bench  | 16          | 1019.4      | 1090.1 (3.19) | 1107.3 (3.29) |

At temperature 1 (drafts sampled, `"draft_sample_method": "probabilistic"`):

| Dataset   | Concurrency | DFlash        | FlashMap      |
| --------- | ----------- | ------------- | ------------- |
| GSM8K     | 1           | 308.0 (4.75)  | 472.2 (6.63)  |
| GSM8K     | 16          | 1683.2 (5.12) | 1985.1 (6.42) |
| HumanEval | 1           | 298.6 (4.82)  | 345.8 (5.07)  |
| HumanEval | 16          | 1709.7 (5.27) | 1621.7 (5.03) |
| MT-Bench  | 1           | 207.8 (3.31)  | 260.4 (3.68)  |
| MT-Bench  | 16          | 944.9 (2.77)  | 1008.8 (3.02) |

A FlashMap draft step (GPU time, CUDA graphs) takes 1.59 / 2.01 / 3.78 ms at batch 1 / 4 / 16, against 2.65 / 3.13 / 7.01 ms for DFlash. Two jumps (`FLASHMAP_NUM_STEPS=2`, exact, with a vocabulary-sized state between the jumps) raise the acceptance length (GSM8K 7.68, HumanEval 6.31, MT-Bench 4.19 at concurrency 1) but cost 5.2 ms per draft step, so one jump is faster end to end on this GPU.

## Research & Citation

FlashMap distills its drafter from a discrete flow following [Flow Map Language Models](https://arxiv.org/abs/2602.16813).

## See Also

- [DFlash](dflash.md) -- The block-parallel drafter whose runtime FlashMap reuses
- [DSpark](dspark.md) -- DFlash with a Markov head, sharing FlashMap's block layout
