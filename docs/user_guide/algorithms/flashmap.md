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

Speculators ships FlashMap's vLLM support as a vLLM plugin, registered under the `vllm.general_plugins` entry point when speculators is installed, so no vLLM changes are needed. It runs on vLLM's DFlash runtime: model loading, the context K/V precompute, attention metadata and the full CUDA graph of the draft step are DFlash's own.

Requirements:

- vLLM 0.31.x, with speculators installed in the same environment (on other versions the plugin logs a warning and stays out of the way);
- vLLM's V2 model runner (the default; not `VLLM_USE_V2_MODEL_RUNNER=0`);
- tensor parallel size 1;
- the FlashAttention attention backend and a bf16 KV cache (no fp8 draft KV cache).

Each of these is checked when the draft model loads, with an error that names it.

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
vllm serve Qwen/Qwen3-8B --speculative-config \
    '{"model": "./FlashMap-Qwen3-8B-speculators", "method": "dflash", "num_speculative_tokens": 16}'
```

For sampling at temperature > 0, add `"draft_sample_method": "probabilistic"` to the speculative config so that drafts are sampled from FlashMap's distribution and verified against it (with the default `greedy` draft sampling verification is still exact, but accepts less).

`num_speculative_tokens` may be smaller than the block (16): the draft still evaluates its whole block and proposes its first `num_speculative_tokens` slots. Verification then checks fewer tokens, which pays off at high concurrency, where verification dominates.

| Environment variable | Default            | Description                                                                                                   |
| -------------------- | ------------------ | ------------------------------------------------------------------------------------------------------------- |
| `FLASHMAP_NUM_STEPS` | config `num_steps` | Jumps per draft block (1 is fastest); also set by `speculators convert --algorithm-kwargs '{"num_steps": 2}'` |

### Draft Vocabulary

The largest weight a draft step reads is the output head over the vocabulary. A released checkpoint ships a ranking of the target's tokens (`draft_vocab.safetensors`), and `speculators convert` restricts the draft head to its 65536 most frequent tokens by default, stored as the usual `d2t`/`t2d` mappings; `--algorithm-kwargs '{"draft_vocab_size": 0}'` keeps the full vocabulary, and `{"draft_vocab_size": N, "draft_vocab_file": "ranking.safetensors"}` picks another size or ranking. A token outside the draft vocabulary is never drafted; verification stays exact.

### Implementation Notes

The draft step (`speculators/models/flashmap/vllm/`) is built for latency:

- everything that depends only on the jump times (every AdaLN modulation, the constant embedding of the uniform posterior at time 0) is computed once at load;
- the one-jump noise is drawn in the embedding's space: `x0 @ E ~ N(0, E^T E)` is drawn as `z @ A` with `A = sqrt(Lambda) Q^T` from `E^T E = Q Lambda Q^T`, so no vocabulary-sized noise is formed, the noise norm's RMS is elementwise in `z`, and `A`, the noise norm and `x_embed` fold into one matrix (`fold.py`);
- the keys/values of the context committed since the last step and of every request's anchor come from one GEMM over all layers' weights and are written to the paged KV cache, k-normed and rotated, by one Triton kernel; the anchor's entry is written at the anchor position and overwritten once the target verifies it;
- cross-attention reads the paged cache with FlashAttention (non-causal, `block_size` queries per request, one FlashAttention 3 tile schedule shared by all layers); self-attention within the block, with its QK norms and RoPE, is one Triton kernel;
- the DiT's output head is folded into the LM head; greedy drafts take a split argmax over it, and sampled drafts with a restricted vocabulary write their proposal distribution straight into vLLM's cached draft logits;
- the noise is keyed by each request's seed and each slot's position, so a seeded request drafts the same blocks on every run;
- the DiT is compiled with torch.compile and the whole draft step is replayed from vLLM's full CUDA graph; the step's inputs are prepared on the GPU, without waiting on the host.

### Upstreaming to vLLM

The plugin's model, speculator and kernels are written to move into vLLM unchanged (as `model_executor/models/flashmap.py` and `v1/worker/gpu/spec_decode/flashmap/`). The plugin's hooks then become one-line registrations, as for DFlash2:

- `transformers_utils/configs/speculators/algos.py`: `@register_speculator("flashmap")` on `update_flashmap`;
- `transformers_utils/configs/speculators/base.py` (`build_vllm_speculative_config`): map `"flashmap"` to `"dflash"`, as `"dflash2"` is;
- `model_executor/models/registry.py`: `"FlashMapDraftModel": ("flashmap", "FlashMapDraftModel")` among the speculative decoding models;
- `v1/worker/gpu/spec_decode/__init__.py` (`init_speculator`): return `FlashMapSpeculator` when `"FlashMapDraftModel"` is in the draft's architectures, next to the `DFlash2DraftModel` case.

### Performance

One H100 PCIe 80GB (350 W), vLLM 0.31.0, bf16, measured with `examples/evaluate/flashmap/bench_flashmap_vllm.sh`: each server gets a warm pass and each point two warmup requests; closed loop at concurrency 1 / 4 / 16 with 16 / 16 / 64 chat-templated prompts, at most 1024 output tokens, prefix caching off, no top-k/top-p. FlashMap is the default conversion (65536-token draft vocabulary, 16 drafts, one jump unless noted); DFlash is `z-lab/Qwen3-*-DFlash-b16` (15 drafts). Each cell gives output tokens/s (speedup over the target alone), mean time per output token and the mean acceptance length τ (accepted drafts plus one).

Qwen3-8B, temperature 0:

| Dataset   | Conc. | Target               | DFlash                        | FlashMap                      | FlashMap 2 jumps              |
| --------- | ----- | -------------------- | ----------------------------- | ----------------------------- | ----------------------------- |
| GSM8K     | 1     | 104 tok/s, 9.56 ms   | 468 (4.50x), 2.03 ms, τ 5.86  | 621 (5.97x), 1.49 ms, τ 7.18  | 549 (5.28x), 1.71 ms, τ 7.77  |
| GSM8K     | 4     | 351 tok/s, 9.68 ms   | 1467 (4.18x), 2.13 ms, τ 5.85 | 2023 (5.76x), 1.52 ms, τ 7.54 | 1627 (4.63x), 1.85 ms, τ 7.79 |
| GSM8K     | 16    | 1280 tok/s, 10.23 ms | 4863 (3.80x), 2.61 ms, τ 6.28 | 5910 (4.62x), 1.99 ms, τ 7.76 | 4503 (3.52x), 2.67 ms, τ 8.43 |
| HumanEval | 1     | 104 tok/s, 9.59 ms   | 511 (4.92x), 1.91 ms, τ 6.48  | 500 (4.82x), 1.97 ms, τ 5.85  | 426 (4.10x), 2.32 ms, τ 6.21  |
| HumanEval | 4     | 387 tok/s, 9.76 ms   | 1808 (4.67x), 1.98 ms, τ 6.65 | 1747 (4.52x), 2.03 ms, τ 6.08 | 1400 (3.62x), 2.57 ms, τ 6.14 |
| HumanEval | 16    | 1318 tok/s, 10.45 ms | 5221 (3.96x), 2.73 ms, τ 6.37 | 4859 (3.69x), 2.93 ms, τ 5.69 | 3523 (2.67x), 4.07 ms, τ 6.01 |
| MT-Bench  | 1     | 104 tok/s, 9.56 ms   | 288 (2.77x), 3.52 ms, τ 3.56  | 316 (3.04x), 3.18 ms, τ 3.59  | 260 (2.50x), 3.89 ms, τ 3.67  |
| MT-Bench  | 4     | 385 tok/s, 9.74 ms   | 1007 (2.61x), 3.64 ms, τ 3.53 | 989 (2.57x), 3.38 ms, τ 3.50  | 791 (2.05x), 4.26 ms, τ 3.66  |
| MT-Bench  | 16    | 1128 tok/s, 10.30 ms | 2306 (2.04x), 5.06 ms, τ 3.15 | 2358 (2.09x), 4.71 ms, τ 3.21 | 1675 (1.49x), 6.87 ms, τ 3.27 |

Qwen3-8B, temperature 1 (drafts sampled, `"draft_sample_method": "probabilistic"`):

| Dataset   | Conc. | Target               | DFlash                        | FlashMap                      |
| --------- | ----- | -------------------- | ----------------------------- | ----------------------------- |
| GSM8K     | 1     | 104 tok/s, 9.56 ms   | 400 (3.85x), 2.32 ms, τ 4.84  | 593 (5.70x), 1.57 ms, τ 6.85  |
| GSM8K     | 4     | 348 tok/s, 9.69 ms   | 1315 (3.78x), 2.49 ms, τ 5.03 | 1716 (4.93x), 1.75 ms, τ 6.70 |
| GSM8K     | 16    | 1243 tok/s, 10.23 ms | 3577 (2.88x), 3.07 ms, τ 5.25 | 5089 (4.10x), 2.32 ms, τ 6.73 |
| HumanEval | 1     | 104 tok/s, 9.60 ms   | 417 (4.01x), 2.35 ms, τ 5.27  | 439 (4.23x), 2.21 ms, τ 5.10  |
| HumanEval | 4     | 377 tok/s, 9.77 ms   | 1395 (3.70x), 2.57 ms, τ 5.01 | 1551 (4.11x), 2.40 ms, τ 4.86 |
| HumanEval | 16    | 1353 tok/s, 10.45 ms | 4208 (3.11x), 3.28 ms, τ 5.29 | 4017 (2.97x), 3.43 ms, τ 4.87 |
| MT-Bench  | 1     | 104 tok/s, 9.57 ms   | 245 (2.36x), 4.00 ms, τ 3.02  | 282 (2.71x), 3.44 ms, τ 3.19  |
| MT-Bench  | 4     | 390 tok/s, 9.75 ms   | 865 (2.22x), 4.03 ms, τ 3.08  | 958 (2.46x), 3.75 ms, τ 3.18  |
| MT-Bench  | 16    | 1174 tok/s, 10.32 ms | 1978 (1.69x), 5.67 ms, τ 2.73 | 2194 (1.87x), 5.14 ms, τ 2.91 |

Qwen3-4B, temperature 0:

| Dataset   | Conc. | Target              | DFlash                        | FlashMap                      | FlashMap 2 jumps              |
| --------- | ----- | ------------------- | ----------------------------- | ----------------------------- | ----------------------------- |
| GSM8K     | 1     | 168 tok/s, 5.89 ms  | 745 (4.42x), 1.24 ms, τ 5.95  | 963 (5.71x), 0.96 ms, τ 7.46  | 790 (4.69x), 1.17 ms, τ 8.03  |
| GSM8K     | 4     | 553 tok/s, 6.01 ms  | 2453 (4.44x), 1.31 ms, τ 5.97 | 3035 (5.49x), 1.01 ms, τ 7.49 | 2391 (4.33x), 1.32 ms, τ 8.06 |
| GSM8K     | 16    | 1937 tok/s, 6.64 ms | 6951 (3.59x), 1.67 ms, τ 6.15 | 7768 (4.01x), 1.35 ms, τ 7.61 | 5456 (2.82x), 2.05 ms, τ 8.06 |
| HumanEval | 1     | 168 tok/s, 5.91 ms  | 818 (4.87x), 1.19 ms, τ 6.56  | 759 (4.52x), 1.27 ms, τ 5.84  | 621 (3.70x), 1.55 ms, τ 6.19  |
| HumanEval | 4     | 568 tok/s, 6.08 ms  | 2776 (4.89x), 1.27 ms, τ 6.46 | 2445 (4.31x), 1.42 ms, τ 5.78 | 1894 (3.33x), 1.90 ms, τ 5.99 |
| HumanEval | 16    | 1857 tok/s, 6.86 ms | 7341 (3.95x), 1.71 ms, τ 6.43 | 6382 (3.44x), 1.95 ms, τ 5.69 | 4491 (2.42x), 2.91 ms, τ 6.13 |
| MT-Bench  | 1     | 169 tok/s, 5.88 ms  | 442 (2.62x), 2.21 ms, τ 3.46  | 461 (2.73x), 2.09 ms, τ 3.49  | 369 (2.19x), 2.58 ms, τ 3.64  |
| MT-Bench  | 4     | 626 tok/s, 6.05 ms  | 1593 (2.54x), 2.28 ms, τ 3.48 | 1656 (2.64x), 2.21 ms, τ 3.62 | 1219 (1.95x), 2.95 ms, τ 3.71 |
| MT-Bench  | 16    | 1836 tok/s, 6.66 ms | 3505 (1.91x), 3.22 ms, τ 3.08 | 3640 (1.98x), 3.11 ms, τ 3.29 | 2365 (1.29x), 4.96 ms, τ 3.31 |

Verifying a prefix of the block (`num_speculative_tokens` below 16) at concurrency 16, Qwen3-8B, temperature 0, output tokens/s (τ):

| Dataset   | FlashMap 16 | FlashMap 12 | FlashMap 8  | DFlash 15   | DFlash 11   | DFlash 7    |
| --------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| GSM8K     | 5910 (7.76) | 6095 (7.23) | 5608 (6.25) | 4863 (6.28) | 4989 (5.67) | 4547 (4.91) |
| HumanEval | 4859 (5.69) | 5099 (5.43) | 4826 (5.09) | 5221 (6.37) | 5145 (5.72) | 4696 (4.90) |
| MT-Bench  | 2358 (3.21) | 2605 (3.20) | 2609 (3.14) | 2306 (3.15) | 2246 (2.99) | 2288 (2.83) |

Draft step (GPU time of `propose`, CUDA graphs, median), in ms:

| Batch | FlashMap 8B | FlashMap 8B, 2 jumps | DFlash 8B | FlashMap 4B | FlashMap 4B, 2 jumps | DFlash 4B |
| ----- | ----------- | -------------------- | --------- | ----------- | -------------------- | --------- |
| 1     | 1.06        | 3.72                 | 2.15      | 1.04        | 3.21                 | 1.34      |
| 4     | 1.17        | 4.73                 | 2.26      | 1.14        | 4.17                 | 1.42      |
| 16    | 1.73        | 9.84                 | 2.84      | 1.65        | 8.57                 | 1.81      |
| 32    | 2.18        |                      | 3.50      |             |                      |           |

At temperature 0 with its noise set to zero, the vLLM draft for the first block of each of eight GSM8K prompts matches the reference implementation token for token except at bf16 near-ties of the reference logits (within 0.125).

## Research & Citation

FlashMap distills its drafter from a discrete flow following [Flow Map Language Models](https://arxiv.org/abs/2602.16813).

## See Also

- [DFlash](dflash.md) -- The block-parallel drafter whose runtime FlashMap reuses
- [DSpark](dspark.md) -- DFlash with a Markov head, sharing FlashMap's block layout
