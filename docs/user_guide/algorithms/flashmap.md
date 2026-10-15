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

!!! tip "Inference only"
    The [flashmap](https://github.com/al-jshen/flashmap) package ships this same plugin without speculators, as `flashmap[vllm]`, and serves the released FlashMap folders as they are:

    ```bash
    uv pip install "vllm==0.31.*" "flashmap[vllm] @ git+https://github.com/al-jshen/flashmap"
    vllm serve Qwen/Qwen3-8B --speculative-config '{"model": "j-shen/FlashMap-Qwen3-8B", "num_speculative_tokens": 16}'
    ```

    The copy here (`speculators.models.flashmap.vllm`) is the one that trains with speculators and that would be upstreamed to vLLM; it also serves speculators checkpoints. Both register under the plugin name `flashmap`, so install one of them.

Requirements:

- vLLM 0.31.x, with speculators installed in the same environment (on other versions the plugin logs a warning and stays out of the way);
- vLLM's V2 model runner (the default; not `VLLM_USE_V2_MODEL_RUNNER=0`);
- tensor parallel size 1;
- the FlashAttention attention backend, and a bf16 draft KV cache or, with FlashAttention 3 (Hopper), an `fp8_e4m3` one.

Each of these is checked when the draft model loads, with an error that names it. FlashMap supports vLLM's prefix caching (on by default): requests that reuse a cached prefix draft and accept as they do without it.

A released FlashMap folder is served as it is, with its 64k-token draft vocabulary (its config.json carries the speculators header). To pick another draft vocabulary, or to serve a DeepSpec flow-map checkpoint, convert it to the speculators format:

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
    '{"model": "./FlashMap-Qwen3-8B-speculators", "num_speculative_tokens": 16}'
```

The plugin fills in `"method": "dflash"` for a FlashMap checkpoint (and `num_speculative_tokens`, the block size, when it is left out).

For sampling at temperature > 0, add `"draft_sample_method": "probabilistic"` to the speculative config so that drafts are sampled from FlashMap's distribution and verified against it (with the default `greedy` draft sampling verification is still exact, but accepts less).

`num_speculative_tokens` may be smaller than the block (16): the draft still evaluates its whole block and proposes its first `num_speculative_tokens` slots. Verification then checks fewer tokens, which pays off at high concurrency, where verification dominates.

| Environment variable | Default            | Description                                                                                                   |
| -------------------- | ------------------ | ------------------------------------------------------------------------------------------------------------- |
| `FLASHMAP_NUM_STEPS` | config `num_steps` | Jumps per draft block (1 is fastest); also set by `speculators convert --algorithm-kwargs '{"num_steps": 2}'` |
| `FLASHMAP_FP8`       | `0`                | `1`: FP8 draft weights (see below; Ada or Hopper)                                                             |

### FP8 Draft Weights and KV Cache

`FLASHMAP_FP8=1` keeps e4m3 copies of the draft step's large GEMM weights (the DiT's linear layers, the folded head, the context/anchor K/V projection, `fc` and the folded noise read-out), quantized per output channel at load, and quantizes their inputs per token, through vLLM's `scaled_fp8_quant` and `cutlass_scaled_mm`. It needs an Ada or Hopper GPU.

An fp8 draft KV cache is vLLM's own option, independent of the target's KV cache dtype: add `"kv_cache_dtype": "fp8_e4m3"` to the speculative config. The context write stores e4m3 with each layer's scale, and the cross-attention reads it with FlashAttention 3's descales (Hopper only). It halves the draft's share of the KV cache.

Both keep the acceptance length unchanged and greedy outputs the target's own, and both are opt-in. FP8 weights shorten the draft step at small batch sizes but not at larger ones, and end-to-end throughput stays within noise. The fp8 draft KV cache leaves the draft step's time unchanged and gives the freed memory to the KV cache.

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
- `v1/worker/gpu/spec_decode/__init__.py` (`init_speculator`): return `FlashMapSpeculator` when `"FlashMapDraftModel"` is in the draft's architectures, next to the `DFlash2DraftModel` case;
- `config/speculative.py`: detect `"FlashMapDraftModel"` among the draft's architectures as `"dflash"`, next to DFlash's detection (the plugin fills in the method before vLLM's own detection instead).

## Training

FlashMap is trained with `speculators train --speculator-type flashmap` in two stages: the flow, then the flow map distilled from it (`--flashmap-teacher`). The flow map's checkpoints are in the layout served above. See [Training FlashMap](flashmap_training.md).

## Research & Citation

FlashMap distills its drafter from a discrete flow following [Flow Map Language Models](https://arxiv.org/abs/2602.16813).

## See Also

- [Training FlashMap](flashmap_training.md) -- The two training stages through `speculators train`
- [DFlash](dflash.md) -- The block-parallel drafter whose runtime FlashMap reuses
- [DSpark](dspark.md) -- DFlash with a Markov head, sharing FlashMap's block layout
