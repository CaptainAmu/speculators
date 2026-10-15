# FlashMap

FlashMap is a block-parallel drafter like [DFlash](dflash.md): one forward pass of a small DiT proposes a whole block of `block_size` tokens, conditioned on the target model's hidden states. Its block is drawn from a distilled discrete *flow map* ([Flow Map Language Models](https://arxiv.org/abs/2602.16813)): each draft slot starts as Gaussian noise over the vocabulary, and the DiT maps the whole block to token distributions in one network evaluation (one "jump"), or in a few for higher acceptance. The target verifies every block, so the output is exactly the target's own.

## Training

Speculators trains FlashMap in two stages (a flow, then the flow map distilled from it) and converts the result to the FlashMap release format; see [Training FlashMap](flashmap_training.md). Install the FlashMap branch of speculators:

```bash
uv pip install "speculators @ git+https://github.com/al-jshen/speculators@flashmap"
```

## Serving

To serve FlashMap, use the [flashmap](https://github.com/al-jshen/flashmap) package. Its vLLM plugin serves the released drafters ([j-shen/FlashMap-Qwen3-8B](https://huggingface.co/j-shen/FlashMap-Qwen3-8B), [j-shen/FlashMap-Qwen3-4B](https://huggingface.co/j-shen/FlashMap-Qwen3-4B)) and drafters converted from speculators checkpoints:

```bash
uv pip install "vllm==0.31.*" "flashmap[vllm] @ git+https://github.com/al-jshen/flashmap"
vllm serve Qwen/Qwen3-8B --speculative-config '{"model": "j-shen/FlashMap-Qwen3-8B", "num_speculative_tokens": 16}'
```

The flashmap package documents the requirements and options of serving, and its SGLang and Hugging Face `transformers` integrations.

## See Also

- [Training FlashMap](flashmap_training.md) -- The two training stages and the conversion to the release format
- [DFlash](dflash.md) -- The block-parallel drafter whose conditioning FlashMap shares
- [flashmap](https://github.com/al-jshen/flashmap) -- Serving FlashMap in vLLM and SGLang, and the `transformers` implementation
