"""FlashMap speculator for vLLM's model runner V2.

FlashMap reuses the DFlash runtime unchanged: the context K/V precompute, the
fused input-preparation kernel, the attention metadata and the FULL CUDA graph
that covers the whole draft step. Its block has DSpark's layout
(``sample_from_anchor``): ``block_size`` query rows per request, row 0 holding the
anchor, every row a draft token. What differs is the draft model (model.py) and
sampling, which reads logits off the folded head over the draft vocabulary.

``num_speculative_tokens`` may be smaller than the block: the draft still
evaluates its whole block (the DiT's self-attention sees all of it) and proposes
the first ``num_speculative_tokens`` slots, so verification, which dominates at
high concurrency, checks fewer tokens.
"""

import torch
from torch import nn
from vllm.compilation.backends import set_model_tag
from vllm.config import VllmConfig, replace
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.model_loader.utils import get_draft_load_config
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.eagle.utils import get_target_lm_head
from vllm.v1.worker.gpu.spec_decode.utils import get_pp_safe_draft_load_config

logger = init_logger(__name__)

__all__ = ["FlashMapSpeculator"]


class FlashMapSpeculator(DFlashSpeculator):
    _speculator_name = "FlashMap"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        block_size = int(self.draft_model_config.hf_config.block_size)
        # Drafts proposed for verification, a prefix of the block.
        self.num_verify = self.num_speculative_steps
        if self.num_verify > block_size:
            raise ValueError(
                f"num_speculative_tokens={self.num_verify} exceeds FlashMap's "
                f"block_size={block_size}"
            )
        # Every one of the block's rows is a draft slot; row 0 is the anchor.
        self.sample_from_anchor = True
        self.num_query_per_req = block_size
        if self.num_verify < block_size:
            # Per-slot buffers for the whole block (the DFlash ones are sized
            # for the verified drafts).
            self.num_speculative_steps = block_size
            num_slots = self.max_num_reqs * block_size
            self.draft_tokens = torch.zeros(
                self.max_num_reqs, block_size, dtype=torch.int64, device=device
            )
            self.sample_indices = torch.zeros(
                num_slots, dtype=torch.int64, device=device
            )
            self.sample_pos = torch.zeros(num_slots, dtype=torch.int64, device=device)
            self.sample_idx_mapping = torch.full(
                (num_slots,), -1, dtype=torch.int32, device=device
            )
            self.sample_col = torch.arange(
                block_size, dtype=torch.int32, device=device
            ).repeat(self.max_num_reqs)
        self._full_logits: torch.Tensor | None = None

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],  # noqa: ARG002
    ) -> nn.Module:
        speculative_config = self.speculative_config
        draft_vllm_config = replace(
            self.vllm_config,
            attention_config=replace(
                self.vllm_config.attention_config,
                use_non_causal=True,
                backend=speculative_config.attention_backend,
            ),
            load_config=get_pp_safe_draft_load_config(
                get_draft_load_config(self.vllm_config)
            ),
        )
        with set_model_tag("flashmap"):
            model = get_model(
                vllm_config=draft_vllm_config,
                model_config=speculative_config.draft_model_config,
            )
        language_model = (
            target_model.get_language_model()
            if hasattr(target_model, "get_language_model")
            else target_model
        )
        embed_tokens = language_model.model.embed_tokens.weight
        lm_head = get_target_lm_head(target_model, language_model).weight
        model.prepare(embed_tokens, lm_head)
        if self.draft_logits is not None and model.hot_ids is not None:
            # Probabilistic verification reads the draft distribution over the
            # target vocabulary: tokens outside the draft vocabulary get -inf.
            self._full_logits = torch.full(
                (self.max_num_reqs * self.num_speculative_steps, self.vocab_size),
                float("-inf"),
                dtype=self.draft_logits.dtype,
                device=self.device,
            )
        return model

    def _precompute_context_kv(
        self, start: int, end: int, dummy_run: bool = False
    ) -> None:
        # The context rows a draft step stores right before its forward go
        # through the model's anchor GEMM (FLASHMAP_MERGE_ANCHOR).
        if self.model.merge_anchor and start == 0 and not dummy_run:
            self.model.defer_context_kv(
                self.hidden_states[start:end],
                self.context_positions[start:end],
                self._context_slot_mappings[0][start:end],
            )
            return
        super()._precompute_context_kv(start, end, dummy_run)

    def capture(self) -> None:
        # Record the KV caches' layout (a host-to-device copy) before capturing.
        self.model.cache_layout()
        super().capture()

    def propose(self, *args, **kwargs) -> torch.Tensor:
        return super().propose(*args, **kwargs)[:, : self.num_verify]

    def sample_draft(
        self,
        hidden_states: torch.Tensor,
        sample_src_positions: torch.Tensor,
        idx_mapping: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        draft_step: torch.Tensor,
        draft_logits: torch.Tensor | None,
    ) -> torch.Tensor:
        block, verify = self.num_speculative_steps, self.num_verify
        if verify == block:
            return self._sample(
                hidden_states,
                sample_src_positions,
                idx_mapping,
                temperature,
                seeds,
                draft_step,
                draft_logits,
            )
        # Sample the verified prefix of each block only; the rest stay zero.
        num_blocks = hidden_states.shape[0] // block

        def prefix(t: torch.Tensor) -> torch.Tensor:
            return t.view(num_blocks, block, *t.shape[1:])[:, :verify].flatten(0, 1)

        sampled = self._sample(
            prefix(hidden_states),
            prefix(sample_src_positions),
            prefix(idx_mapping),
            temperature,
            seeds,
            prefix(draft_step),
            draft_logits,
        )
        tokens = sampled.new_zeros(num_blocks, block)
        tokens[:, :verify] = sampled.view(num_blocks, verify)
        return tokens.view(-1)

    def _sample(
        self,
        hidden_states: torch.Tensor,
        sample_src_positions: torch.Tensor,
        idx_mapping: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        draft_step: torch.Tensor,
        draft_logits: torch.Tensor | None,
    ) -> torch.Tensor:
        logits = self.model.compute_draft_logits(hidden_states)
        hot_ids = self.model.hot_ids
        if draft_logits is None:
            draft_ids = logits.argmax(dim=-1)
            return draft_ids if hot_ids is None else hot_ids[draft_ids]
        if hot_ids is not None:
            assert self._full_logits is not None
            full = self._full_logits[: logits.shape[0]]
            full.index_copy_(1, hot_ids, logits.to(full.dtype))
            logits = full
        sampled = gumbel_sample(
            logits,
            idx_mapping,
            temperature,
            seeds,
            sample_src_positions,
            apply_temperature=True,
            is_drafting=True,
            logits_cache=draft_logits,
            logits_cache_col=draft_step,
            use_fp64=self.use_fp64_gumbel,
        )
        self._maybe_predict_acceptance(logits, idx_mapping, draft_step)
        return sampled
