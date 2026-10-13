"""FlashMap speculator for vLLM's model runner V2.

FlashMap reuses the DFlash runtime unchanged: the context K/V precompute, the
fused input-preparation kernel, the attention metadata and the FULL CUDA graph
that covers the whole draft step. Its block has DSpark's layout
(``sample_from_anchor``): ``block_size`` query rows per request, row 0 holding the
anchor, every row a draft token. What differs is the draft model (model.py) and
sampling, which reads logits off the folded head over the draft vocabulary.
"""

import torch
import torch.nn as nn

from vllm.config import VllmConfig
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
        # Every one of the block's rows is a draft token; row 0 is the anchor.
        self.sample_from_anchor = True
        self.num_query_per_req = self.num_speculative_steps
        self._full_logits: torch.Tensor | None = None

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        from vllm.compilation.backends import set_model_tag
        from vllm.config import replace

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
