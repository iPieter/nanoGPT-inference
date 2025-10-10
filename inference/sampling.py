"""
Sampling improvements with KV cache.
"""

import time
import torch
from flashinfer import top_k_top_p_sampling_from_logits
from . import register
from .kv_cache import KVCacheEngine

@register
class SamplingEngine(KVCacheEngine):
    """KV cache with optimized sampling kernels (flashinfer)."""

    name = "sampling"
    description = "Sampling (KV cache + rejection sampling via Flashinfer)"

    def __init__(self, model):
        super().__init__(model)

    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate tokens with KV cache and optimized sampling kernels.

        Args:
            idx: (B, T) input token indices
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature
            top_k: top-k sampling
            top_p: top-p (nucleus) sampling

        Returns:
            (B, T + max_new_tokens) generated tokens
        """
        device = idx.device
        per_token_times = []

        max_seq_len = idx.size(1) + max_new_tokens
        for block in self.model.transformer.h:
            block.attn.max_seq_len = max_seq_len
            block.attn.clear_cache()

        for i in range(max_new_tokens):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t_start = time.perf_counter()

            idx_input = idx if i == 0 else idx[:, -1:]
            position_offset = 0 if i == 0 else idx.size(1) - 1
            logits, _ = self.model(idx_input, position_offset=position_offset)
            logits = logits[:, -1, :] / temperature

            idx_next = top_k_top_p_sampling_from_logits(
                logits,
                top_k=top_k if top_k is not None else -1,
                top_p=top_p if top_p is not None else 1.0,
                filter_apply_order="joint"
            ).unsqueeze(-1)

            idx = torch.cat((idx, idx_next), dim=1)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            t_end = time.perf_counter()
            per_token_times.append(t_end - t_start)

        self._calculate_token_stats(per_token_times)
        return idx

