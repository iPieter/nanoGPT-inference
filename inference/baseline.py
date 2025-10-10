"""
Baseline inference engine - no optimizations.
"""

import time
import torch
import torch.nn.functional as F
from . import register
from .base import InferenceEngine


@register
class BaselineEngine(InferenceEngine):
    """Baseline generation with no optimizations but proper timing."""

    name = "baseline"
    description = "Baseline (no optimizations, recomputes all attention)"

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
        Generate tokens autoregressively with timing instrumentation.

        This is essentially model.generate() but with timing measurements
        to properly track TTFT (time to first token) and ITL (inter-token latency).

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

        for i in range(max_new_tokens):
            # Start timing this token
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t_start = time.perf_counter()

            # Crop context if needed (same as model.generate)
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            # Forward the model to get logits
            logits, _ = self.model(idx_cond)

            # Pluck logits at final step and scale by temperature
            logits = logits[:, -1, :] / temperature

            # Optionally crop logits to only top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Apply softmax after our tok_k to get probabilities
            probs = F.softmax(logits, dim=-1)

            # Apply top-p (nucleus) sampling
            if top_p is not None:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Keep at least the first token
                sorted_indices_to_remove[..., 0] = False
                # Scatter back to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                probs[indices_to_remove] = 0.0

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)

            # End timing this token
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t_end = time.perf_counter()

            per_token_times.append(t_end - t_start)

        # Calculate and store timing statistics
        self._calculate_token_stats(per_token_times)

        return idx

