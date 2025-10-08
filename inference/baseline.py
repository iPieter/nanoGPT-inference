"""
Baseline inference engine - no optimizations.

This is a thin wrapper around model.generate() for:
1. Establishing a performance baseline
2. Demonstrating the engine interface
3. Providing a reference for correctness

Educational notes:
- Recomputes full attention for all previous tokens on every step
- Compute complexity: O(n²) per generated token where n is sequence length
- Memory complexity: O(n) for storing the sequence
- No caching or optimization of any kind
"""

import torch
from . import register
from .base import InferenceEngine


@register
class BaselineEngine(InferenceEngine):
    """Uses original model.generate() with no optimizations."""

    name = "baseline"
    description = "Baseline (no optimizations, recomputes all attention)"

    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = None,
        **kwargs
    ) -> torch.Tensor:
        """Delegate to model.generate()."""
        return self.model.generate(
            idx,
            max_new_tokens,
            temperature=temperature,
            top_k=top_k
        )
