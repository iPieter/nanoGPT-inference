"""
Base class for inference implementations.

All inference engines inherit from InferenceEngine and implement
the generate() method with a consistent interface.
"""

from abc import ABC, abstractmethod
from typing import Optional
import torch
from model import GPT


class InferenceEngine(ABC):
    """
    Base class for inference implementations.

    Philosophy: Engines are orchestrators, not replacements.
    They wrap a GPT model and define how to generate tokens.

    Each engine can:
    - Modify the forward pass (via hooks, subclassing, etc.)
    - Maintain state (caches, schedulers, etc.)
    - Implement different generation strategies

    But they all expose the same interface.
    """

    # Metadata (subclasses set these)
    name: str = "base"
    description: str = "Base inference engine"

    def __init__(self, model: GPT):
        self.model = model
        self.config = model.config

    @abstractmethod
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.

        Args:
            idx: (B, T) input token indices
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature (1.0 = no change)
            top_k: if set, only sample from top k tokens
            **kwargs: engine-specific parameters

        Returns:
            (B, T + max_new_tokens) generated token indices
        """
        pass

    def reset_stats(self):
        """Reset any tracked statistics."""
        pass

    def get_stats(self) -> dict:
        """Return statistics about the last generation."""
        return {}
