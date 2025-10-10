"""
Base class for inference implementations.

All inference engines inherit from InferenceEngine and implement
the generate() method with a consistent interface.
"""

from abc import ABC, abstractmethod
import torch
from model import GPT


class InferenceEngine(ABC):
    """
    Base class for inference implementations.

    Philosophy: Engines are orchestrators, not replacements.
    They can wrap a GPT model and define how to generate tokens.
    """

    # Metadata (subclasses set these)
    name: str = "base"
    description: str = "Base inference engine"

    def __init__(self, model: GPT):
        # Ensure model is on GPU if available
        if torch.cuda.is_available() and next(model.parameters()).device.type == 'cpu':
            self.model = model.to('cuda')
        else:
            self.model = model
        self.config = model.config

        # Track timing stats
        self._last_ttft = 0.0
        self._last_itl = 0.0
        self._last_prefill_time = 0.0
        self._last_decode_time = 0.0

    @abstractmethod
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
        Generate tokens autoregressively.

        Args:
            idx: (B, T) input token indices where B is batch size
                 Engines should handle batched inputs natively
            max_new_tokens: number of tokens to generate per sequence
            temperature: sampling temperature (1.0 = no change)
            top_k: if set, only sample from top k tokens
            top_p: if set, only sample from top p cumulative probability
            **kwargs: engine-specific parameters

        Returns:
            (B, T + max_new_tokens) generated token indices
        """
        pass

    def _calculate_token_stats(self, per_token_times):
        if per_token_times:
            self._last_ttft = per_token_times[0]  # First token time
            self._last_prefill_time = per_token_times[0]

            if len(per_token_times) > 1:
                # Average of remaining tokens
                decode_times = per_token_times[1:]
                self._last_itl = sum(decode_times) / len(decode_times)
                self._last_decode_time = sum(decode_times)
            else:
                self._last_itl = 0.0
                self._last_decode_time = 0.0

    def get_stats(self) -> dict:
        """Return timing statistics from the last generation."""
        return {
            'ttft': self._last_ttft,
            'itl': self._last_itl,
            'prefill_time': self._last_prefill_time,
            'decode_time': self._last_decode_time,
        }

    def reset_stats(self):
        """Reset timing statistics."""
        self._last_ttft = 0.0
        self._last_itl = 0.0
        self._last_prefill_time = 0.0
        self._last_decode_time = 0.0

