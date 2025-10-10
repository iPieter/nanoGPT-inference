import time
import copy
import torch
import torch.nn.functional as F
from model import CausalSelfAttention
from . import register
from .base import InferenceEngine


class CachedSelfAttention(CausalSelfAttention):
    def __init__(self, config):
        super().__init__(config)
        self.k_cache = None
        self.v_cache = None
        self.cache_len = 0
        self.max_seq_len = None

    def forward(self, x):
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.k_cache is None:
            self.k_cache = torch.zeros(B, self.n_head, self.max_seq_len, C // self.n_head,
                                      dtype=k.dtype, device=k.device)
            self.v_cache = torch.zeros(B, self.n_head, self.max_seq_len, C // self.n_head,
                                      dtype=v.dtype, device=v.device)

        self.k_cache[:, :, self.cache_len:self.cache_len+T, :] = k
        self.v_cache[:, :, self.cache_len:self.cache_len+T, :] = v

        k_full = self.k_cache[:, :, :self.cache_len+T, :]
        v_full = self.v_cache[:, :, :self.cache_len+T, :]

        self.cache_len += T

        # We set causality to true only during prefill
        #
        # Prefill phase (T>1): Processing initial prompt tokens
        #   q shape: (B, nh, T, hs)      e.g. (1, 12, 10, 64) - 10 prompt tokens
        #   k shape: (B, nh, T, hs)      e.g. (1, 12, 10, 64)
        #   Need is_causal=True so token i can only attend to tokens 0..i
        #
        # Decode phase (T=1): Generating one new token at a time
        #   q shape: (B, nh, 1, hs)      e.g. (1, 12, 1, 64)  - 1 new query
        #   k shape: (B, nh, N, hs)      e.g. (1, 12, 15, 64) - all N cached keys
        #   With is_causal=True, q[0] would only attend to k[0] 
        #   Need is_causal=False so the new token attends to all cached tokens
        is_causal = T > 1

        y = F.scaled_dot_product_attention(q, k_full, v_full, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=is_causal)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

    def clear_cache(self):
        self.k_cache = None
        self.v_cache = None
        self.cache_len = 0


@register
class KVCacheEngine(InferenceEngine):
    name = "kv_cache"
    description = "KV cache (no recomputation of past keys/values)"

    def __init__(self, model):
        super().__init__(copy.deepcopy(model))
        # Move original model to CPU to save GPU memory
        model.to('cpu')

        # We modify our transformer's attention to the cached variant
        for block in self.model.transformer.h:
            cached_attn = CachedSelfAttention(self.config).to("cuda" if torch.cuda.is_available() else "cpu")
            cached_attn.load_state_dict(block.attn.state_dict())
            block.attn = cached_attn

    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        **kwargs
    ) -> torch.Tensor:
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

            idx = torch.cat((idx, idx_next), dim=1)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            t_end = time.perf_counter()
            per_token_times.append(t_end - t_start)

        self._calculate_token_stats(per_token_times)
        return idx
