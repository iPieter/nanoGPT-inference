import time
import copy
import torch
import torch.nn.functional as F
from model import CausalSelfAttention
from . import register
from .base import InferenceEngine

class KVCache:
    """
    External KV cache for transformer attention layers.
    Manages key and value caches across all layers.
    """
    def __init__(self, num_layers: int, batch_size: int, num_heads: int,
                 head_dim: int, max_seq_len: int, dtype: torch.dtype, device: torch.device):
        """
        Initialize KV cache for all layers.

        Args:
            num_layers: Number of transformer layers
            batch_size: Batch size
            num_heads: Number of attention heads
            head_dim: Dimension of each attention head
            max_seq_len: Maximum sequence length to cache
            dtype: Data type for cache tensors
            device: Device to store cache on
        """
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.device = device

        # Initialize cache tensors for all layers
        # Shape: (num_layers, batch_size, num_heads, max_seq_len, head_dim)
        self.k_cache = torch.zeros(
            num_layers, batch_size, num_heads, max_seq_len, head_dim,
            dtype=dtype, device=device
        )
        self.v_cache = torch.zeros(
            num_layers, batch_size, num_heads, max_seq_len, head_dim,
            dtype=dtype, device=device
        )
        self.cache_len = 0

    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache for a specific layer and return full static-sized cache tensors.

        Args:
            layer_idx: Index of the transformer layer
            k: New key tensor (B, nh, T, hs)
            v: New value tensor (B, nh, T, hs)

        Returns:
            Tuple of (cached_k, cached_v) with full static max_seq_len size
            (use attention mask to ignore positions beyond cache_len)
        """
        T = k.size(2)

        # Store new keys and values in cache
        self.k_cache[layer_idx, :, :, self.cache_len:self.cache_len+T, :] = k
        self.v_cache[layer_idx, :, :, self.cache_len:self.cache_len+T, :] = v

        # Return full cache with static shape (max_seq_len)
        # Positions beyond cache_len+T will be zeros and should be masked out
        k_full = self.k_cache[layer_idx]  # (B, nh, max_seq_len, hs)
        v_full = self.v_cache[layer_idx]  # (B, nh, max_seq_len, hs)

        return k_full, v_full

    def advance(self, num_tokens: int = 1):
        """Advance the cache position by num_tokens."""
        self.cache_len += num_tokens

    def clear(self):
        """Reset cache to initial state."""
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.cache_len = 0

    def get_current_length(self) -> int:
        """Return current cache length."""
        return self.cache_len

class ExternalCachedSelfAttention(CausalSelfAttention):
    """
    Self-attention layer that uses an external KV cache passed in forward().
    """
    def __init__(self, config, layer_idx: int):
        super().__init__(config)
        self.layer_idx = layer_idx

    def forward(self, x, kv_cache: KVCache | None = None, attn_mask: torch.Tensor | None = None):
        """
        Forward pass with optional external KV cache and attention mask.

        Args:
            x: Input tensor (B, T, C)
            kv_cache: Optional external KV cache to use
            attn_mask: Optional attention mask (B, 1, T, S) where S is cached sequence length
                      True/1.0 = attend, False/-inf = ignore

        Returns:
            Output tensor (B, T, C)
        """
        B, T, C = x.size()

        # Compute queries, keys, and values
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        k_full, v_full = kv_cache.update(self.layer_idx, k, v)

        y = F.scaled_dot_product_attention(q, k_full, v_full, attn_mask=attn_mask)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


@register
class CUDAGraphEngine(InferenceEngine):
    name = "cuda_graphs"
    description = "CUDA Graphs (External KV cache, CUDA graphs for generation)"

    def __init__(self, model):
        super().__init__(copy.deepcopy(model))
        # Move original model to CPU to save GPU memory
        model.to('cpu')

        # Replace attention layers with external cached variant
        for layer_idx, block in enumerate(self.model.transformer.h):
            cached_attn = ExternalCachedSelfAttention(self.config, layer_idx).to(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            cached_attn.load_state_dict(block.attn.state_dict())
            block.attn = cached_attn

        self.kv_cache = None

        # CUDA graphs cache: dict[(batch_size, seq_len)] -> (graph, static_inputs, static_outputs)
        self.cuda_graphs = {}

    def _forward_with_cache(self, idx, position_offset=0, attn_mask=None):
        """
        Forward pass that injects KV cache and attention mask into attention layers.
        Based on GPT.forward() from model.py, modified to pass kv_cache and attn_mask to attention.
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(position_offset, position_offset + t, dtype=torch.long, device=device)

        # Forward the GPT model itself (from model.py)
        tok_emb = self.model.transformer.wte(idx)
        pos_emb = self.model.transformer.wpe(pos)
        x = self.model.transformer.drop(tok_emb + pos_emb)
        for block in self.model.transformer.h:
            # Modified: pass kv_cache and attn_mask to attention instead of using block(x)
            x = x + block.attn(block.ln_1(x), self.kv_cache, attn_mask)
            x = x + block.mlp(block.ln_2(x))
        x = self.model.transformer.ln_f(x)

        logits = self.model.lm_head(x[:, [-1], :])
        loss = None

        return logits, loss

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

        # Initialize external KV cache
        batch_size = idx.size(0)
        max_seq_len = idx.size(1) + max_new_tokens
        head_dim = self.config.n_embd // self.config.n_head

        self.kv_cache = KVCache(
            num_layers=self.config.n_layer,
            batch_size=batch_size,
            num_heads=self.config.n_head,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            dtype=next(self.model.parameters()).dtype,
            device=device
        )

        for i in range(max_new_tokens):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t_start = time.perf_counter()

            idx_input = idx if i == 0 else idx[:, -1:]
            position_offset = 0 if i == 0 else idx.size(1) - 1

            # Calculate attention mask with static shape (max_seq_len)
            # Mask shape must be broadcastable to (B, num_heads, L, S)
            # where L = query length, S = max_seq_len (static cache size)
            if i == 0:
                # Prefill: causal mask for prompt, zeros for future positions
                # L = prompt_len, S = max_seq_len
                prompt_len = idx.size(1)
                # Create full mask with static shape: (1, 1, prompt_len, max_seq_len) (this gets broadcasted)
                attn_mask = torch.zeros(1, 1, prompt_len, max_seq_len, dtype=torch.bool, device=device)
                # Fill in causal mask for actual prompt positions
                causal = torch.tril(torch.ones(prompt_len, prompt_len, dtype=torch.bool, device=device))
                attn_mask[0, 0, :, :prompt_len] = causal
            else:
                # Decode: attend to all tokens in cache, ignore future positions
                # L = 1 (single query), S = max_seq_len (static)
                # The cache has been updated with new k,v, so cache_len includes the new token
                cache_len = self.kv_cache.get_current_length() + 1  # +1 for the token being added
                # Create mask: (1, 1, 1, max_seq_len) (again broad casted to batch)
                attn_mask = torch.zeros(1, 1, 1, max_seq_len, dtype=torch.bool, device=device)
                # Allow attention to all valid cached positions
                attn_mask[0, 0, 0, :cache_len] = True

            # Check if we have a CUDA graph for this (batch_size, seq_len) combination
            B, L = idx_input.size()
            graph_key = (B, L)

            if graph_key not in self.cuda_graphs:
                print("Capture CUDA graph for", graph_key)
                static_input = torch.zeros(B, L, dtype=torch.long, device=device)
                static_attn_mask = torch.zeros_like(attn_mask)

                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    static_logits, _ = self._forward_with_cache(static_input, position_offset, static_attn_mask)

                self.cuda_graphs[graph_key] = (g, static_input, static_attn_mask, static_logits)

            # Use captured graph
            g, static_input, static_attn_mask, static_logits = self.cuda_graphs[graph_key]
            static_input.copy_(idx_input)
            static_attn_mask.copy_(attn_mask)
            g.replay()
            logits = static_logits.clone()  # Clone to avoid overwriting in next iteration

            logits = logits[:, -1, :] / temperature

            # Advance cache position after prefill
            if i == 0:
                self.kv_cache.advance(idx_input.size(1))
            else:
                self.kv_cache.advance(1)

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
