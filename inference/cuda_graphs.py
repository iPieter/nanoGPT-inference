import time
import copy
import torch
import torch.nn.functional as F
from model import CausalSelfAttention
from flashinfer import top_k_top_p_sampling_from_logits
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
            num_heads: Number of attention heads (LOCAL to this device in TP mode)
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
        # In TP mode, num_heads is the local number of heads on this device
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

    def forward(self, x, k_cache=None, v_cache=None, cache_position=None, attn_mask: torch.Tensor | None = None):
        """
        Forward pass with external KV cache tensors and attention mask.

        Args:
            x: Input tensor (B, T, C)
            k_cache: Key cache tensor (B, nh, max_seq_len, hs)
            v_cache: Value cache tensor (B, nh, max_seq_len, hs)
            cache_position: Tensor scalar indicating where to write in cache
            attn_mask: Optional attention mask (B, 1, T, S) where S is cached sequence length
                      True/1.0 = attend, False/-inf = ignore

        Returns:
            Output tensor (B, T, C) 
        """
        B, T, C = x.size()

        # Compute queries, keys, and values
        # In TP mode, c_attn is ColwiseParallel, so output is already sharded across heads
        # c_attn output shape: (B, T, 3 * n_embd_local) where n_embd_local may be < n_embd
        qkv = self.c_attn(x)
        qkv_dim = qkv.size(-1) // 3  # Local embedding dimension per Q/K/V
        q, k, v = qkv.split(qkv_dim, dim=2)

        # n_head here reflects the LOCAL number of heads on this device
        # head_dim is computed from local dimensions
        head_dim = qkv_dim // self.n_head
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        if k_cache is not None and v_cache is not None and cache_position is not None:
            # Write k, v to cache at the specified position using tensor indexing
            # Use index_copy to avoid .item() which isn't allowed in CUDA graphs
            pos = cache_position[0]  # Extract scalar but keep as tensor operation

            # Create indices for the positions to write to
            indices = torch.arange(T, device=k.device) + pos

            # Use index_copy_ to write k, v to cache (CUDA graph compatible)
            k_cache.index_copy_(2, indices, k)
            v_cache.index_copy_(2, indices, v)

            # Use full cache for attention
            k_full, v_full = k_cache, v_cache
        else:
            # No cache - use current k, v directly
            k_full, v_full = k, v

        y = F.scaled_dot_product_attention(q, k_full, v_full, attn_mask=attn_mask)
        # Reshape back: use local qkv_dim since heads are sharded
        y = y.transpose(1, 2).contiguous().view(B, T, qkv_dim)

        # In TP mode, c_proj is RowwiseParallel, so it performs all-reduce internally
        # c_proj input is sharded (qkv_dim)
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

    def _forward_with_cache(self, idx, position_offset=0, attn_mask=None, last_token_idx=None, cache_position=None):
        """
        Forward pass that injects KV cache and attention mask into attention layers.
        Based on GPT.forward() from model.py, modified to pass cache tensors and attn_mask to attention.

        Args:
            idx: Input tensor
            position_offset: Position offset for embeddings
            attn_mask: Attention mask
            last_token_idx: Index tensor for selecting which position to compute logits for.
            cache_position: Tensor scalar for cache write position
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(position_offset, position_offset + t, dtype=torch.long, device=device)

        # Forward the GPT model itself (from model.py)
        tok_emb = self.model.transformer.wte(idx)
        pos_emb = self.model.transformer.wpe(pos)
        x = self.model.transformer.drop(tok_emb + pos_emb)
        for layer_idx, block in enumerate(self.model.transformer.h):
            # Get this layer's cache slices
            k_cache = self.kv_cache.k_cache[layer_idx] if self.kv_cache else None
            v_cache = self.kv_cache.v_cache[layer_idx] if self.kv_cache else None
            # Modified: pass cache tensors, position, and attn_mask to attention
            x = x + block.attn(block.ln_1(x), k_cache, v_cache, cache_position, attn_mask)
            x = x + block.mlp(block.ln_2(x))
        x = self.model.transformer.ln_f(x)

        # equivalent to lm_head([:, [-1], :]) but CUDA graph friendly
        if last_token_idx is not None:
            x = torch.index_select(x, 1, last_token_idx)  # (B, 1, n_embd)

        logits = self.model.lm_head(x)
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

        # Initialize or reuse external KV cache
        batch_size = idx.size(0)
        max_seq_len = idx.size(1) + max_new_tokens
        head_dim = self.config.n_embd // self.config.n_head

        # Reuse cache if it exists and has the right shape
        if (self.kv_cache is None or
            self.kv_cache.batch_size != batch_size or
            self.kv_cache.max_seq_len < max_seq_len):
            self.kv_cache = KVCache(
                num_layers=self.config.n_layer,
                batch_size=batch_size,
                num_heads=self.config.n_head,
                head_dim=head_dim,
                max_seq_len=max_seq_len,
                dtype=torch.bfloat16,
                device=device
            )
        else:
            # Clear existing cache for reuse
            self.kv_cache.clear()

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
                # Static index and cache position tensors
                static_last_token_idx = torch.tensor([L - 1], dtype=torch.long, device=device)
                static_cache_position = torch.tensor([self.kv_cache.get_current_length()], dtype=torch.long, device=device)
                static_input.copy_(idx_input)
                static_attn_mask.copy_(attn_mask)

                # Synchronize streams before capture
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    # Warmup iterations
                    for _ in range(3):
                        self._forward_with_cache(static_input, position_offset, static_attn_mask, static_last_token_idx, static_cache_position)
                torch.cuda.current_stream().wait_stream(s)

                # Capture graph
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    static_logits, _ = self._forward_with_cache(static_input, position_offset, static_attn_mask, static_last_token_idx, static_cache_position)

                self.cuda_graphs[graph_key] = (g, static_input, static_attn_mask, static_last_token_idx, static_cache_position, static_logits)

            # Use captured graph
            g, static_input, static_attn_mask, static_last_token_idx, static_cache_position, static_logits = self.cuda_graphs[graph_key]
            static_input.copy_(idx_input)
            static_attn_mask.copy_(attn_mask)
            # Update cache position for this replay
            static_cache_position.fill_(self.kv_cache.get_current_length())
            g.replay()

            # No need to clone, we can work with static_logits directly since we're not modifying it
            logits = static_logits.squeeze(1) / temperature

            # Advance cache position
            if i == 0:
                self.kv_cache.advance(idx_input.size(1))
            else:
                self.kv_cache.advance(1)

            # Use flashinfer optimized sampling kernel
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
