import jax
import jax.numpy as jnp
import numpy as np

def tiled_attention(queries, keys, values, tile_size=1024, pre=True, position_offset=0):
    m, d_k = queries.shape
    n, _ = keys.shape
    head_dim = d_k

    # Initialize accumulators
    Fi = jnp.zeros((m, values.shape[-1]))   # weighted sum
    Li = jnp.zeros((m, 1))                  # normalizer (sum of exp logits)
    Mi = jnp.full((m, 1), -jnp.inf)         # running max

    init = (Fi, Li, Mi)

    # Reshape keys and values into tiles
    num_tiles = n // tile_size
    ks = jnp.reshape(keys[:num_tiles * tile_size], (num_tiles, tile_size, d_k))
    vs = jnp.reshape(values[:num_tiles * tile_size], (num_tiles, tile_size, d_k))
    xs = (ks, vs)

    def bodyfn(carry, x):
        k_chunk, v_chunk = x
        F, L, M = carry

        # Compute attention logits: (m, tile_sz)
        logits = jnp.matmul(queries, k_chunk.T) / jnp.sqrt(head_dim)

        # Apply causal mask
        if pre:
            # Prefill mask: (m, tile_sz)
            start_idx = x[0].shape[0] * xs[0].index(x)  # this doesn't work due to scan
            # Instead: compute offset directly from loop index via scan state trick
            offset = jnp.arange(tile_size)[None, :] + start_idx  # shape: (1, tile_sz)
            q_positions = jnp.arange(m)[:, None]  # (m, 1)
            causal_mask = offset > q_positions  # (m, tile_sz)
        else:
            # Generation mode: only attend up to position_offset + 1
            start_idx = x[0].shape[0] * xs[0].index(x)  # again not usable directly
            # So we simulate: compute tile start index
            offset = jnp.arange(tile_size)[None, :] + start_idx  # (1, tile_sz)
            causal_mask = offset > (position_offset + 1)

        # Mask out future tokens
        logits = jnp.where(causal_mask, -jnp.inf, logits)

        # Numerically stable softmax
        max_logits = jnp.max(logits, axis=-1, keepdims=True)  # (m, 1)
        new_max = jnp.maximum(M, max_logits)
        exp_logits = jnp.exp(logits - new_max)
        old_shift = jnp.exp(M - new_max)
        L = L * old_shift + jnp.sum(exp_logits, axis=-1, keepdims=True)
        F = F * old_shift + jnp.matmul(exp_logits, v_chunk)
        M = new_max

        return (F, L, M), None

    # Use jax.lax.scan to process each tile
    (F, L, _), _ = jax.lax.scan(bodyfn, init, xs)
    context = F / L
    return context

def tiled_attention_cleaner(queries, keys, values, tile_size=1024, pre=True, position_offset=0):
    m, d_k = queries.shape
    n, _ = keys.shape
    head_dim = d_k

    Fi = jnp.zeros((m, values.shape[-1]))
    Li = jnp.zeros((m, 1))
    Mi = jnp.full((m, 1), -jnp.inf)
    init = (Fi, Li, Mi)

    num_full_tiles = n // tile_size
    ks = jnp.reshape(keys[:num_full_tiles * tile_size], (num_full_tiles, tile_size, d_k))
    vs = jnp.reshape(values[:num_full_tiles * tile_size], (num_full_tiles, tile_size, d_k))

    def bodyfn(carry, scan_input):
        k_chunk, v_chunk, start_idx = scan_input
        F, L, M = carry

        logits = jnp.matmul(queries, k_chunk.T) / jnp.sqrt(head_dim)  # (m, tile_sz)

        # Causal masking
        q_positions = jnp.arange(m)[:, None]  # (m, 1)
        k_positions = jnp.arange(tile_size)[None, :] + start_idx  # (1, tile_sz)

        if pre:
            causal_mask = k_positions > q_positions  # (m, tile_sz)
        else:
            causal_mask = k_positions > (position_offset + 1)  # (m, tile_sz)

        logits = jnp.where(causal_mask, -jnp.inf, logits)

        # Softmax
        max_logits = jnp.max(logits, axis=-1, keepdims=True)
        new_max = jnp.maximum(M, max_logits)
        exp_logits = jnp.exp(logits - new_max)
        old_shift = jnp.exp(M - new_max)
        L = L * old_shift + jnp.sum(exp_logits, axis=-1, keepdims=True)
        F = F * old_shift + jnp.matmul(exp_logits, v_chunk)
        M = new_max

        return (F, L, M), None

    # Pass start indices with chunks
    start_indices = jnp.arange(0, num_full_tiles * tile_size, tile_size)
    scan_inputs = (ks, vs, start_indices)

    (F, L, _), _ = jax.lax.scan(bodyfn, init, scan_inputs)
    return F / L

def att_head_coder_causal(queries, keys, values, pre, position_offset, tile_size=1024):
    #set_trace()
    dtype_l = jnp.float64
    queries = queries.astype(dtype_l)
    keys = keys.astype(dtype_l)
    values = values.astype(dtype_l)

    m, d_k = queries.shape
    n, _ = keys.shape
    head_dim = d_k

    Fi = jnp.zeros((m, values.shape[-1]), dtype=dtype_l)
    Li = jnp.zeros((m, 1), dtype=dtype_l)
    Mi = jnp.full((m, 1), -jnp.inf, dtype=dtype_l)
    init = (Fi, Li, Mi)

    num_full_tiles = n // tile_size
    ks = jnp.reshape(keys[:num_full_tiles * tile_size], (num_full_tiles, tile_size, d_k))
    vs = jnp.reshape(values[:num_full_tiles * tile_size], (num_full_tiles, tile_size, d_k))

    def bodyfn(carry, scan_input):
        k_chunk, v_chunk, start_idx = scan_input
        F, L, M = carry

        logits = jnp.matmul(queries, k_chunk.T) / jnp.sqrt(head_dim)  # (m, tile_sz)

        # Causal masking
        q_positions = jnp.arange(m)[:, None]  # (m, 1)
        k_positions = jnp.arange(tile_size)[None, :] + start_idx  # (1, tile_sz)

        if pre:
            causal_mask = k_positions > q_positions  # (m, tile_sz)
        else:
            causal_mask = k_positions > (position_offset + 1)  # (m, tile_sz)

        logits = jnp.where(causal_mask, -jnp.inf, logits)

        # Softmax
        max_logits = jnp.max(logits, axis=-1, keepdims=True)
        new_max = jnp.maximum(M, max_logits)
        exp_logits = jnp.exp(logits - new_max)
        old_shift = jnp.exp(M - new_max)
        L = L * old_shift + jnp.sum(exp_logits, axis=-1, keepdims=True)
        F = F * old_shift + jnp.matmul(exp_logits, v_chunk)
        M = new_max

        return (F, L, M), None

    # Pass start indices with chunks
    start_indices = jnp.arange(0, num_full_tiles * tile_size, tile_size)
    scan_inputs = (ks, vs, start_indices)

    (F, L, _), _ = jax.lax.scan(bodyfn, init, scan_inputs)
    #return (F / L).astype(dtype)
    return jnp.where(L != 0, F / L, 0.0)

def att_head_orig(queries, keys_expanded, values_expanded, pre, position_offset):
    attn_scores = jnp.matmul(queries, keys_expanded.transpose(1,0)) / jnp.sqrt(128)

    if pre:
        q_len, k_len = queries.shape[0], keys_expanded.shape[0]
        causal_mask = jnp.triu(jnp.ones((q_len, k_len)), k=1)
        attn_scores = jnp.where(causal_mask, -jnp.inf, attn_scores)
    else:
        mask = np.arange(keys_expanded.shape[0]) > position_offset + 1
        attn_scores = jnp.where(mask, -jnp.inf, attn_scores)

    attn_weights = jax.nn.softmax(attn_scores, axis=-1)
    context = jnp.matmul(attn_weights, values_expanded)
    return context

import jax.random as random

# Config
cfg = {"head_dim": 128}
tile_size = 1024

for i  in range(10):
    # Random inputs
    key = random.PRNGKey(i)
    #m, n = 8192, 8192  # query_len, key/value_len
    m, n = 8192, 8192  # query_len, key/value_len
    d_k = cfg["head_dim"]

    #dtype = jax.dtypes.bfloat16
    #dtype = jnp.float16
    dtype = jnp.float32
    #dtype = jnp.float64

    queries = random.uniform(key, (m, d_k),dtype=dtype)
    keys = random.uniform(key, (n, d_k),dtype=dtype)
    values = random.uniform(key, (n, d_k),dtype=dtype)

    # queries = jnp.zeros((m, d_k),dtype=dtype)
    # keys = jnp.zeros((n, d_k),dtype=dtype)
    # values = jnp.zeros((n, d_k),dtype=dtype)

    for position_offset in range(10):
        # Run original attention
        context_orig = att_head_orig(queries, keys, values, pre=True, position_offset=0)

        # Run tiled attention
        #context_tiled = tiled_attention_cleaner(queries, keys, values, tile_size=tile_size, pre=True, position_offset=0)
        context_tiled = att_head_coder_causal(queries, keys, values, pre=True, position_offset=0)# tile_size=tile_size, pre=True, position_offset=0)

        # Compare
        diff = jnp.abs(context_orig - context_tiled).max()
        print(f"Max absolute difference: {diff}")
        print(jnp.allclose(context_orig,context_tiled))

    #position_offset = 64
    for position_offset in range(10):
        context_orig_gen = att_head_orig(queries, keys, values, pre=False, position_offset=position_offset)
        #context_tiled_gen = tiled_attention_cleaner(queries, keys, values, tile_size=tile_size, pre=False, position_offset=position_offset)
        context_tiled_gen = att_head_coder_causal(queries, keys, values, pre=False, position_offset=position_offset)# tile_size=tile_size, pre=True, position_offset=0)

        diff = jnp.abs(context_orig_gen - context_tiled_gen).max()

        print(f"Max absolute difference: {diff}")
        print(jnp.allclose(context_orig_gen,context_tiled_gen))
