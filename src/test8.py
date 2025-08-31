import jax
import jax.numpy as jnp
import numpy as np

def att_head_orig(queries, keys_expanded, values_expanded, pre, position_offset, head_dim):
    # Untiled reference attention
    attn_scores = jnp.matmul(queries, keys_expanded.T) / jnp.sqrt(head_dim)

    if pre:
        q_len, k_len = queries.shape[0], keys_expanded.shape[0]
        causal_mask = jnp.triu(jnp.ones((q_len, k_len), dtype=bool), k=1)
        attn_scores = jnp.where(causal_mask, -jnp.inf, attn_scores)
    else:
        # Generation mask: mask out positions > position_offset + 1
        mask = jnp.arange(keys_expanded.shape[0]) > (position_offset + 1)
        attn_scores = jnp.where(mask[None, :], -jnp.inf, attn_scores)

    attn_weights = jax.nn.softmax(attn_scores, axis=-1)
    context = jnp.matmul(attn_weights, values_expanded)
    return context

def tiled_attention_causal(queries, keys, values, pre=True, position_offset=0, tile_size=1024):
    """
    Tiled attention with causal masking for both prefill (triangular) and generation (prefix).
    Numerically stable online softmax accumulation across tiles.

    queries: (m, d_k)
    keys:    (n, d_k)
    values:  (n, d_v)
    """
    m, d_k = queries.shape
    n, d_k2 = keys.shape
    assert d_k == d_k2, "queries and keys must have the same head dimension"
    d_v = values.shape[-1]
    head_dim = d_k

    F = jnp.zeros((m, d_v), dtype=queries.dtype)     # weighted sum accumulator
    L = jnp.zeros((m, 1), dtype=queries.dtype)       # normalizer accumulator
    M = jnp.full((m, 1), -jnp.inf, dtype=queries.dtype)  # running max accumulator

    # Initialize accumulators
    Fi = jnp.zeros((m, values.shape[-1]))   # weighted sum
    Li = jnp.zeros((m, 1))                  # normalizer (sum of exp logits)
    Mi = jnp.full((m, 1), -jnp.inf)         # running max

    init = ( Fi, Li, Mi )
    ks = jnp.reshape(keys, (m//tile_size, tile_size, d_k))
    vs = jnp.reshape(values, (m//tile_size, tile_size, d_k))

    masks = []
    for k_start in range(0, n, tile_size):
        if pre:
            # prefill: mask keys j > i for each query index i (triangular)
            q_idx = jnp.arange(m)[:, None]                           # (m, 1)
            k_idx = (k_start + jnp.arange(tile_size))[None, :]        # (1, tile_len)
            mask = k_idx > q_idx                                     # (m, tile_len) boolean
            masks.append(mask)
        else:
            # generation: mask keys j > position_offset + 1, independent of i
            thresh = position_offset + 1
            k_idx = (k_start + jnp.arange(tile_size))[None, :]        # (1, tile_len)
            mask = k_idx > thresh
            mask = jnp.broadcast_to(mask, [tile_size, tile_size]) #logits.shape)              # (m, tile_len)
            masks.append(mask)

    xs = (ks,vs, masks)

    # Iterate over tiles of keys/values
    def bodyfn(carry, x):
        k_chunk, v_chunk, mask = x
        F, L, M = carry

        # logits: (m, tile_len)
        logits = (queries @ k_chunk.T) / jnp.sqrt(head_dim)

        masked_logits = jnp.where(mask, -jnp.inf, logits)

        # Online softmax merge with numerical stability and masking safety
        max_logits_tile = jnp.max(masked_logits, axis=-1, keepdims=True)  # -inf if all masked
        new_M = jnp.maximum(M, max_logits_tile)

        # When new_M == -inf for a row, set alpha/beta to 0 to avoid NaNs
        is_finite_new = jnp.isfinite(new_M)
        alpha = jnp.where(is_finite_new, jnp.exp(M - new_M), 0.0)
        beta = jnp.where(jnp.isfinite(masked_logits), jnp.exp(masked_logits - new_M), 0.0)

        L = L * alpha + jnp.sum(beta, axis=-1, keepdims=True)        # (m,1)
        F = F * alpha + beta @ v_chunk                               # (m,d_v)
        M = new_M
        return (F,L,M), _

    # Final context
    (F,L,_), _ = jax.lax.scan(bodyfn, init, xs)
    context = F / L
    return context

def att_head_scan_nocausal(queries, keys, values, pre, position_offset, tile_size=32):
    m, d_k = queries.shape
    n, _ = keys.shape
    head_dim = d_k

    # Initialize accumulators
    Fi = jnp.zeros((m, values.shape[-1]))   # weighted sum
    Li = jnp.zeros((m, 1))                  # normalizer (sum of exp logits)
    Mi = jnp.full((m, 1), -jnp.inf)         # running max

    init = ( Fi, Li, Mi )
    ks = jnp.reshape(keys, (m//tile_size, tile_size, d_k))
    vs = jnp.reshape(values, (m//tile_size, tile_size, d_k))
    xs = (ks,vs)

    def bodyfn(carry, x):
        k_chunk, v_chunk = x
        F, L, M = carry

        # Compute attention logits for this tile: (m, tile_sz)
        logits = jnp.matmul(queries, k_chunk.T) / jnp.sqrt(head_dim)

        # Numerically stable online softmax
        max_logits = jnp.max(logits, axis=-1, keepdims=True)  # (m, 1)
        new_max = jnp.maximum(M, max_logits)

        # Compute exp of shifted logits
        exp_logits = jnp.exp(logits - new_max)

        # Update normalizer and weighted sum
        old_shift = jnp.exp(M - new_max)
        L = L * old_shift + jnp.sum(exp_logits, axis=-1, keepdims=True)
        F = F * old_shift + jnp.matmul(exp_logits, v_chunk)

        # Update max
        M = new_max
        return (F,L,M), _

    (F,L,_), _ = jax.lax.scan(bodyfn, init, xs)
    context = F / L
    return context

def tiled_attention_causal_scan(queries, keys, values, pre=True, position_offset=0, tile_size=1024):
    """
    Tiled attention with causal masking for both prefill (triangular) and generation (prefix).
    Numerically stable online softmax accumulation across tiles.

    queries: (m, d_k)
    keys:    (n, d_k)
    values:  (n, d_v)
    """
    m, d_k = queries.shape
    n, d_k2 = keys.shape
    assert d_k == d_k2, "queries and keys must have the same head dimension"
    d_v = values.shape[-1]
    head_dim = d_k

    F = jnp.zeros((m, d_v), dtype=queries.dtype)     # weighted sum accumulator
    L = jnp.zeros((m, 1), dtype=queries.dtype)       # normalizer accumulator
    M = jnp.full((m, 1), -jnp.inf, dtype=queries.dtype)  # running max accumulator

    # Iterate over tiles of keys/values
    for k_start in range(0, n, tile_size):
        k_end = min(k_start + tile_size, n)
        k_chunk = keys[k_start:k_end]     # (tile_len, d_k)
        v_chunk = values[k_start:k_end]   # (tile_len, d_v)
        tile_len = k_end - k_start

        # logits: (m, tile_len)
        logits = (queries @ k_chunk.T) / jnp.sqrt(head_dim)

        # Build causal mask for this tile
        if pre:
            # prefill: mask keys j > i for each query index i (triangular)
            q_idx = jnp.arange(m)[:, None]                           # (m, 1)
            k_idx = (k_start + jnp.arange(tile_len))[None, :]        # (1, tile_len)
            mask = k_idx > q_idx                                     # (m, tile_len) boolean
        else:
            # generation: mask keys j > position_offset + 1, independent of i
            thresh = position_offset + 1
            k_idx = (k_start + jnp.arange(tile_len))[None, :]        # (1, tile_len)
            mask = k_idx > thresh
            mask = jnp.broadcast_to(mask, logits.shape)              # (m, tile_len)

        masked_logits = jnp.where(mask, -jnp.inf, logits)

        # Online softmax merge with numerical stability and masking safety
        max_logits_tile = jnp.max(masked_logits, axis=-1, keepdims=True)  # -inf if all masked
        new_M = jnp.maximum(M, max_logits_tile)

        # When new_M == -inf for a row, set alpha/beta to 0 to avoid NaNs
        is_finite_new = jnp.isfinite(new_M)
        alpha = jnp.where(is_finite_new, jnp.exp(M - new_M), 0.0)
        beta = jnp.where(jnp.isfinite(masked_logits), jnp.exp(masked_logits - new_M), 0.0)

        L = L * alpha + jnp.sum(beta, axis=-1, keepdims=True)        # (m,1)
        F = F * alpha + beta @ v_chunk                               # (m,d_v)
        M = new_M

    # Final context
    context = F / L
    return context


def compare_once(rng_key, m, n, d, tile_size, pre=True):
    k1, k2, k3 = jax.random.split(rng_key, 3)
    queries = jax.random.normal(k1, (m, d), dtype=jnp.float32)
    keys    = jax.random.normal(k2, (n, d), dtype=jnp.float32)
    values  = jax.random.normal(k3, (n, d), dtype=jnp.float32)

    # For prefill, triangular mask assumes m <= n or m == n is typical. We'll set m == n for fairness.
    position_offset = int(n // 2)  # arbitrary for generation case

    if pre:
        assert m == n, "For prefill test, set m == n to match triangular mask semantics."
        ref = att_head_orig(queries, keys, values, pre=True, position_offset=0, head_dim=d)
        tiled = tiled_attention_causal(queries, keys, values, pre=True, position_offset=0, tile_size=tile_size)
    else:
        # Generation typically uses a single query, but we keep it general.
        ref = att_head_orig(queries, keys, values, pre=False, position_offset=position_offset, head_dim=d)
        tiled = tiled_attention_causal(queries, keys, values, pre=False, position_offset=position_offset, tile_size=tile_size)

    max_abs_diff = jnp.max(jnp.abs(ref - tiled))
    l2_diff = jnp.linalg.norm((ref - tiled).reshape(-1))
    return float(max_abs_diff), float(l2_diff)


    # Settings
    d = 64
    tests = [
        # (m, n, tile_size, pre)
        (128, 128, 32, True),     # prefill, triangular causal mask
        (256, 256, 64, True),     # prefill with different tile
        (1,   257, 64, False),    # generation, single query, non-divisible tile
        (4,   301, 50, False),    # generation, multiple queries
    ]

    for i, (m, n, tile_size, pre) in enumerate(tests, 1):
        if pre and m != n:
            # Ensure prefill uses m == n
            m = n
        max_abs, l2 = compare_once(rng, m, n, d, tile_size, pre=pre)
        mode = "prefill" if pre else "generation"
        print(f"Test {i} ({mode}): m={m}, n={n}, d={d}, tile_size={tile_size}")
        print(f"  max |diff| = {max_abs:.6e},  L2 diff = {l2:.6e}")

if name == "main":
    main()
