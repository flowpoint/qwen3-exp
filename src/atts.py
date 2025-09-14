
def att_head3(queries, keys, values, pre, position_offset, tile_size=1024):
    m, d_k = queries.shape
    n, _ = keys.shape
    head_dim = d_k

    # Initialize accumulators
    F = jnp.zeros((m, values.shape[-1]))   # weighted sum
    L = jnp.zeros((m, 1))                  # normalizer (sum of exp logits)
    M = jnp.full((m, 1), -jnp.inf)         # running max

    # Loop over key/value tiles
    for j in range(0, n, tile_size):
        k_chunk = keys[j:j+tile_size]      # (tile_sz, d_k)
        v_chunk = values[j:j+tile_size]    # (tile_sz, d_v)

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

    # Final normalization
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
