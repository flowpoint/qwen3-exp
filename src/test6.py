import jax
import jax.numpy as jnp
from pdb import set_trace

def att_head_o(queries, keys_expanded, values_expanded):
    attn_scores = jnp.matmul(queries, keys_expanded.transpose(1,0)) / jnp.sqrt(head_dim)
    attn_weights = jax.nn.softmax(attn_scores, axis=-1)
    context = jnp.matmul(attn_weights.transpose(1,0), values_expanded)
    return context

def tiled_attention2(queries, keys, values, tile_size=1024):
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


def tiled_attention(queries, keys, values, tile_size=1024):
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
        set_trace()
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


# Full attention (reference)
def full_attention(Q, K, V):
    logits = jnp.matmul(Q, K.T) / jnp.sqrt(Q.shape[-1])
    weights = jax.nn.softmax(logits, axis=-1)
    return jnp.matmul(weights, V)

# Test
s = 8192
head_dim = 128
key = jax.random.PRNGKey(0)
Q = jax.random.normal(key, (s, head_dim))
K = jax.random.normal(key, (s, head_dim))
V = jax.random.normal(key, (s, head_dim))

out_full = full_attention(Q, K, V)
#out_full = att_head_o(Q, K, V)#, tile_size=1024)
out_tiled = tiled_attention(Q, K, V, tile_size=1024)
#out_tiled = att_head_o(Q, K, V)#, tile_size=1024)

print("Max diff:", jnp.max(jnp.abs(out_full - out_tiled)))
