#attn_scores = jnp.einsum('bnqh,bnkh->bnqk', queries, keys_expanded) / jnp.sqrt(head_dim)


import jax
import jax.numpy as jnp
import time

# Compile and benchmark the attention score computation
@jax.jit
def compute_attn_scores(queries, keys_expanded, head_dim):
    return jnp.einsum('bnqh,bnkh->bnqk', queries, keys_expanded) / jnp.sqrt(head_dim)

def benchmark_attn_scores():
    # Test with specified input shape
    batch_size, num_heads, seq_len, head_dim = 1, 16, 40960, 128
    key = jax.random.key(0)
    queries = jax.random.uniform(key, shape=(batch_size, num_heads, seq_len, head_dim))
    keys_expanded = jax.random.uniform(key, shape=(batch_size, num_heads, seq_len, head_dim))
    
    # Warmup
    _ = compute_attn_scores(queries, keys_expanded, head_dim)
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(10):
        result = compute_attn_scores(queries, keys_expanded, head_dim)
    end = time.perf_counter()
    
    avg_time = (end - start) / 10 * 1000  # ms per call
    print(f"Input shape: {queries.shape}")
    print(f"Output shape: {result.shape}")
    print(f"Average time: {avg_time:.4f} ms")
    print(f"Memory usage: {result.nbytes / (1024**2):.2f} MB")

if __name__ == "__main__":
    benchmark_attn_scores()
