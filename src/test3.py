import jax
import jax.numpy as jnp
import time

@jax.jit
def compute_attn_scores_chunked(queries, keys_expanded, head_dim, seq_chunk_size=1024):
    """Compute attention scores with sequence length chunking"""
    batch_size, num_heads, seq_len, head_dim = queries.shape
    
    @jax.jit
    def compute_chunk(args):
        q_chunk, k_chunk = args
        return jnp.einsum('bnqh,bnkh->bnqk', q_chunk, k_chunk) / jnp.sqrt(head_dim)
    
    # Process sequence in chunks
    results = []
    for i in range(0, seq_len, seq_chunk_size):
        q_chunk = queries[:, :, i:i+seq_chunk_size, :]
        k_chunk = keys_expanded[:, :, i:i+seq_chunk_size, :]
        result_chunk = compute_chunk((q_chunk, k_chunk))
        results.append(result_chunk)
    
    # Concatenate results
    return jnp.concatenate(results, axis=3)

def benchmark_attn_scores():
    # Test with specified input shape
    batch_size, num_heads, seq_len, head_dim = 1, 16, 40960, 128
    key = jax.random.PRNGKey(0)
    queries = jax.random.normal(key, (batch_size, num_heads, seq_len, head_dim))
    keys_expanded = jax.random.normal(key, (batch_size, num_heads, seq_len, head_dim))
    
    # Warmup
    _ = compute_attn_scores_chunked(queries, keys_expanded, head_dim)
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(3):
        result = compute_attn_scores_chunked(queries, keys_expanded, head_dim)
    end = time.perf_counter()
    
    avg_time = (end - start) / 3 * 1000  # ms per call
    print(f"Input shape: {queries.shape}")
    print(f"Output shape: {result.shape}")
    print(f"Average time: {avg_time:.4f} ms")
    print(f"Memory usage: {result.nbytes / (1024**2):.2f} MB")

if __name__ == "__main__":
    benchmark_attn_scores()

