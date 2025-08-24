
import jax
import jax.numpy as jnp
import time

# Compile and benchmark the attention score computation with head dimension breakdown
def compute_attn_scores_chunked(queries, keys_expanded, head_dim):
    # Break down computation across head dimension
    #@jax.jit
    def compute_single_head(q, k):
        print(q.shape)
        return jnp.einsum('bqh,bkh->bqk', q, k) / jnp.sqrt(head_dim)
    
    # Process each head separately
    #heads = jax.vmap(compute_single_head, in_axes=(1, 1))(queries, keys_expanded)
    heads = [compute_single_head(queries[:,i],keys_expanded[:,i]) for i in range(queries.shape[1])]
    #ax.vmap(compute_single_head, in_axes=(1, 1))(queries, keys_expanded)
    return heads

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
    for _ in range(5):
        result = compute_attn_scores_chunked(queries, keys_expanded, head_dim)
    end = time.perf_counter()
    
    avg_time = (end - start) / 5 * 1000  # ms per call
    print(f"Input shape: {queries.shape}")
    print(f"Output shape: {result.shape}")
    print(f"Average time: {avg_time:.4f} ms")
    print(f"Memory usage: {result.nbytes / (1024**2):.2f} MB")

if __name__ == "__main__":
    benchmark_attn_scores()

