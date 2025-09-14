
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

from qwen3 import *
from model import *

QWEN3_CONFIG = cfg
context_size = cfg['context_length']

HF_REPO_ID = "Qwen/Qwen3-0.6B"

model_path = download_model_from_hf(HF_REPO_ID)
safetensors_files = list(Path(model_path).glob("*.safetensors"))
safetensors_files.sort()

tokenizer_path = model_path / "tokenizer.json"
tokenizer = Qwen3Tokenizer(str(tokenizer_path) if tokenizer_path.exists() else "tokenizer.json", repo_id=HF_REPO_ID)

#pref_mul = 20_000
pref_mul = 1
prompt = "Give me a short introduction to large language models."*pref_mul
input_ids = tokenizer.encode(prompt)
if len(input_ids) > QWEN3_CONFIG["context_length"]:
    input_ids = input_ids[:QWEN3_CONFIG["context_length"]]

# Keep input on device from start
input_token_ids = jnp.array(input_ids)

cfg = QWEN3_CONFIG
key = jax.random.PRNGKey(0)
params = init_qwen3_params(key, cfg)
params = load_qwen3_weights_jax_optimized(cfg, params, safetensors_files)
#import pickle
#pickle.dumps(params, 'params.pickle')
model = {"params": params, "cfg": cfg}

params, cfg = model["params"], model["cfg"]
cfg.pop('dtype')
import operator

# Keep input on device
cur_ids = jnp.array([input_ids])
key = jax.random.PRNGKey(42)

# Initialize KV cache for batch processing
n_layers = cfg['n_layers']
n_kv_groups = cfg['n_kv_groups']
head_dim = cfg['head_dim']
kv_cache = {"keys": jnp.zeros((1, n_layers, n_kv_groups, context_size, head_dim), dtype=dtype), 
             "values": jnp.zeros((1, n_layers, n_kv_groups, context_size, head_dim),dtype=dtype)} 
position_offset = 0

# prefill1
logits2, kv_cache2, position_offset2 = qwen3_forward_kv(params, cur_ids, cfg, kv_cache, position_offset,pre=True)

'''
logits, kv_cache, position_offset = qwen3_forward_kv(params, cur_ids, cfg, kv_cache, position_offset,pre=False)
#for i in range(cur_ids.shape[1]):
#logits, kv_cache, position_offset = qwen3_forward_kv(params, cur_ids, cfg, kv_cache, position_offset)
print('---')
print(kv_cache2 == kv_cache)
print(logits2 == logits)
print(position_offset2 == position_offset)
'''

'''
x = cur_ids

a = qwen3_forward_kv_pre_unchunk(params, x, cfg, kv_cache, position_offset)
print(a)
b = qwen3_forward_kv_pre(params, x, cfg, kv_cache, position_offset)
print(b)

'''
max_seq = 40960
#max_seq = 26
#max_seq = 1024


num_heads = cfg['n_heads']
num_kv_groups = cfg['n_kv_groups']
head_dim = cfg['head_dim']
cos, sin = params['cos'], params['sin']
qk_norm = True

x = jnp.ones([1,max_seq],dtype=jnp.int64)
x = params["tok_emb"][x]

attn_params = params['trf_blocks'][0]['att']
print('-----')

layer_cache = {"keys":kv_cache['keys'][:,0], "values":kv_cache["values"][:,0]}

out1 = grouped_query_attention_forward_kv_pre(num_heads, num_kv_groups, head_dim, cos, sin, attn_params, layer_cache, qk_norm, position_offset, x)

print('pass1')

out2 = grouped_query_attention_forward_kv(num_heads, num_kv_groups, head_dim, cos, sin, attn_params, layer_cache, qk_norm, position_offset, x)

'''
#jnp.all(out1 == out2)

#grouped_query_attention_forward_kv(cfg["n_heads"], cfg["n_kv_groups"], cfg["head_dim"], cos, sin, params["att"], kv_cache, cfg["qk_norm"], position_offset, x)


keys_expanded = jnp.ones([1, 16, max_seq, 128])
queries = jnp.ones([1, 16, max_seq, 128])

print(keys_expanded.shape)
attn_scores = jnp.einsum('bnqh,bnkh->bnqk', queries, keys_expanded) / jnp.sqrt(head_dim)

#print(params['out_proj'])
'''

'''
if position_offset == 0:
    q_len, k_len = queries.shape[2], keys.shape[2]
    causal_mask = jnp.triu(jnp.ones((q_len, k_len)), k=1)
    attn_scores = jnp.where(causal_mask[None, None, :, :], -jnp.inf, attn_scores)

attn_weights = jax.nn.softmax(attn_scores, axis=-1)
context = jnp.einsum('bnqk,bnkh->bnqh', attn_weights, values_expanded)
context = context.transpose(0,2,1,3).reshape(b, seq, num_heads * head_dim)
output = jnp.einsum('bsh,hd->bsd', context, params["out_proj"])

return output, new_cache, position_offset_new
'''

import jax
import jax.numpy as jnp
from timeit import timeit

from pdb import set_trace
from functools import partial
from tqdm import tqdm

max_seq = 40960
#max_seq = 1024*8
print(max_seq)

bs = 1
n_heads = 16

head_dim = 128

keys_expanded = jnp.ones([n_heads, max_seq, head_dim])
values_expanded = jnp.ones([n_heads, max_seq, head_dim])
queries = jnp.ones([n_heads, max_seq, head_dim])

out_proj = jnp.ones([2048, 1024])


import os
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_triton_gemm_any=True '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
)

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.95'
#os.environ['JAX_ENABLE_COMPILATION_CACHE'] = 'false'
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['JAX_PLATFORMS'] = 'gpu'
os.environ['JAX_DEFAULT_MATMUL_PRECISION'] = 'highest'
os.environ['JAX_ENABLE_X64'] = 'True'

'''
print(keys_expanded.shape)
#@jax.jit
def a(keys_expanded, queries):
    head_dim = 128
    chunk_size = 1024
    #for bi in batch_size
    #for hi in range(hum_heads)
    attn_scores = jnp.einsum('qh,kh->qk', queries, keys_expanded)
    return attn_scores

attn_scores = jnp.einsum('nqh,nkh->nqk', queries, keys_expanded)
#attn_scores2 = jnp.einsum('qh,kh->qk', queries, keys_expanded)
#attn_scores2 = jnp.matmul(queries, keys_expanded.transpose(1,0))

#print(jnp.all(attn_scores == attn_scores2))

#attn_scores = a(keys_expanded, queries)
#attn_scores.block_until_ready()
'''

def chunked_vmap(fn, *args, chunk_size):
    leading_dim = args[0].shape[0]
    assert all(arg.shape[0] == leading_dim for arg in args), "All inputs must have the same leading dimension."

    outputs = []

    for start in range(0, leading_dim, chunk_size):
        end = min(start + chunk_size, leading_dim)
        chunked_args = [arg[start:end] for arg in args]
        chunk_out = jax.vmap(fn)(*chunked_args)
        outputs.append(chunk_out)

    return jnp.concatenate(outputs, axis=0)

def chunked_vmap2(fn, inputs1, inputs2, chunk_size):
    num_chunks = inputs1.shape[0] // chunk_size
    chunks = []
    for i in range(num_chunks):
        q_chunk = inputs1[i*chunk_size:(i+1)*chunk_size]
        k_chunk = inputs2[i*chunk_size:(i+1)*chunk_size]
        chunks.append(jax.vmap(fn)(q_chunk, k_chunk))
    # Handle remainder if any
    if inputs1.shape[0] % chunk_size != 0:
        q_chunk = inputs1[num_chunks*chunk_size:]
        k_chunk = inputs2[num_chunks*chunk_size:]
        chunks.append(jax.vmap(fn)(q_chunk, k_chunk))
    return jnp.concatenate(chunks, axis=0)

def manual_vmap(fn, *args):
	dim1 = args[0].shape[0]
	outputs = []
	for i in range(dim1):
		inputs_i = [arg[i] for arg in args]  # Grab the i-th slice from each arg
		out = fn(*inputs_i)
		outputs.append(out)

	return jnp.stack(outputs)

'''
def tiled_matmul(a,b):
    chunk_size = 1024
    chunks = a.shape[0] // chunk_size
    res = jnp.zeros([a.shape[0], b.shape[1]])

    for i in range(0,chunks+1):
        for k in range(0,chunks+1):
            for j in range(0,chunks+1):
                x = a[i*chunk_size:(i+1)*chunk_size]
                y = b[:, i*chunk_size:(i+1)*chunk_size]
                res = res.at[i*chunk_size:(i+1)*chunk_size, k*chunk_size:(k+1)*chunk_size].add( jnp.matmul(x,y) )
                '''

def tiled_matmul(A, B, tile_size=1024):
    m, k = A.shape
    k, n = B.shape
    C = jnp.zeros((m, n))
    for i in range(0, m, tile_size):
        for k0 in range(0, k, tile_size):
            for j in range(0, n, tile_size):
                a = A[i:i+tile_size, k0:k0+tile_size]
                b = B[k0:k0+tile_size, j:j+tile_size]
                C = C.at[i:i+tile_size, j:j+tile_size].add(jnp.matmul(a, b))
    return C

@jax.jit
def tiled_matmul_scan(A, B):
    #m, k = A.shape
    #k, n = B.shape
    tile_size = 1024
    m, k = 40960, 128
    k, n = 128, 40960
    k0s = jnp.arange(0, k, tile_size)
    C = jnp.zeros((m, n))

    def update_C(C, k0):
        A_block = A[:, k0:k0 + tile_size]
        B_block = B[k0:k0 + tile_size, :]
        return C + A_block @ B_block, None

    return jax.lax.scan(update_C, C, k0s)[0]

def softmax(mat):
    e = jnp.exp(mat)
    return e / jnp.sum(e, axis=-1)

#@jax.jit
def att_head_o(queries, keys_expanded, values_expanded):
    #set_trace()
    #attn_scores = jnp.einsum('qh,kh->qk', queries, keys_expanded) / jnp.sqrt(head_dim)
    #set_trace()
    if True:
        ket = keys_expanded.transpose(1,0)
        res = tiled_matmul(queries,ket)
        attn_scores = res / jnp.sqrt(head_dim)
        attn_weights = jax.nn.softmax(attn_scores, axis=-1)
        #attn_weights = softmax(attn_scores)
        context = tiled_matmul(attn_weights.transpose(1,0), values_expanded)
    else:
        attn_scores = jnp.matmul(queries, keys_expanded.transpose(1,0)) / jnp.sqrt(head_dim)
        #attn_scores2 = jnp.matmul(queries, keys_expanded.transpose(1,0)) / jnp.sqrt(head_dim)
        #set_trace()
        #assert jnp.all(attn_scores == attn_scores2)
        
        attn_weights = jax.nn.softmax(attn_scores, axis=-1)
        #context = jnp.einsum('qk,kh->qh', attn_weights, values_expanded)
        context = jnp.matmul(attn_weights.transpose(1,0), values_expanded)
    return context

#@jax.jit
def att_head_g(queries, keys_expanded, values_expanded, simp=False):
    tile_size = 1024
    # matmul1
    A = queries
    B = keys_expanded.transpose(1,0)


    m, k = A.shape
    k, n = B.shape
    C = jnp.zeros((m, n))
    # can actually skip tiling over the small 128 (head dimension)
    for i in range(0, m, tile_size):
        for j in range(0, n, tile_size):
            a = A[i:i+tile_size]#, k0:k0+tile_size]
            b = B[:, j:j+tile_size]
            C = C.at[i:i+tile_size, j:j+tile_size].add(jnp.matmul(a, b))
            #print((i,j))
    res = C
    assert jnp.allclose(C, tiled_matmul(queries, B))
    # 40960, 40960

    attn_scores = res / jnp.sqrt(head_dim)
    attn_weights = jax.nn.softmax(attn_scores, axis=-1)

    # softmax
    #set_trace()
    '''
    e = jnp.exp(attn_scores)
    attn_weights = e / jnp.sum(e, axis=-1)
    assert jnp.allclose(attn_weights, jax.nn.softmax(attn_scores, axis=-1))
    '''

    # matmul2
    D = attn_weights.transpose(1,0)
    E = values_expanded

    '''
    set_trace()
    assert m == D.shape[0]
    assert k == D.shape[1]
    assert n == E.shape[1]
    '''

    m, k = D.shape
    k, n = E.shape
    F = jnp.zeros((m, n))

    for i in range(0, m, tile_size):
        for k0 in range(0, k, tile_size):
            a = D[i:i+tile_size, k0:k0+tile_size]
            b = E[k0:k0+tile_size]
            F = F.at[i:i+tile_size].add(jnp.matmul(a, b))

    context = F
    return context

import jax
import jax.numpy as jnp

def att_head_vibe1(queries, keys, values, tile_size=1024):
    # qwen3-coder 
    m, d_k = queries.shape
    n, _ = keys.shape
    head_dim = d_k

    # Initialize accumulators
    F = jnp.zeros((m, values.shape[-1]))  # weighted sum (numerator)
    L = jnp.zeros((m, 1))                 # normalizer (denominator)
    M = jnp.full((m, 1), -jnp.inf)        # max per row

    for j in range(0, n, tile_size):
        k_chunk = keys[j:j+tile_size]       # (tile_size, d_k)
        v_chunk = values[j:j+tile_size]     # (tile_size, d_v)

        # Compute logits for this tile: (m, tile_size)
        logits = jnp.matmul(queries, k_chunk.transpose(1, 0)) / jnp.sqrt(head_dim)

        # Track max for numerical stability
        max_logits = jnp.max(logits, axis=-1, keepdims=True)  # (m, 1)
        new_max = jnp.maximum(M, max_logits)

        # Compute exp of shifted logits
        exp_logits = jnp.exp(logits - new_max)  # (m, tile_size)

        # Update normalizer (sum of exp logits)
        L = L * jnp.exp(M - new_max) + jnp.sum(exp_logits, axis=-1, keepdims=True)

        # Update weighted values
        F = F * jnp.exp(M - new_max) + jnp.matmul(exp_logits, v_chunk)

        # Update max
        M = new_max

    # Normalize to get final context
    context = F / L
    return context

def att_head(queries, keys, values, tile_size=1024):
    set_trace()
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


def att_head_i(queries, keys_expanded, values_expanded):
    tile_size = 1024
    # matmul1
    A = queries
    B = keys_expanded.transpose(1,0)

    mm_ref = jnp.matmul(queries, keys_expanded.transpose(1,0)) 
    mm_ref2 = tiled_matmul(queries, keys_expanded.transpose(1,0)) 

    attn_scores = jnp.matmul(queries, keys_expanded.transpose(1,0)) / jnp.sqrt(head_dim)
    attn_scores_ref = attn_scores
    attn_weights_ref = jax.nn.softmax(attn_scores, axis=-1)
    context_ref = jnp.matmul(attn_weights_ref.transpose(1,0), values_expanded)

    m, k = A.shape
    k, n = B.shape
    #C = jnp.zeros((m, n))
    #C = jnp.zeros((m, 128))
    # square in seq dims
    F = jnp.zeros((m, values_expanded.shape[-1]))
    R = jnp.zeros((m, values_expanded.shape[-1]))

    # can actually skip tiling over the small 128 (head dimension)
    for i in range(0, m, tile_size):
        k, n = B.shape
        for j in range(0, n, tile_size):
            a = A[i:i+tile_size]#, k0:k0+tile_size]
            b = B[:, j:j+tile_size]
            #C = C.at[i:i+tile_size, j:j+tile_size].add(jnp.matmul(a, b))

            res_ij = jnp.matmul(a,b)
            #res_ij = C[i:i+tile_size,j:j+tile_size]

            # normalize by head dim
            attn_scores_ij = res_ij / jnp.sqrt(head_dim)
            #attn_weights = jax.nn.softmax(attn_scores, axis=-1)

            #if not jnp.allclose(mm_ref2[i:i+tile_size], mm_ref[i:i+tile_size]): set_trace()
            #if not jnp.allclose(C[i:i+tile_size, j:j+tile_size], mm_ref[i:i+tile_size, j:j+tile_size]): set_trace()

            # normalize by head dim
            #if not jnp.allclose(attn_scores_ij, attn_scores_ref[i:i+tile_size, j:j+tile_size]): set_trace()

            #attn_weights_ij = jax.nn.softmax(attn_scores_ij, axis=-1)
            attn_weights_ij = attn_scores_ij
            #if not jnp.allclose(attn_weights_ij, attn_weights_ref[i:i+tile_size,j:j+tile_size]): set_trace()

            # matmul2
            # in tiling/fusing, the transpose seems to automatically fall away
            D = attn_weights_ij#.transpose(1,0)
            E = values_expanded

            #m, k = D.shape
            #k, n = E.shape

            #b = E[k02:k02+tile_size]#, j2:j2+tile_size]
            #set_trace()
            #F = F.at[i:i+tile_size, j2:j2+tile_size].add(jnp.matmul(a, b))
            #F = F.at[i:i+tile_size, j2:j2+tile_size].add(jnp.matmul(a, b))
            #F = F.at[k02:k02+tile_size].add(jnp.matmul(a, b))

            #assert jnp.allclose(F[i:i+tile_size], tiled_matmul(attn_weights.transpose(1,0), values_expanded)[i:i+tile_size])
            #assert jnp.allclose(F, tiled_matmul(attn_weights.transpose(1,0), values_expanded))

            #set_trace()
            #for k0 in range(0, n, tile_size):
                #F = F.at[k0:k0+tile_size].add(jnp.matmul(a, b))
            #k0 = j
            a = D#[i:i+tile_size]#, k0:k0+tile_size]
            b = E[j:j+tile_size]
            #print((i,j))
            v = jnp.matmul(a,b)
            F = F.at[i:i+tile_size].add(v)

    #context = F #/ jnp.sqrt(head_dim)
    context = jax.nn.softmax(F, axis=-2)  #/ jnp.sqrt(head_dim)
    return context


#partial(jax.jit, static_argnums=[2])
def att2(queries, keys_expanded, chunk_size=1):
    #context = jax.vmap(att_head, in_axes=(0,0,0))(queries, keys_expanded, values_expanded)
    #context = chunked_vmap(att_head, queries, keys_expanded, values_expanded, chunk_size=chunk_size)
    context = manual_vmap(att_head, queries, keys_expanded, values_expanded)

    context = context.transpose(1,0,2).reshape(max_seq, n_heads * head_dim)
    output = jnp.einsum('sh,hd->sd', context, out_proj)
    return output


def att(queries, keys_expanded):
    attn_scores = jnp.einsum('nqh,nkh->nqk', queries, keys_expanded) / jnp.sqrt(head_dim)
    
    '''
    if position_offset == 0:
        q_len, k_len = queries.shape[2], keys.shape[2]
        causal_mask = jnp.triu(jnp.ones((q_len, k_len)), k=1)
        attn_scores = jnp.where(causal_mask[None, None, :, :], -jnp.inf, attn_scores)
    '''
    
    attn_weights = jax.nn.softmax(attn_scores, axis=-1)
    context = jnp.einsum('nqk,nkh->nqh', attn_weights, values_expanded)
    context = context.transpose(1,0,2).reshape(max_seq, n_heads * head_dim)
    output = jnp.einsum('sh,hd->sd', context, out_proj)
    return output


key = jax.random.PRNGKey(1)
a = jax.random.uniform(key, shape=[8192, 128])
b = jax.random.uniform(key, shape=[128, 8192])
c = jax.random.uniform(key, shape=[8192, 128])
#set_trace()
#assert jnp.all(tiled_matmul(a,b) == jnp.matmul(a,b))

queries = a
keys_expanded = a
values_expanded = c

r1 = att_head_o(queries, keys_expanded, values_expanded)
r2 = att_head_g(queries, keys_expanded, values_expanded)
r3 = att_head_g(queries, keys_expanded, values_expanded,simp=True)
r4 = att_head(queries, keys_expanded, values_expanded)
print(r3 - r4)
set_trace()
assert jnp.allclose(r1, r2)
assert jnp.allclose(r3, r4)
assert jnp.allclose(r1, r4)

#jax.clear_caches()
run_nojit = 0
if run_nojit:
    print(att2(queries, keys_expanded))#, chunk_size=i)

run_compiled = 0
if run_compiled:
    print('starting tracing')
    traced = jax.jit(att2, static_argnums=[2]).trace(queries, keys_expanded)
    lowered = traced.lower()


    print('compiling')
    compiled = lowered.compile()
    print('compiling dont, running')


    #for i in tqdm(range(1, 2)):
    def atta():
        a2 = compiled(queries, keys_expanded)#, chunk_size=i)
        a2.block_until_ready()
        print('tick')

    atta()

    tn = 10
    print(timeit(atta, number=tn)/tn)
    jax.clear_caches()

    #a1 = att(queries, keys_expanded)
    #print(jnp.all(a1 == a2))

    #print(compiled.as_text())
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

    # note, bodyfn isnt pure on inputs, it uses queries
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
from pdb import set_trace
import jax
import jax.numpy as jnp


#@jax.jit
def tiled_matmul_scan(A, B):
    m, k = A.shape
    k, n = B.shape
    tile_size = m
    #m, k = 40960, 128
    #k, n = 128, 40960
    #k0s = jnp.arange(0, k, tile_size)
    set_trace()
    C = jnp.zeros((m, n))
    num_tiles = m // tile_size
    A = jnp.reshape(A, [num_tiles, m // num_tiles])
    B = jnp.reshape(B, [tile_size, n // tile_size])
    xs = (A, B) #k0s,

    def update_C(C, x):
        A_block, B_block = x
        #A_block = A[:, k0:k0 + tile_size]
        #B_block = B[k0:k0 + tile_size, :]
        return C + A_block @ B_block, None

    return jax.lax.scan(update_C, C, k0s)[0]

#@jax.jit
def tiled_matmul_v(x, y):
    tile_size = 1024
    
    def scan_body(carry, y_tile):
        set_trace()
        acc = carry
        result_tile = jnp.einsum('se,ev->sv', x, y_tile)
        acc2 = acc + result_tile
        return acc2, None
    
    y_tiles = y.reshape(y.shape[0] // tile_size, tile_size, y.shape[1])
    init_acc = jnp.zeros((x.shape[0], y.shape[1]))
    final_result, _ = jax.lax.scan(scan_body, init_acc, y_tiles)
    return final_result

def tiled_matmul(x, y):
    tile_size = 1024
    
    def scan_body(carry, x_tile):
        acc = carry
        result_tile = jnp.matmul(x_tile, y)
        set_trace()
        acc = jnp.concatenate([acc, result_tile], axis=0)
        return acc, None
    
    x_tiles = x.reshape(x.shape[0] // tile_size, tile_size, x.shape[1])
    init_acc = jnp.zeros((0, y.shape[1]))
    final_result, _ = jax.lax.scan(scan_body, init_acc, x_tiles)
    return final_result


seqlen = 2048
x = jnp.ones([seqlen,1024])
y = jnp.ones([1024,150000])

logits = jnp.einsum('se,ev->sv', x, y)
print(logits)
print(logits.shape)
l3 = jnp.matmul(x, y)
print(jnp.allclose(logits, l3))
'''

l2 = tiled_matmul(x,y)
print(l2)
'''
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
import pytest
import jax
import jax.numpy as jnp
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
from pathlib import Path
import os
from qwen3 import Qwen3Tokenizer, download_model_from_hf, load_qwen3_weights_jax_optimized, init_qwen3_params, generate_kv_optimized, QWEN3_CONFIG

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

def test_qwen3_jax_vs_hf():
    HF_REPO_ID = "Qwen/Qwen3-0.6B"

    # Download model and tokenizer
    model_path = download_model_from_hf(HF_REPO_ID)
    safetensors_files = list(Path(model_path).glob("*.safetensors"))
    safetensors_files.sort()

    tokenizer_path = model_path / "tokenizer.json"
    tokenizer = Qwen3Tokenizer(str(tokenizer_path) if tokenizer_path.exists() else "tokenizer.json", repo_id=HF_REPO_ID)

    # Prompt for testing
    prompt = "Give me a short introduction to large language models."
    input_ids = tokenizer.encode(prompt)
    if len(input_ids) > QWEN3_CONFIG["context_length"]:
        input_ids = input_ids[:QWEN3_CONFIG["context_length"]]

    # HuggingFace model
    hf_model = AutoModelForCausalLM.from_pretrained(HF_REPO_ID, torch_dtype=torch.bfloat16).to(torch_device)
    hf_tokenizer = AutoTokenizer.from_pretrained(HF_REPO_ID)

    # Ensure same input
    hf_input_ids = torch.tensor([input_ids], dtype=torch.long).to(torch_device)

    # JAX model
    cfg = QWEN3_CONFIG
    key = jax.random.PRNGKey(0)
    params = init_qwen3_params(key, cfg)
    params = load_qwen3_weights_jax_optimized(cfg, params, safetensors_files)
    model = {"params": params, "cfg": cfg}

    # Generate with HuggingFace model
    hf_model.eval()
    with torch.no_grad():
        hf_output = hf_model.generate(hf_input_ids, max_new_tokens=50, do_sample=False, top_k=1, temperature=0.0)

    # Generate with JAX model
    jax_output = generate_kv_optimized(
        model=model, idx=jnp.array(input_ids), max_new_tokens=50,
        context_size=QWEN3_CONFIG["context_length"], top_k=1,
        temperature=0, eos_id=None
    )

    # Convert JAX output to numpy for comparison
    jax_output_np = np.array(jax_output[0])

    # Compare outputs
    a = hf_output[0].cpu().numpy()
    b = jax_output_np
    print(a.shape)
    print(b.shape)
    b = b[:a.shape[-1]]
    assert np.array_equal(a,b), "Outputs do not match"

if __name__ == "__main__":
    pytest.main([__file__])

