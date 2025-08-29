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
