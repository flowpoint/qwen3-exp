import jax
import jax.numpy as jnp
from tokenizers import Tokenizer
#import torch
from safetensors.numpy import load_file 
import os
from pathlib import Path
import gc
from collections import defaultdict
import numpy as np 
from tqdm import tqdm
from pdb import set_trace
from timeit import timeit
import time

bp = jax.debug.breakpoint

from functools import partial, reduce

use_gpu = True
if use_gpu and jax.default_backend() == 'gpu':
    device =  jax.devices('gpu')[0]
else:
    device =  jax.devices('cpu')[0]

from model_utils import *

#dtype = jax.dtypes.bfloat16
#dtype = jnp.float16
dtype = jnp.float32
#cl = 40960
cl = 1024

cfg = {
    "vocab_size": 151936, "context_length": cl, "emb_dim": 1024, "n_heads": 16,
    "n_layers": 28, "hidden_dim": 3072, "head_dim": 128, "qk_norm": True,
    "n_kv_groups": 8, "rope_base": 1000000.0, "dtype": 'bfloat16', #torch.bfloat16,
}


'''
def manual_vmap(fn, *args):
    dim1 = args[0].shape[0]
    outputs = []
    for i in range(dim1):
        inputs_i = [arg[i] for arg in args]  # Grab the i-th slice from each arg
        out = fn(*inputs_i)
        outputs.append(out)

    return jnp.stack(outputs)

def manual_vmap(att_head, queries, keys_expanded, values_expanded, pre, position_offset):    
    xr = []
    for query, key, value in zip(queries, keys_expanded, values_expanded):
        x = att_head(query, key, value, pre, position_offset)
        xr.append(x)
    return jnp.stack(xr)
'''

def manual_vmap(att_head, queries, keys_expanded, values_expanded, pre, position_offset):
    def bodyfn(carry, x):
        queries, keys_expanded, values_expanded = x
        x = att_head(queries, keys_expanded, values_expanded, pre, position_offset)
        return carry, x
    carry, xr = jax.lax.scan(bodyfn, None, xs=(queries, keys_expanded, values_expanded), unroll=False)
    return xr



#@partial(jax.jit, static_argnums=[3])
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
    init = (Fi, Li, Mi, position_offset)

    num_full_tiles = n // tile_size
    ks = jnp.reshape(keys[:num_full_tiles * tile_size], (num_full_tiles, tile_size, d_k))
    vs = jnp.reshape(values[:num_full_tiles * tile_size], (num_full_tiles, tile_size, d_k))
    #bp()

    def causal_bodyfn(carry, scan_input):
        #set_trace()
        #bp()
        k_chunk, v_chunk, start_idx = scan_input
        F, L, M, position_offset = carry
        #jax.debug.print("pos: {0}", position_offset)
        logits = jnp.matmul(queries, k_chunk.T) / jnp.sqrt(head_dim)  # (m, tile_sz)
        k_positions = jnp.arange(tile_size)[None, :] + start_idx  # (1, tile_sz)

        '''
        if pre:
            # Causal masking
            q_positions = jnp.arange(m)[:, None]  # (m, 1)
            causal_mask = k_positions > q_positions  # (m, tile_sz)
        else:
            causal_mask = k_positions > (position_offset + 1)  # (m, tile_sz)
            '''

        causal_mask = k_positions > (position_offset + 1)  # (m, tile_sz)
        '''
        q_positions = jnp.arange(m)[:, None]  # (m, 1)
        causal_mask = k_positions > q_positions  # (m, tile_sz)
        causal_mask2 = k_positions > (position_offset + 1)  # (m, tile_sz)
        '''

        logits = jnp.where(causal_mask, -jnp.inf, logits)

        # Softmax
        max_logits = jnp.max(logits, axis=-1, keepdims=True)
        new_max = jnp.maximum(M, max_logits)
        exp_logits = jnp.exp(logits - new_max)
        old_shift = jnp.exp(M - new_max)
        L2 = L * old_shift + jnp.sum(exp_logits, axis=-1, keepdims=True)
        F2 = F * old_shift + jnp.matmul(exp_logits, v_chunk)
        M2 = new_max

        return (F2, L2, M2, position_offset), None

    def causal_bodyfn_pre(carry, scan_input):
        #set_trace()
        #bp()
        k_chunk, v_chunk, start_idx = scan_input
        F, L, M, position_offset = carry
        #jax.debug.print("pos: {0}", position_offset)
        logits = jnp.matmul(queries, k_chunk.T) / jnp.sqrt(head_dim)  # (m, tile_sz)
        k_positions = jnp.arange(tile_size)[None, :] + start_idx  # (1, tile_sz)

        # Causal masking
        q_positions = jnp.arange(m)[:, None]  # (m, 1)
        causal_mask = k_positions > q_positions  # (m, tile_sz)

        logits = jnp.where(causal_mask, -jnp.inf, logits)

        # Softmax
        max_logits = jnp.max(logits, axis=-1, keepdims=True)
        new_max = jnp.maximum(M, max_logits)
        exp_logits = jnp.exp(logits - new_max)
        old_shift = jnp.exp(M - new_max)
        L2 = L * old_shift + jnp.sum(exp_logits, axis=-1, keepdims=True)
        F2 = F * old_shift + jnp.matmul(exp_logits, v_chunk)
        M2 = new_max

        return (F2, L2, M2, position_offset), None

    # Pass start indices with chunks
    start_indices = jnp.arange(0, num_full_tiles * tile_size, tile_size)
    scan_inputs = (ks, vs, start_indices)

    if pre:
        (Fn, Ln, _, _), _ = jax.lax.scan(causal_bodyfn_pre, init, scan_inputs)
    else:
        (Fn, Ln, _, _), _ = jax.lax.scan(causal_bodyfn_pre, init, scan_inputs)
    #assert jnp.any(L != 0.)
    return (Fn / Ln).astype(dtype)
    #return jnp.where(L != 0, F / L, 0.0)


def att_head_orig(queries, keys_expanded, values_expanded, pre, position_offset):
    #attn_scores = jnp.einsum('qh,kh->qk', queries, keys_expanded) / jnp.sqrt(head_dim)
    attn_scores = jnp.matmul(queries, keys_expanded.transpose(1,0)) / jnp.sqrt(cfg['head_dim'])

    if pre:
        q_len, k_len = queries.shape[0], keys_expanded.shape[0]
        causal_mask = jnp.triu(jnp.ones((q_len, k_len)), k=1)
        attn_scores = jnp.where(causal_mask, -jnp.inf, attn_scores)

    else:
        mask = np.arange(keys_expanded.shape[0]) > position_offset + 1
        attn_scores = jnp.where(mask, -jnp.float_('inf'), attn_scores)

    #return attn_scores
    attn_weights = jax.nn.softmax(attn_scores, axis=-1)

    #context = jnp.einsum('qk,kh->qh', attn_weights, values_expanded)
    context = jnp.matmul(attn_weights, values_expanded)
    return context

#partial(jax.jit, static_argnums=[])
def att2(queries, keys_expanded, values_expanded, out_proj, pre, position_offset):
    #set_trace()
    tiled = False
    comp = False

    if tiled:
        context = manual_vmap(att_head_coder_causal, queries, keys_expanded, values_expanded, pre, position_offset)
    else:
        context = manual_vmap(att_head_orig, queries, keys_expanded, values_expanded, pre, position_offset)

    if comp:
        context = manual_vmap(att_head_orig, queries, keys_expanded, values_expanded, pre, position_offset)
        context2 = manual_vmap(att_head_coder_causal, queries, keys_expanded, values_expanded, pre, position_offset)
        if not jnp.allclose(context, context2): #jnp.any(context - context1 > 0.001):
            set_trace()

    context = context.transpose(1,0,2).reshape(queries.shape[1], cfg['n_heads'] * cfg['head_dim'])
    output = jnp.einsum('sh,hd->sd', context, out_proj)
    return output
    

#@partial(jax.jit, static_argnums=[0,1,2,7,])
def grouped_query_attention_forward_kv(num_heads, num_kv_groups, head_dim, params, kv_cache, qk_norm, position_offset, x, pre=False):
    cos, sin = compute_rope_params(cfg["head_dim"], cfg["rope_base"], cfg["context_length"])

    b, seq, d_in = x.shape
    group_size = num_heads // num_kv_groups
    
    queries = jnp.einsum('bsd,dh->bsh', x, params["W_query"]).reshape(b, seq, num_heads, head_dim).transpose(0,2,1,3)
    keys = jnp.einsum('bsd,dh->bsh', x, params["W_key"]).reshape(b, seq, num_kv_groups, head_dim).transpose(0,2,1,3)
    values = jnp.einsum('bsd,dh->bsh', x, params["W_value"]).reshape(b, seq, num_kv_groups, head_dim).transpose(0,2,1,3)
    keys = keys.astype(dtype)
    values = values.astype(dtype)

    if qk_norm and "q_norm" in params and "k_norm" in params:
        queries = apply_qk_norm(queries, params["q_norm"])
        keys = apply_qk_norm(keys, params["k_norm"])

    queries = apply_rope_with_offset(queries, cos, sin, position_offset)
    keys = apply_rope_with_offset(keys, cos, sin, position_offset)
    set_trace()
    
    if pre:
        kv_cache["keys"] = kv_cache["keys"].at[:,:,:keys.shape[2]].set(keys)
        kv_cache["values"] = kv_cache["values"].at[:,:,:values.shape[2]].set(values)
    else:
        kv_cache["keys"] = kv_cache["keys"].at[:,:,position_offset].set(keys[:,:,0])
        kv_cache["values"] = kv_cache["values"].at[:,:,position_offset].set(values[:,:,0])
        keys = kv_cache['keys']
        values = kv_cache['values']

    if pre:
        position_offset_new = keys.shape[2]
    else:
        position_offset_new = position_offset + 1
        
    new_cache = kv_cache
    keys_expanded = jnp.repeat(keys, group_size, axis=1)
    values_expanded = jnp.repeat(values, group_size, axis=1)

    # unbatch
    queries = queries[0]
    keys_expanded = keys_expanded[0]
    values_expanded = values_expanded[0]

    output = att2(queries, keys_expanded, values_expanded, params['out_proj'], pre, position_offset)[None,:]

    return output, new_cache, position_offset_new

def rms2(norm1, x):
    return jax.vmap(lambda y: rmsnorm_forward(norm1, y))(x)

def transformer_block_forward_kv(params, kv_cache, position_offset, x, pre=False):
    shortcut = x
    x = rmsnorm_forward(params["norm1"], x)
    #pre = False
    x, new_cache, position_offset = grouped_query_attention_forward_kv(cfg["n_heads"], cfg["n_kv_groups"], cfg["head_dim"], params["att"], kv_cache, cfg["qk_norm"], position_offset, x, pre)
    x = x + shortcut
    shortcut = x
    x = rmsnorm_forward(params["norm2"], x)
    x = feedforward_forward(params["ff"], x)
    return x + shortcut, new_cache, position_offset



def get_logits_old(cfg, x, params):
    logits_chunks = []
    seq_chunk_size = 1024
    for chunk_start in range(0, cfg["context_length"], seq_chunk_size):
        chunk_end = min(chunk_start + seq_chunk_size, cfg["context_length"])
        x_chunk = x[:, chunk_start:chunk_end]
        print(x_chunk.shape)
        print(params['out_head'].shape)
        logits_chunk = jnp.einsum('bse,ev->bsv', x_chunk, params["out_head"])
        logits_chunks.append(logits_chunk)
    
    logits = jnp.concatenate(logits_chunks, axis=1)
    return logits

def get_logits(cfg, x, params, vocab_chunk_size=128):
    """Chunked version of get_logits across the vocab/embedding dimension"""
    logits_chunks = []
    seq_chunk_size = 1
    
    for chunk_start in range(0, cfg["context_length"], seq_chunk_size):
        chunk_end = min(chunk_start + seq_chunk_size, cfg["context_length"])
        x_chunk = x[:, chunk_start:chunk_end]
        
        vocab_chunks = []
        vocab_size = params['out_head'].shape[0]
        
        for vocab_chunk_start in range(0, vocab_size, vocab_chunk_size):
            vocab_chunk_end = min(vocab_chunk_start + vocab_chunk_size, vocab_size)
            #print('--')
            #print(params['out_head'].shape)
            out_head_chunk = params["out_head"][:, vocab_chunk_start:vocab_chunk_end]
            #print(x_chunk.shape)
            #print(out_head_chunk.shape)
            logits_chunk = jnp.einsum('bse,ev->bsv', x_chunk, out_head_chunk)
            vocab_chunks.append(logits_chunk)
        
        # Concatenate vocab chunks for this sequence chunk
        logits_seq_chunk = jnp.concatenate(vocab_chunks, axis=2)
        logits_chunks.append(logits_seq_chunk)
    
    logits = jnp.concatenate(logits_chunks, axis=1)
    return logits


#@partial(jax.jit, static_argnums=[5], donate_argnums=[3])
def qwen3_forward_kv(params, x, cfg, kv_cache, position_offset, pre=False):
    x = params["tok_emb"][x]

    new_cache = {"keys": [], "values":[]}
    #set_trace()
    for i, block_params in enumerate(params["trf_blocks"]):
        layer_cache = {"keys": kv_cache["keys"][:,i], "values": kv_cache["values"][:,i]}
        x, updated_cache, position_offset_new = transformer_block_forward_kv(block_params, layer_cache, position_offset, x, pre=pre)
        kv_cache['keys'] = kv_cache['keys'].at[:,i].set(updated_cache['keys'])
        kv_cache['values'] = kv_cache['values'].at[:,i].set(updated_cache['values'])
    new_cache = kv_cache

    x = rmsnorm_forward(params["final_norm"], x)
    logits = jnp.einsum('bse,ev->bsv', x, params["out_head"])
    
    return logits, new_cache, position_offset_new


def steptop(params, cfg):
    #@jax.jit
    def step(args,_):#logits, kv_cache, position_offset, cur_ids):
        logits, kv_cache, position_offset = args

        next_token_logits = logits[:, -1, :]
        next_token = jnp.argmax(next_token_logits, axis=-1)

        '''
        if eos_id is not None and jnp.any(next_token == eos_id):
            break
        '''
        
        #cur_ids = jnp.concatenate([cur_ids, next_token[:, None]], axis=1)
        
        # Process next tokens for entire batch
        logits, kv_cache, position_offset = qwen3_forward_kv(params, next_token[:, None], cfg, kv_cache, position_offset)
        return [logits, kv_cache, position_offset ], next_token
        #return [logits, kv_cache, position_offset, cur_ids], None
    return step


#scan(step, init=[params, cfg, kv_cache])(next_token, position_offset)
def decode_step(carry,x):#logits, kv_cache, position_offset, cur_ids):
    params, logits, kv_cache, position_offset = carry

    next_token_logits = logits[:, -1, :]
    next_token = jnp.argmax(next_token_logits, axis=-1)
    
    # Process next tokens for entire batch
    logits, kv_cache, position_offset = qwen3_forward_kv(params, next_token[:, None], cfg, kv_cache, position_offset)
    return [params, logits, kv_cache, position_offset ], next_token

#@jax.jit
def gen(params, logits, kv_cache, position_offset,  max_new_tokens):
    [params, logits, kv_cache, position_offset], seq = jax.lax.scan(
            decode_step, 
            init=[params, logits, kv_cache, position_offset], length=max_new_tokens,
            unroll=False, #20,
            )
    return [logits, kv_cache, position_offset], seq

def gen2(f, logits, kv_cache, position_offset,  max_new_tokens):
    [logits, kv_cache, position_offset], seq = jax.lax.scan(
            f, 
            init=[logits, kv_cache, position_offset], length=max_new_tokens,
            unroll=False, #20,
            )
    return [logits, kv_cache, position_offset], seq

def block(args):
    for a in args:
        if isinstance(a, dict):
            for v in a.values():
                v.block_until_ready()
        elif isinstance(a, jax.Array):
            a.block_until_ready()
        elif isinstance(a, int):
            pass
        else:
            raise RuntimeError(a)

def generate_kv_optimized(model, idx, max_new_tokens, context_size, temperature=0.7, top_k=50, eos_id=None):
    params, cfg = model["params"], model["cfg"]
    cfg.pop('dtype')
    import operator
    
    # Keep input on device
    cur_ids = jnp.array([idx])
    key = jax.random.PRNGKey(42)
    
    # Initialize KV cache for batch processing
    n_layers = cfg['n_layers']
    n_kv_groups = cfg['n_kv_groups']
    head_dim = cfg['head_dim']
    kv_cache = {"keys": jnp.zeros((1, n_layers, n_kv_groups, context_size, head_dim), dtype=dtype), 
                 "values": jnp.zeros((1, n_layers, n_kv_groups, context_size, head_dim),dtype=dtype)} 
    csk = reduce(operator.mul, kv_cache['keys'].shape)
    csv = reduce(operator.mul, kv_cache['values'].shape)
    fs = csk+csv
    prec_factor = 2 # for bfloat16 or float16
    fs_gb = (fs / 1_000_000_000) * prec_factor

    print(f"cache size is: {fs}")
    print(f"cache size is: {fs_gb} GB")
    position_offset = 0
    
    # prefill1
    #if 0:
    logits, kv_cache, position_offset = qwen3_forward_kv(params, cur_ids, cfg, kv_cache, position_offset, pre=True)
    block([logits, kv_cache, position_offset])

    '''
    logits2, kv_cache2, position_offset2 = qwen3_forward_kv(params, cur_ids, cfg, kv_cache, position_offset, pre=True)
    block([logits2, kv_cache2, position_offset2])

    from time import perf_counter

    for i in range(3):
        stt = perf_counter()
        #logits, kv_cache, position_offset = qwen3_forward_kv(params, cur_ids, cfg, kv_cache, position_offset, pre=True)
        logits2, kv_cache2, position_offset2 = qwen3_forward_kv(params, cur_ids, cfg, kv_cache, position_offset, pre=True)
        block([logits2, kv_cache2, position_offset2])
        ft = perf_counter()
        print(ft-stt)

    set_trace()
    '''


    # get generation function
    '''
    f = steptop(params, cfg)
    # prefill2
    [logits, kv_cache, position_offset], _ = f([logits, kv_cache, position_offset], None)
    block([logits, kv_cache, position_offset])
    jax.clear_caches()
    '''
    f = steptop(params, cfg)
    # prefill2
    [logits, kv_cache, position_offset], _ = f([logits, kv_cache, position_offset], None)
    block([logits, kv_cache, position_offset])
    jax.clear_caches()
    '''

    cur_ids2 = jnp.array([[999]*26])

    #logits, kv_cache, position_offset, cur_ids = gen(f, logits, kv_cache, position_offset, cur_ids2, max_new_tokens)
    set_trace()
    traced = jax.jit(gen, static_argnums=[3,4]).trace(params, logits, kv_cache, int(position_offset), max_new_tokens)
    lowered = traced.lower()
    compiled_gen = lowered.compile()
    import gc
    '''

    use_lax = False
    if use_lax:
        # warmup
        cur_ids3 = jnp.array([[1999]*26])
        [params1,logits1, kv_cache1, position_offset1], seq2 = compiled_gen(params, logits, kv_cache)
        block([params1, logits1, kv_cache1, position_offset1, seq2])
        del logits1, kv_cache1, position_offset1, 
        gc.collect()

        stt = time.time()
        [logits2, kv_cache2, position_offset2], seq2 = compiled_gen(logits, kv_cache)
        block([logits2, kv_cache2, position_offset2, seq2])
        fin = time.time()
        del logits2, kv_cache2, position_offset2, 
        print(f"time: {fin-stt}")
        gc.collect()

        time.sleep(5)

        stt = time.time()
        options = jax.profiler.ProfileOptions()
        options.python_tracer_level = 0
        options.host_tracer_level = 3
        #with jax.profiler.trace("/tmp/jax-trace1", create_perfetto_link=True):
        profile = False
        if profile:
            jax.profiler.start_trace("/tmp/jax-trace1")#, profiler_options=options)

        [logits2, kv_cache2, position_offset2], seq = compiled_gen(logits, kv_cache )
        block([logits2, kv_cache2, position_offset2, seq])
        fin = time.time()
        del logits2, kv_cache2, position_offset2, 
        tps = max_new_tokens / (fin-stt) 
        print(f"time: {fin-stt} for {max_new_tokens} at {tps} tok/s")
        gc.collect()

        if profile:
            jax.profiler.stop_trace()
    else:
        #set_trace()
        seq = []
        for i in tqdm(range(max_new_tokens), desc="Generating"):
            #[logits, kv_cache, position_offset, ], seq = f([logits, kv_cache, position_offset, ], None)
            [params, logits, kv_cache, position_offset], nt = decode_step([params, logits, kv_cache, position_offset], None )
            seq.append(nt)
            if eos_id is not None and cur_ids[-1] == eos_id:
                break
        seq = jnp.stack(seq)

    
    '''
    for i in tqdm(range(max_new_tokens), desc="Generating"):
        next_token_logits = logits[:, -1, :]
        
        if top_k is not None and top_k > 0:
            # Vectorized top_k for batch processing
            top_k_logits, top_k_indices = jax.lax.top_k(next_token_logits, top_k)
            mask = jnp.full_like(next_token_logits, -jnp.inf)
            mask = jnp.take_along_axis(mask, top_k_indices, axis=-1)
            mask = jnp.where(jnp.arange(mask.shape[-1])[None, :] < top_k, top_k_logits, -jnp.inf)
            next_token_logits = jnp.full_like(next_token_logits, -jnp.inf)
            next_token_logits = next_token_logits.at[jnp.arange(1)[:, None], top_k_indices].set(mask)
        
        if temperature > 0.0:
            next_token_logits = next_token_logits / temperature
            key, subkey = jax.random.split(key)
            next_token = jax.random.categorical(subkey, next_token_logits, axis=-1)
        else:
            next_token = jnp.argmax(next_token_logits, axis=-1)
        
        # Check EOS for all sequences in batch - keep on device
        if eos_id is not None and jnp.any(next_token == eos_id):
            break
        
        cur_ids = jnp.concatenate([cur_ids, next_token[:, None]], axis=1)
        
        # Process next tokens for entire batch
        logits, kv_cache, position_offset = qwen3_forward_kv(params, next_token[:, None], cfg, kv_cache, position_offset)
        '''
    
    return jnp.stack(seq, axis=-1)
