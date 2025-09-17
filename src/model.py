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
import gc
import operator
from math import ceil

bp = jax.debug.breakpoint
jax.config.update('jax_numpy_dtype_promotion', 'strict')

from functools import partial, reduce

# set to speed up jit during experimentation
unroll = True

use_gpu = True
if use_gpu and jax.default_backend() == 'gpu':
    device =  jax.devices('gpu')[0]
else:
    device =  jax.devices('cpu')[0]

from model_utils import *

dtype = jax.dtypes.bfloat16
#dtype = jnp.float16
#dtype = jnp.float32
cl = 40960
#cl = 32000
#cl = 128
#cl = 8192
#cl = 1024
#cl = 4096

cfg = {
    "vocab_size": 151936, "context_length": cl, "emb_dim": 1024, "n_heads": 16,
    "n_layers": 28, "hidden_dim": 3072, "head_dim": 128, "qk_norm": True,
    "n_kv_groups": 8, "rope_base": 1000000.0, "dtype": 'bfloat16', #torch.bfloat16,
}

def att_head_tiled_pre(queries, keys, values, pre, position_offset, tile_size=2*1024):
    dtype_l = dtype
    queries = queries.astype(dtype_l)
    keys = keys.astype(dtype_l)
    values = values.astype(dtype_l)

    m, d_k = queries.shape
    n, _ = keys.shape
    head_dim = d_k

    # TODO add check about correct tilesize evenly dividing the shapes
    # only needed during prefill, otherwise m is 1, bc 1 token decode
    # make tile_size match the tiny prefill
    if m % tile_size != 0:
        tile_size = queries.shape[0]

    Fi = jnp.zeros((m, values.shape[-1]), dtype=dtype_l)
    Li = jnp.zeros((m, 1), dtype=dtype_l)
    Mi = jnp.full((m, 1), jnp.finfo(dtype_l).min, dtype=dtype_l)
    init = (Fi, Li, Mi, position_offset)
    init_pre = (Fi, Li, Mi)

    num_full_tiles = n // tile_size
    ks = jnp.reshape(keys[:num_full_tiles * tile_size], (num_full_tiles, tile_size, d_k))
    vs = jnp.reshape(values[:num_full_tiles * tile_size], (num_full_tiles, tile_size, d_k))

    def causal_bodyfn_pre(carry, scan_input):
        k_chunk, v_chunk, start_idx = scan_input
        F, L, M, = carry
        logits = jnp.matmul(queries, k_chunk.T) / jnp.sqrt(head_dim)  # (m, tile_sz)
        k_positions = jnp.arange(tile_size)[None, :] + start_idx  # (1, tile_sz)

        # Causal masking prefill
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

        return (F2, L2, M2), None

    # Pass start indices with chunks
    start_indices = jnp.arange(0, num_full_tiles * tile_size, tile_size)
    scan_inputs = (ks, vs, start_indices)

    (Fn, Ln, _), _ = jax.lax.scan(causal_bodyfn_pre, init_pre, scan_inputs, unroll=False)

    # no nans allowed
    # cant assert #assert jnp.all(Ln != 0.)
    # needs checkify wrapping #jax.experimental.checkify.check(jnp.all(Ln != 0, "null in softmax denominator"))
    return (Fn / Ln).astype(dtype)
    #return jnp.where(L != 0, F / L, 0.0)


def attention_head(queries, keys_expanded, values_expanded, position_offset):
    ''' simpler, untiled attention for generation '''
    attn_scores = jnp.matmul(queries, keys_expanded.transpose(1,0)) / jnp.sqrt(cfg['head_dim'])
    mask = np.arange(keys_expanded.shape[0]) > position_offset + 1
    attn_scores = jnp.where(mask, jnp.finfo(dtype).min, attn_scores)
    attn_weights = jax.nn.softmax(attn_scores, axis=-1)
    context = jnp.matmul(attn_weights, values_expanded)
    return context

def manual_vmap(att_head, queries, keys_expanded, values_expanded, pre, position_offset):
    def bodyfn(carry, x):
        queries, keys_expanded, values_expanded = x
        x = att_head(queries, keys_expanded, values_expanded, pre, position_offset)
        return carry, x
    carry, xr = jax.lax.scan(bodyfn, None, xs=(queries, keys_expanded, values_expanded), unroll=True)
    return xr


def attention_heads_prefill(queries, keys_expanded, values_expanded, out_proj, pre, position_offset, tiled=True):
    # use tiling iff prefill
    context = manual_vmap(att_head_tiled_pre, queries, keys_expanded, values_expanded, pre, position_offset)

    context = context.transpose(1,0,2).reshape(queries.shape[1], cfg['n_heads'] * cfg['head_dim'])
    output = jnp.einsum('sh,hd->sd', context, out_proj)
    return output
    
def grouped_query_attention_forward_kv_pre(num_heads, num_kv_groups, head_dim, params, kv_cache, qk_norm, position_offset, x, pre=True):
    cos, sin = compute_rope_params(cfg["head_dim"], cfg["rope_base"], cfg["context_length"])
    group_size = num_heads // num_kv_groups
    seq, d_in = x.shape
    
    queries = jnp.einsum('sd,dh->sh', x, params["W_query"]).reshape(seq, num_heads, head_dim).transpose(1,0,2)
    keys = jnp.einsum('sd,dh->sh', x, params["W_key"]).reshape(seq, num_kv_groups, head_dim).transpose(1,0,2)

    if qk_norm and "q_norm" in params and "k_norm" in params:
        queries = apply_qk_norm(queries, params["q_norm"])
        keys = apply_qk_norm(keys, params["k_norm"])


    queries = apply_rope_with_offset(queries[None], cos, sin, position_offset)[0]
    keys = apply_rope_with_offset(keys[None], cos, sin, position_offset)[0]
    values = jnp.einsum('sd,dh->sh', x, params["W_value"]).reshape(seq, num_kv_groups, head_dim).transpose(1,0,2)

    kv_cache["keys"] = kv_cache["keys"].at[0,:,:keys.shape[1]].set(keys)
    kv_cache["values"] = kv_cache["values"].at[0,:,:values.shape[1]].set(values)
    position_offset_new = keys.shape[1]
    #keys = kv_cache['keys'][0]
    #values = kv_cache['values'][0]
        
    new_cache = kv_cache
    keys_expanded = jnp.repeat(keys, group_size, axis=0)
    values_expanded = jnp.repeat(values, group_size, axis=0)

    output = attention_heads_prefill(queries, keys_expanded, values_expanded, params['out_proj'], pre, position_offset)

    return output, new_cache, position_offset_new


def grouped_query_attention_forward_kv_gen(num_heads, num_kv_groups, head_dim, params, kv_cache, qk_norm, position_offset, x, pre=False):
    cos, sin = compute_rope_params(cfg["head_dim"], cfg["rope_base"], cfg["context_length"])
    group_size = num_heads // num_kv_groups
    d_in = x.shape
    
    # during inference seqlen (new prefill tokens) is 1
    queries = jnp.einsum('d,dh->h', x, params["W_query"]).reshape(num_heads, head_dim)[:, None]
    keys = jnp.einsum('d,dh->h', x, params["W_key"]).reshape(num_kv_groups, head_dim)[:, None]

    if qk_norm and "q_norm" in params and "k_norm" in params:
        queries = apply_qk_norm(queries, params["q_norm"])
        keys = apply_qk_norm(keys, params["k_norm"])

    queries = apply_rope_with_offset(queries[None], cos, sin, position_offset)[0]
    keys = apply_rope_with_offset(keys[None, :, 0:1], cos, sin, position_offset)[0, :, 0]

    #values = jnp.einsum('sd,dh->sh', x, params["W_value"]).reshape(seq, num_kv_groups, head_dim).transpose(1,0,2)[:, 0:1]
    values = jnp.einsum('d,dh->h', x, params["W_value"]).reshape(num_kv_groups, head_dim)

    #keys2 = jnp.concat([kv_cache['keys'][0,:,:26], keys[:,0]])[0]
    #values2 = jnp.concat([kv_cache['values'][0,:,:26], values[:,0]])[0]
    #bp()

    kv_cache["keys"] = kv_cache["keys"].at[0,:,position_offset].set(keys)
    kv_cache["values"] = kv_cache["values"].at[0,:,position_offset].set(values)

    new_k = keys
    new_v = values

    # this still atm runs the whole seqlen of the kv cache
    # batch, heads, seqlen, embdim
    keys = kv_cache['keys'][0]
    values = kv_cache['values'][0]

    #keys = jax.lax.dynamic_slice(keys, [0, 0, 0], (keys.shape[0], 1024, keys.shape[2]))
    #values = jax.lax.dynamic_slice(values, [0, 0, 0], (values.shape[0], 1024, values.shape[2]))

    # hardcoding the max amount of new tokens, limits prefill too for now
    #keys = keys[:,:1024]
    #values = values[:,:1024]

    position_offset_new = position_offset + 1
        
    new_cache = {"keys": new_k, "values":new_v}
    keys_expanded = jnp.repeat(keys, group_size, axis=0)
    values_expanded = jnp.repeat(values, group_size, axis=0)

    # only compute on the non cached values
    qp = jax.lax.dynamic_slice(queries, (0, position_offset,0), (16,1,128))
    context = jax.vmap(attention_head, (0,0,0,None))(qp, keys_expanded, values_expanded, position_offset)
    context = context.transpose(1,0,2).reshape(qp.shape[1], cfg['n_heads'] * cfg['head_dim'])
    output = jnp.einsum('sh,hd->sd', context, params['out_proj'])

    return output, new_cache, position_offset_new


def transformer_block_forward_kv(params, kv_cache, position_offset, x, pre=False):
    shortcut = x
    x = rmsnorm_forward(params["norm1"], x)
    if pre:
        x, new_cache, position_offset = grouped_query_attention_forward_kv_pre(cfg["n_heads"], cfg["n_kv_groups"], cfg["head_dim"], params["att"], kv_cache, cfg["qk_norm"], position_offset, x, pre)
    else:
        x, new_cache, position_offset = grouped_query_attention_forward_kv_gen(cfg["n_heads"], cfg["n_kv_groups"], cfg["head_dim"], params["att"], kv_cache, cfg["qk_norm"], position_offset, x, pre)

    x = x + shortcut
    shortcut = x
    x = rmsnorm_forward(params["norm2"], x)
    x = feedforward_forward(params["ff"], x[None])
    return x + shortcut, new_cache, position_offset


def qwen3_forward_kv(params, x, cfg, kv_cache, position_offset, pre=False):
    if pre:
        # unbatch
        x = x[0]
    else:
        # unbatch and unseq
        x = x[0,0]

    x = params["tok_emb"][x]
    new_cache = {"keys": [], "values":[]}

    for i, block_params in enumerate(params["trf_blocks"]):
        layer_cache = {"keys": kv_cache["keys"][:,i], "values": kv_cache["values"][:,i]}
        x, updated_cache, position_offset_new = transformer_block_forward_kv(block_params, layer_cache, position_offset, x, pre=pre)
        if pre:

            kv_cache['keys'] = kv_cache['keys'].at[:,i].set(updated_cache['keys'])
            kv_cache['values'] = kv_cache['values'].at[:,i].set(updated_cache['values'])
            x = x[0]
        else:
            #set_trace()
            kv_cache["keys"] = kv_cache["keys"].at[0, i,:, position_offset].set(updated_cache['keys'])
            kv_cache["values"] = kv_cache["values"].at[0, i,:, position_offset].set(updated_cache['values'])
            x = x[0,0]

    new_cache = kv_cache

    x = rmsnorm_forward(params["final_norm"], x)
    # cant do logits on prefill because it would be of shape [ctxlen, vocab] <- too big
    if pre:
        logits = jnp.matmul(x[-1:], params["out_head"])[None, :]
    else:
        logits = jnp.matmul(x[None], params["out_head"])[None, :]
    
    return logits, new_cache, position_offset_new


def decode_step(carry,x):#logits, kv_cache, position_offset, cur_ids):
    params, logits, kv_cache, position_offset = carry

    next_token_logits = logits[:, -1, :]
    next_token = jnp.argmax(next_token_logits, axis=-1)
    
    # Process next tokens for entire batch
    # TODO access bounds position here
    logits, kv_cache, position_offset2 = qwen3_forward_kv(params, next_token[:, None], cfg, kv_cache, position_offset)
    return [params, logits, kv_cache, position_offset ], next_token

def gen(params, logits, kv_cache, position_offset,  max_new_tokens):
    [params, logits, kv_cache, position_offset], seq = jax.lax.scan(
            decode_step, 
            init=[params, logits, kv_cache, position_offset], 
            length=max_new_tokens,
            unroll=4, 
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
    
    options = jax.profiler.ProfileOptions()
    options.python_tracer_level = 0
    options.host_tracer_level = 3
    #with jax.profiler.trace("/tmp/jax-trace1", create_perfetto_link=True):

    print('compiling prefill program')
    traced_pre = jax.jit(qwen3_forward_kv, static_argnums=[4,5], donate_argnums=[3]).trace(
            params, cur_ids, cfg, kv_cache, position_offset, pre=True)
    lowered = traced_pre.lower()
    compiled_pre = lowered.compile()
    print('finished compiling prefill program')


    print('compile generation program')
    #cur_ids2 = jnp.array([[999]*26])
    #logits, kv_cache, position_offset, cur_ids = gen(f, logits, kv_cache, position_offset, cur_ids2, max_new_tokens)
    #set_trace()
    #traced = jax.jit(gen, static_argnums=[3,4]).trace(params, logits, kv_cache, int(position_offset), max_new_tokens)
    vocab_size = cfg['vocab_size']
    logits = jnp.ones([1,1,vocab_size], dtype=dtype) / vocab_size
    profile = False
    if profile:
        #max_new_tokens = 2
        pass
    traced = jax.jit(gen, static_argnums=[4], donate_argnums=[2]).trace(params, logits, kv_cache, int(position_offset), max_new_tokens)
    lowered = traced.lower()
    compiled_gen = lowered.compile()
    print('finished compile generation program')

    # run prefill
    print('running prefill')
    stt = time.perf_counter()
    if profile:
        jax.profiler.start_trace("/tmp/jax-trace1")#, profiler_options=options)

    logits, kv_cache, position_offset = compiled_pre(params, cur_ids, cfg, kv_cache)
    block([logits, position_offset])

    if profile:
        jax.profiler.stop_trace()
    ft = time.perf_counter()
    tt = ft - stt
    toks = cur_ids.shape[-1]
    print(f"took: {tt} for {cur_ids.shape[-1]} at {toks/tt} toks/sec")
    print('prefilled')
    print('running generation')


    # warmup
    warmup = False
    if warmup:
        cur_ids3 = jnp.array([[1999]*26])
        #[logits1, kv_cache1, position_offset1], seq = compiled_gen(params, logits, kv_cache)#, int(position_offset), max_new_tokens)
        [logits1, kv_cache1, position_offset1], seq = compiled_gen(params, logits, kv_cache, int(position_offset))#, max_new_tokens)
        block([logits1, position_offset1, seq])
        del logits1, kv_cache1, position_offset1
        gc.collect()

    stt = time.time()
    profile = True
    if profile:
        jax.profiler.start_trace("/tmp/jax-trace1")#, profiler_options=options)

    stt2 = time.perf_counter()
    #[logits1, kv_cache1, position_offset1], seq = compiled_gen(params, logits, kv_cache)#, int(position_offset), max_new_tokens)
    [logits1, kv_cache1, position_offset1], seq = compiled_gen(params, logits, kv_cache, int(position_offset))#, max_new_tokens)
    block([logits1, kv_cache1, position_offset1, seq])

    ft2 = time.perf_counter()
    tt2 = ft2 - stt2
    #toks = 2*max_new_tokens
    toks2 = seq.shape[-2]
    #set_trace()
    print(f"took: {tt2} for {toks2} at {toks2/tt2} toks/sec")

    if profile:
        jax.profiler.stop_trace()

    return jnp.stack(seq, axis=-1)

def generate_kv_optimized_programs(model, idx, max_new_tokens, context_size, temperature=0.7, top_k=50, eos_id=None):
    params, cfg = model["params"], model["cfg"]
    
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
    
    options = jax.profiler.ProfileOptions()
    options.python_tracer_level = 0
    options.host_tracer_level = 3
    #with jax.profiler.trace("/tmp/jax-trace1", create_perfetto_link=True):

    print('compiling prefill program')
    traced_pre = jax.jit(qwen3_forward_kv, static_argnums=[4,5], donate_argnums=[3]).trace(
            params, cur_ids, cfg, kv_cache, position_offset, pre=True)
    lowered = traced_pre.lower()
    compiled_pre = lowered.compile()
    print('finished compiling prefill program')


    print('compile generation program')
    #cur_ids2 = jnp.array([[999]*26])
    #logits, kv_cache, position_offset, cur_ids = gen(f, logits, kv_cache, position_offset, cur_ids2, max_new_tokens)
    #set_trace()
    #traced = jax.jit(gen, static_argnums=[3,4]).trace(params, logits, kv_cache, int(position_offset), max_new_tokens)
    vocab_size = cfg['vocab_size']
    logits = jnp.ones([1,1,vocab_size], dtype=dtype) / vocab_size
    traced = jax.jit(gen, static_argnums=[4], donate_argnums=[2]).trace(params, logits, kv_cache, int(position_offset), max_new_tokens)
    lowered = traced.lower()
    compiled_gen = lowered.compile()
    print('finished compile generation program')

    return kv_cache, compiled_pre, compiled_gen

def infer(params, cur_ids, cfg, kv_cache, compiled_pre, compiled_gen):
    print('running prefill')
    logits, kv_cache1, position_offset = compiled_pre(params, cur_ids, cfg, kv_cache)
    print('prefilled')
    print('running generation')
    [logits1, kv_cache2, position_offset1], seq = compiled_gen(params, logits, kv_cache1, int(position_offset))

    return jnp.stack(seq, axis=-1), kv_cache2
