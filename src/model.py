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
#cl = 8192
#cl = 1024

cfg = {
    "vocab_size": 151936, "context_length": cl, "emb_dim": 1024, "n_heads": 16,
    "n_layers": 28, "hidden_dim": 3072, "head_dim": 128, "qk_norm": True,
    "n_kv_groups": 8, "rope_base": 1000000.0, "dtype": 'bfloat16', #torch.bfloat16,
}

def manual_vmap(att_head, queries, keys_expanded, values_expanded, pre, position_offset):
    def bodyfn(carry, x):
        queries, keys_expanded, values_expanded = x
        x = att_head(queries, keys_expanded, values_expanded, pre, position_offset)
        return carry, x
    carry, xr = jax.lax.scan(bodyfn, None, xs=(queries, keys_expanded, values_expanded), unroll=False)
    return xr

def att_head_coder_causal_pre(queries, keys, values, pre, position_offset, tile_size=2*1024):
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
    Mi = jnp.full((m, 1), -jnp.inf, dtype=dtype_l)
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


def att_head_orig(queries, keys_expanded, values_expanded, pre, position_offset):
    attn_scores = jnp.matmul(queries, keys_expanded.transpose(1,0)) / jnp.sqrt(cfg['head_dim'])

    if pre:
        q_len, k_len = queries.shape[0], keys_expanded.shape[0]
        causal_mask = jnp.triu(jnp.ones((q_len, k_len)), k=1)
        attn_scores = jnp.where(causal_mask, -jnp.inf, attn_scores)

    else:
        mask = np.arange(keys_expanded.shape[0]) > position_offset + 1
        attn_scores = jnp.where(mask, -jnp.float_('inf'), attn_scores)

    attn_weights = jax.nn.softmax(attn_scores, axis=-1)
    context = jnp.matmul(attn_weights, values_expanded)
    return context


def gen_att(queries, keys_expanded, values_expanded, position_offset):
    ''' simpler, untiled attention for generation '''
    attn_scores = jnp.matmul(queries, keys_expanded.transpose(1,0)) / jnp.sqrt(cfg['head_dim'])
    mask = np.arange(keys_expanded.shape[0]) > position_offset + 1
    attn_scores = jnp.where(mask, -jnp.float_('inf'), attn_scores)
    attn_weights = jax.nn.softmax(attn_scores, axis=-1)
    context = jnp.matmul(attn_weights, values_expanded)
    return context


def att2(queries, keys_expanded, values_expanded, out_proj, pre, position_offset, tiled=True):
    # use tiling iff prefill
    if pre:
        context = manual_vmap(att_head_coder_causal_pre, queries, keys_expanded, values_expanded, pre, position_offset)
    else:
        context = manual_vmap(att_head_orig, queries, keys_expanded, values_expanded, pre, position_offset)

    context = context.transpose(1,0,2).reshape(queries.shape[1], cfg['n_heads'] * cfg['head_dim'])
    output = jnp.einsum('sh,hd->sd', context, out_proj)
    return output
    

def grouped_query_attention_forward_kv(num_heads, num_kv_groups, head_dim, params, kv_cache, qk_norm, position_offset, x, pre=False):
    cos, sin = compute_rope_params(cfg["head_dim"], cfg["rope_base"], cfg["context_length"])
    group_size = num_heads // num_kv_groups
    if pre:
        seq, d_in = x.shape
    else:
        d_in = x.shape
    
    if pre:
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
    else:
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

        kv_cache["keys"] = kv_cache["keys"].at[0,:,position_offset].set(keys)
        kv_cache["values"] = kv_cache["values"].at[0,:,position_offset].set(values)

        # this still atm runs the whole seqlen of the kv cache
        # batch, heads, seqlen, embdim
        keys = kv_cache['keys'][0]
        values = kv_cache['values'][0]

        #set_trace()
        #keys = jax.lax.dynamic_slice(keys, [0, 0, 0], (keys.shape[0], 1024, keys.shape[2]))
        #values = jax.lax.dynamic_slice(values, [0, 0, 0], (values.shape[0], 1024, values.shape[2]))

        # hardcoding the max amount of new tokens, limits prefill too for now
        #keys = keys[:,:1024]
        #values = values[:,:1024]

        position_offset_new = position_offset + 1
        
    new_cache = kv_cache
    keys_expanded = jnp.repeat(keys, group_size, axis=0)
    values_expanded = jnp.repeat(values, group_size, axis=0)

    if pre:
        output = att2(queries, keys_expanded, values_expanded, params['out_proj'], pre, position_offset)
    else:
        # only compute on the non cached values
        qp = jax.lax.dynamic_slice(queries, (0, position_offset,0), (16,1,128))
        context = jax.vmap(gen_att, (0,0,0,None))(qp, keys_expanded, values_expanded, position_offset)
        context = context.transpose(1,0,2).reshape(qp.shape[1], cfg['n_heads'] * cfg['head_dim'])
        output = jnp.einsum('sh,hd->sd', context, params['out_proj'])

    return output, new_cache, position_offset_new


def transformer_block_forward_kv(params, kv_cache, position_offset, x, pre=False):
    shortcut = x
    x = rmsnorm_forward(params["norm1"], x)
    x, new_cache, position_offset = grouped_query_attention_forward_kv(cfg["n_heads"], cfg["n_kv_groups"], cfg["head_dim"], params["att"], kv_cache, cfg["qk_norm"], position_offset, x, pre)
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
        kv_cache['keys'] = kv_cache['keys'].at[:,i].set(updated_cache['keys'])
        kv_cache['values'] = kv_cache['values'].at[:,i].set(updated_cache['values'])
        if pre:
            x = x[0]
        else:
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
    logits, kv_cache, position_offset = qwen3_forward_kv(params, next_token[:, None], cfg, kv_cache, position_offset)
    return [params, logits, kv_cache, position_offset ], next_token

def gen(params, logits, kv_cache, position_offset,  max_new_tokens):
    [params, logits, kv_cache, position_offset], seq = jax.lax.scan(
            decode_step, 
            init=[params, logits, kv_cache, position_offset], length=max_new_tokens,
            unroll=1, #20, # unroll = 2 crashes with internal error, unroll = 3 produces different result, false, 4, 5 are fine
            # unroll 6 also is different like unroll 3
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

    # compile prefill1
    traced_pre = jax.jit(qwen3_forward_kv, static_argnums=[4,5], donate_argnums=[3]).trace(
            params, cur_ids, cfg, kv_cache, position_offset, pre=True)
    lowered = traced_pre.lower()
    compiled_pre = lowered.compile()

    # run prefill
    stt = time.perf_counter()
    profile = False
    if profile:
        jax.profiler.start_trace("/tmp/jax-trace1")#, profiler_options=options)

    print('starting prefill')
    logits, kv_cache, position_offset = compiled_pre(params, cur_ids, cfg, kv_cache)
    block([logits, position_offset])

    if profile:
        jax.profiler.stop_trace()
    ft = time.perf_counter()
    tt = ft - stt
    toks = cur_ids.shape[-1]
    print(f"took: {tt} for {cur_ids.shape[-1]} at {toks/tt} toks/sec")
    print('prefilled')

    #cur_ids2 = jnp.array([[999]*26])
    #logits, kv_cache, position_offset, cur_ids = gen(f, logits, kv_cache, position_offset, cur_ids2, max_new_tokens)
    #set_trace()
    #traced = jax.jit(gen, static_argnums=[3,4]).trace(params, logits, kv_cache, int(position_offset), max_new_tokens)
    traced = jax.jit(gen, static_argnums=[4], donate_argnums=[]).trace(params, logits, kv_cache, int(position_offset), max_new_tokens)
    lowered = traced.lower()
    compiled_gen = lowered.compile()
    print('compiled')

    use_lax = True
    if use_lax:
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
        profile = False
        if profile:
            jax.profiler.start_trace("/tmp/jax-trace1")#, profiler_options=options)

        '''
        cur_ids3 = jnp.array([[1999]*26])
        #[logits1, kv_cache1, position_offset1], seq = compiled_gen(params, logits, kv_cache)#, int(position_offset), max_new_tokens)
        [logits1, kv_cache1, position_offset1], seq = compiled_gen(params, logits, kv_cache, int(position_offset))#, max_new_tokens)
        block([logits1, kv_cache1, position_offset1, seq])
        '''

        stt = time.perf_counter()
        cur_ids3 = jnp.array([[1999]*26])
        #[logits1, kv_cache1, position_offset1], seq = compiled_gen(params, logits, kv_cache)#, int(position_offset), max_new_tokens)
        [logits1, kv_cache1, position_offset1], seq = compiled_gen(params, logits, kv_cache, int(position_offset))#, max_new_tokens)
        block([logits1, position_offset1, seq])

        ft = time.perf_counter()
        tt = ft - stt
        toks = 2*max_new_tokens
        print(f"took: {tt} for {2*max_new_tokens} at {toks/tt} toks/sec")

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
