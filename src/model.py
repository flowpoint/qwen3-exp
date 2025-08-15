import jax
import jax.numpy as jnp
from tokenizers import Tokenizer
import torch
from safetensors.numpy import load_file 
import os
from pathlib import Path
import gc
from collections import defaultdict
import numpy as np 
from tqdm import tqdm

from functools import partial

if jax.default_backend() == 'gpu':
    device =  jax.devices('gpu')[0]
else:
    device =  jax.devices('cpu')[0]

@jax.jit
def feedforward_forward(params, x):
    gate = jax.nn.silu(jnp.einsum('bse,eh->bsh', x, params["gate_proj"]))
    up = jnp.einsum('bse,eh->bsh', x, params["up_proj"])
    return jnp.einsum('bsh,he->bse', gate * up, params["down_proj"])

dtype = jax.dtypes.bfloat16
cfg = {
    "vocab_size": 151936, "context_length": 40960, "emb_dim": 1024, "n_heads": 16,
    "n_layers": 28, "hidden_dim": 3072, "head_dim": 128, "qk_norm": True,
    "n_kv_groups": 8, "rope_base": 1000000.0, "dtype": 'bfloat16', #torch.bfloat16,
}

@jax.jit
def rmsnorm_forward(params, x, eps=1e-6):
    orig_dtype = dtype
    x = x.astype(jnp.float32)
    variance = jnp.mean(x ** 2, axis=-1, keepdims=True)
    norm_x = x * jax.lax.rsqrt(variance + eps) * params["scale"]
    return norm_x.astype(orig_dtype)

def compute_rope_params(head_dim, theta_base=10000.0, context_length=4096):
    inv_freq = 1.0 / (theta_base ** (jnp.arange(0, head_dim, 2) / head_dim))
    positions = jnp.arange(context_length)
    angles = jnp.concatenate([positions[:, None] * inv_freq[None, :]] * 2, axis=1)
    return jnp.cos(angles), jnp.sin(angles)

def apply_rope(x, cos, sin):
    seq_len = x.shape[2]
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    cos, sin = cos[:seq_len, :][None, None, :, :], sin[:seq_len, :][None, None, :, :]
    rotated = jnp.concatenate([-x2, x1], axis=-1)
    return ((x * cos) + (rotated * sin)).astype(dtype)

#@jax.jit
'''
def apply_rope_with_offset(x, cos, sin, position_offset=0):
    #seq_len = x.shape[2]
    os = x.shape[2]
    xs = x.shape
    seq_len = cfg['context_length']

    x = jnp.resize(x, (xs[0], xs[1], seq_len,xs [-1]))
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    
    positions = jnp.arange(position_offset, position_offset + seq_len)
    cos_slice = cos[positions, :][None, None, :, :]
    sin_slice = sin[positions, :][None, None, :, :]
    
    rotated = jnp.concatenate([-x2, x1], axis=-1)
    return ((x * cos_slice) + (rotated * sin_slice)).astype(dtype)[:,:,:os,:]
'''
def apply_rope_with_offset_pre(x, cos, sin, position_offset=0):
    seq_len = x.shape[2]
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    
    positions = jnp.arange(position_offset, position_offset + seq_len)
    cos_slice = cos[positions, :][None, None, :, :]
    sin_slice = sin[positions, :][None, None, :, :]
    
    rotated = jnp.concatenate([-x2, x1], axis=-1)
    return ((x * cos_slice) + (rotated * sin_slice)).astype(dtype)

def apply_rope_with_offset(x, cos, sin, position_offset=0):
    seq_len = x.shape[2]
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    
    positions = jnp.arange(0, 0 + seq_len) + position_offset
    cos_slice = cos[positions, :][None, None, :, :]
    sin_slice = sin[positions, :][None, None, :, :]
    
    rotated = jnp.concatenate([-x2, x1], axis=-1)
    return ((x * cos_slice) + (rotated * sin_slice)).astype(dtype)

def apply_qk_norm(x, norm_params):
    b, h, s, d = x.shape
    x_reshaped = x.reshape(b * h * s, d)
    x_normed = rmsnorm_forward(norm_params, x_reshaped)
    return x_normed.reshape(b, h, s, d)

def grouped_query_attention_forward_kv_pre(num_heads, num_kv_groups, head_dim, cos, sin, params, mask,  kv_cache, qk_norm, position_offset, x):
    b, seq, d_in = x.shape
    group_size = num_heads // num_kv_groups
    
    #assert position_offset == kv_cache["keys"].shape[2], f"{kv_cache['keys'].shape[2]} {position_offset}"
    
    queries = jnp.einsum('bsd,dh->bsh', x, params["W_query"]).reshape(b, seq, num_heads, head_dim).transpose(0,2,1,3)
    keys = jnp.einsum('bsd,dh->bsh', x, params["W_key"]).reshape(b, seq, num_kv_groups, head_dim).transpose(0,2,1,3)
    values = jnp.einsum('bsd,dh->bsh', x, params["W_value"]).reshape(b, seq, num_kv_groups, head_dim).transpose(0,2,1,3)

    if qk_norm and "q_norm" in params and "k_norm" in params:
        queries = apply_qk_norm(queries, params["q_norm"])
        keys = apply_qk_norm(keys, params["k_norm"])

    queries = apply_rope_with_offset_pre(queries, cos, sin, position_offset)
    keys = apply_rope_with_offset_pre(keys, cos, sin, position_offset)
    
    keys = jnp.concatenate([kv_cache["keys"][:,:, :position_offset], keys], axis=2)
    values = jnp.concatenate([kv_cache["values"][:,:,:position_offset], values], axis=2)
    
    kv_cache["keys"] = kv_cache["keys"].at[:,:,:keys.shape[2]].set(keys)
    kv_cache["values"] = kv_cache["values"].at[:,:,:values.shape[2]].set(values)
    new_cache = kv_cache
    position_offset_new = keys.shape[2]
    
    keys_expanded = jnp.repeat(keys, group_size, axis=1)
    values_expanded = jnp.repeat(values, group_size, axis=1)
    
    attn_scores = jnp.einsum('bnqh,bnkh->bnqk', queries, keys_expanded) / jnp.sqrt(head_dim)
    
    if position_offset == 0:
        q_len, k_len = queries.shape[2], keys.shape[2]
        causal_mask = jnp.triu(jnp.ones((q_len, k_len)), k=1)
        attn_scores = jnp.where(causal_mask[None, None, :, :], -jnp.inf, attn_scores)
    
    attn_weights = jax.nn.softmax(attn_scores, axis=-1)
    context = jnp.einsum('bnqk,bnkh->bnqh', attn_weights, values_expanded)
    context = context.transpose(0,2,1,3).reshape(b, seq, num_heads * head_dim)
    output = jnp.einsum('bsh,hd->bsd', context, params["out_proj"])

    return output, new_cache, position_offset_new

@partial(jax.jit, static_argnums=[0,1,2,3,4, 6, 8])
def grouped_query_attention_forward_kv(num_heads, num_kv_groups, head_dim, cos, sin, params, mask,  kv_cache, qk_norm, position_offset, x):
    b, seq, d_in = x.shape
    group_size = num_heads // num_kv_groups
    
    #assert position_offset == kv_cache["keys"].shape[2], f"{kv_cache['keys'].shape[2]} {position_offset}"
    
    queries = jnp.einsum('bsd,dh->bsh', x, params["W_query"]).reshape(b, seq, num_heads, head_dim).transpose(0,2,1,3)
    keys = jnp.einsum('bsd,dh->bsh', x, params["W_key"]).reshape(b, seq, num_kv_groups, head_dim).transpose(0,2,1,3)
    values = jnp.einsum('bsd,dh->bsh', x, params["W_value"]).reshape(b, seq, num_kv_groups, head_dim).transpose(0,2,1,3)

    if qk_norm and "q_norm" in params and "k_norm" in params:
        queries = apply_qk_norm(queries, params["q_norm"])
        keys = apply_qk_norm(keys, params["k_norm"])

    queries = apply_rope_with_offset(queries, cos, sin, position_offset)
    keys = apply_rope_with_offset(keys, cos, sin, position_offset)
    
    keys = jnp.concatenate([kv_cache["keys"][:,:, :position_offset], keys], axis=2)
    values = jnp.concatenate([kv_cache["values"][:,:,:position_offset], values], axis=2)
    
    kv_cache["keys"] = kv_cache["keys"].at[:,:,:keys.shape[2]].set(keys)
    kv_cache["values"] = kv_cache["values"].at[:,:,:values.shape[2]].set(values)
    new_cache = kv_cache
    position_offset_new = keys.shape[2]
    
    keys_expanded = jnp.repeat(keys, group_size, axis=1)
    values_expanded = jnp.repeat(values, group_size, axis=1)
    
    attn_scores = jnp.einsum('bnqh,bnkh->bnqk', queries, keys_expanded) / jnp.sqrt(head_dim)
    
    if position_offset == 0:
        q_len, k_len = queries.shape[2], keys.shape[2]
        causal_mask = jnp.triu(jnp.ones((q_len, k_len)), k=1)
        attn_scores = jnp.where(causal_mask[None, None, :, :], -jnp.inf, attn_scores)
    
    attn_weights = jax.nn.softmax(attn_scores, axis=-1)
    context = jnp.einsum('bnqk,bnkh->bnqh', attn_weights, values_expanded)
    context = context.transpose(0,2,1,3).reshape(b, seq, num_heads * head_dim)
    output = jnp.einsum('bsh,hd->bsd', context, params["out_proj"])

    return output, new_cache, position_offset_new

def rms2(norm1, x):
    return jax.vmap(lambda y: rmsnorm_forward(norm1, y))(x)

def transformer_block_forward_kv(params, mask, cos, sin, kv_cache, position_offset, x):
    shortcut = x
    x = rmsnorm_forward(params["norm1"], x)
    x, new_cache, position_offset = grouped_query_attention_forward_kv(cfg["n_heads"], cfg["n_kv_groups"], cfg["head_dim"], cos, sin, params["att"], mask,  kv_cache, cfg["qk_norm"], position_offset, x)
    x = x + shortcut
    shortcut = x
    x = rmsnorm_forward(params["norm2"], x)
    x = feedforward_forward(params["ff"], x)
    return x + shortcut, new_cache, position_offset

def transformer_block_forward_kv_pre(params, mask, cos, sin, kv_cache, position_offset, x):
    shortcut = x
    x = rmsnorm_forward(params["norm1"], x)
    x, new_cache, position_offset = grouped_query_attention_forward_kv_pre(cfg["n_heads"], cfg["n_kv_groups"], cfg["head_dim"], cos, sin, params["att"], mask,  kv_cache, cfg["qk_norm"], position_offset, x)
    x = x + shortcut
    shortcut = x
    x = rmsnorm_forward(params["norm2"], x)
    x = feedforward_forward(params["ff"], x)
    return x + shortcut, new_cache, position_offset


def qwen3_forward_kv(params, x, cfg, kv_cache, position_offset):
    x = params["tok_emb"][x]
    mask = jnp.triu(jnp.ones((cfg["context_length"], cfg["context_length"]), dtype=bool), k=1)
    
    new_cache = []
    for i, block_params in enumerate(params["trf_blocks"]):
        layer_cache = kv_cache[i]
        x, updated_cache, position_offset_new = transformer_block_forward_kv(block_params, mask, params["cos"], params["sin"], layer_cache, position_offset, x)
        new_cache.append(updated_cache)
    
    x = rmsnorm_forward(params["final_norm"], x)
    logits = jnp.einsum('bse,ev->bsv', x, params["out_head"])
    
    return logits, new_cache, position_offset_new

def qwen3_forward_kv_pre(params, x, cfg, kv_cache, position_offset):
    x = params["tok_emb"][x]
    mask = jnp.triu(jnp.ones((cfg["context_length"], cfg["context_length"]), dtype=bool), k=1)
    
    new_cache = []
    for i, block_params in enumerate(params["trf_blocks"]):
        layer_cache = kv_cache[i]
        x, updated_cache, position_offset_new = transformer_block_forward_kv_pre(block_params, mask, params["cos"], params["sin"], layer_cache, position_offset, x)
        new_cache.append(updated_cache)
    
    x = rmsnorm_forward(params["final_norm"], x)
    logits = jnp.einsum('bse,ev->bsv', x, params["out_head"])
    
    return logits, new_cache, position_offset_new




def steptop(params, cfg):
    def step(args,_):#logits, kv_cache, position_offset, cur_ids):
        logits, kv_cache, position_offset, cur_ids = args

        next_token_logits = logits[:, -1, :]
        next_token = jnp.argmax(next_token_logits, axis=-1)

        '''
        if eos_id is not None and jnp.any(next_token == eos_id):
            break
        '''
        
        cur_ids = jnp.concatenate([cur_ids, next_token[:, None]], axis=1)
        
        # Process next tokens for entire batch
        logits, kv_cache, position_offset = qwen3_forward_kv(params, next_token[:, None], cfg, kv_cache, position_offset)
        return [logits, kv_cache, position_offset, cur_ids], None
    return step


#scan(step, init=[params, cfg, kv_cache])(next_token, position_offset)

def generate_kv_optimized(model, idx, max_new_tokens, context_size, temperature=0.7, top_k=50, eos_id=None):
    params, cfg = model["params"], model["cfg"]
    cfg.pop('dtype')
    
    # Keep input on device
    cur_ids = jnp.array([idx])
    key = jax.random.PRNGKey(42)
    
    # Initialize KV cache for batch processing
    kv_cache = [{"keys": jnp.zeros((1, cfg["n_kv_groups"], context_size, cfg["head_dim"])), 
                 "values": jnp.zeros((1, cfg["n_kv_groups"], context_size, cfg["head_dim"]))} 
                for _ in range(cfg["n_layers"])]
    position_offset = 0
    
    logits, kv_cache, position_offset = qwen3_forward_kv_pre(params, cur_ids, cfg, kv_cache, position_offset)

    f = steptop(params, cfg)
    logits, kv_cache, position_offset, cur_ids = jax.lax.scan(
            f, 
            init=[logits, kv_cache, position_offset, cur_ids], length=max_new_tokens
            )()


    #for i in tqdm(range(max_new_tokens), desc="Generating"):
    #    [logits, kv_cache, position_offset, cur_ids], _ = f([logits, kv_cache, position_offset, cur_ids], None)

    
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
    
    return cur_ids
