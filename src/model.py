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

if jax.default_backend() == 'gpu':
    device =  jax.devices('gpu')[0]
else:
    device =  jax.devices('cpu')[0]

@jax.jit
def feedforward_forward(params, x):
    gate = jax.nn.silu(jnp.einsum('bse,eh->bsh', x, params["gate_proj"]))
    up = jnp.einsum('bse,eh->bsh', x, params["up_proj"])
    return jnp.einsum('bsh,he->bse', gate * up, params["down_proj"])

@jax.jit
def rmsnorm_forward(params, x, eps=1e-6):
    orig_dtype = x.dtype
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
    return ((x * cos) + (rotated * sin)).astype(x.dtype)

def apply_rope_with_offset(x, cos, sin, position_offset=0):
    seq_len = x.shape[2]
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    
    positions = jnp.arange(position_offset, position_offset + seq_len)
    cos_slice = cos[positions, :][None, None, :, :]
    sin_slice = sin[positions, :][None, None, :, :]
    
    rotated = jnp.concatenate([-x2, x1], axis=-1)
    return ((x * cos_slice) + (rotated * sin_slice)).astype(x.dtype)

def apply_qk_norm(x, norm_params):
    b, h, s, d = x.shape
    x_reshaped = x.reshape(b * h * s, d)
    x_normed = rmsnorm_forward(norm_params, x_reshaped)
    return x_normed.reshape(b, h, s, d)

def grouped_query_attention_forward_kv(params, x, mask, cos, sin, num_heads, num_kv_groups, head_dim, kv_cache=None, qk_norm=False):
    b, seq, d_in = x.shape
    group_size = num_heads // num_kv_groups
    
    if kv_cache is not None and kv_cache["keys"].shape[2] > 0:
        position_offset = kv_cache["keys"].shape[2]
    else:
        position_offset = 0
    
    queries = jnp.einsum('bsd,dh->bsh', x, params["W_query"]).reshape(b, seq, num_heads, head_dim).transpose(0,2,1,3)
    keys = jnp.einsum('bsd,dh->bsh', x, params["W_key"]).reshape(b, seq, num_kv_groups, head_dim).transpose(0,2,1,3)
    values = jnp.einsum('bsd,dh->bsh', x, params["W_value"]).reshape(b, seq, num_kv_groups, head_dim).transpose(0,2,1,3)

    if qk_norm and "q_norm" in params and "k_norm" in params:
        queries = apply_qk_norm(queries, params["q_norm"])
        keys = apply_qk_norm(keys, params["k_norm"])

    queries = apply_rope_with_offset(queries, cos, sin, position_offset)
    keys = apply_rope_with_offset(keys, cos, sin, position_offset)
    
    if kv_cache is not None and kv_cache["keys"].shape[2] > 0:
        keys = jnp.concatenate([kv_cache["keys"], keys], axis=2)
        values = jnp.concatenate([kv_cache["values"], values], axis=2)
    
    new_cache = {"keys": keys, "values": values}
    
    keys_expanded = jnp.repeat(keys, group_size, axis=1)
    values_expanded = jnp.repeat(values, group_size, axis=1)
    
    attn_scores = jnp.einsum('bnqh,bnkh->bnqk', queries, keys_expanded) / jnp.sqrt(head_dim)
    
    if kv_cache is None or kv_cache["keys"].shape[2] == 0:
        q_len, k_len = queries.shape[2], keys.shape[2]
        causal_mask = jnp.triu(jnp.ones((q_len, k_len)), k=1)
        attn_scores = jnp.where(causal_mask[None, None, :, :], -jnp.inf, attn_scores)
    
    attn_weights = jax.nn.softmax(attn_scores, axis=-1)
    context = jnp.einsum('bnqk,bnkh->bnqh', attn_weights, values_expanded)
    context = context.transpose(0,2,1,3).reshape(b, seq, num_heads * head_dim)
    output = jnp.einsum('bsh,hd->bsd', context, params["out_proj"])
    
    return output, new_cache

def transformer_block_forward_kv(params, x, mask, cos, sin, cfg, kv_cache=None):
    shortcut = x
    x = rmsnorm_forward(params["norm1"], x)
    x, new_cache = grouped_query_attention_forward_kv(params["att"], x, mask, cos, sin, cfg["n_heads"], cfg["n_kv_groups"], cfg["head_dim"], kv_cache, cfg["qk_norm"])
    x = x + shortcut
    shortcut = x
    x = rmsnorm_forward(params["norm2"], x)
    x = feedforward_forward(params["ff"], x)
    return x + shortcut, new_cache


def qwen3_forward_kv(params, x, cfg, kv_cache=None):
    x = params["tok_emb"][x]
    mask = jnp.triu(jnp.ones((cfg["context_length"], cfg["context_length"]), dtype=bool), k=1)
    
    new_cache = []
    for i, block_params in enumerate(params["trf_blocks"]):
        layer_cache = kv_cache[i] if kv_cache else None
        x, updated_cache = transformer_block_forward_kv(block_params, x, mask, params["cos"], params["sin"], cfg, layer_cache)
        new_cache.append(updated_cache)
    
    x = rmsnorm_forward(params["final_norm"], x)
    logits = jnp.einsum('bse,ev->bsv', x, params["out_head"])
    
    return logits, new_cache

def generate_kv_optimized(model, idx, max_new_tokens, context_size, temperature=0.7, top_k=50, eos_id=None, batch_size=1):
    params, cfg = model["params"], model["cfg"]
    
    # Keep input on device
    cur_ids = jnp.array([idx] * batch_size) if batch_size > 1 else jnp.array([idx])
    key = jax.random.PRNGKey(42)
    
    # Initialize KV cache for batch processing
    kv_cache = [{"keys": jnp.zeros((batch_size, cfg["n_kv_groups"], 0, cfg["head_dim"])), 
                 "values": jnp.zeros((batch_size, cfg["n_kv_groups"], 0, cfg["head_dim"]))} 
                for _ in range(cfg["n_layers"])]
    
    logits, kv_cache = qwen3_forward_kv(params, cur_ids, cfg, kv_cache)
    
    for i in tqdm(range(max_new_tokens), desc="Generating"):
        next_token_logits = logits[:, -1, :]
        
        if top_k is not None and top_k > 0:
            # Vectorized top_k for batch processing
            top_k_logits, top_k_indices = jax.lax.top_k(next_token_logits, top_k)
            mask = jnp.full_like(next_token_logits, -jnp.inf)
            mask = jnp.take_along_axis(mask, top_k_indices, axis=-1)
            mask = jnp.where(jnp.arange(mask.shape[-1])[None, :] < top_k, top_k_logits, -jnp.inf)
            next_token_logits = jnp.full_like(next_token_logits, -jnp.inf)
            next_token_logits = next_token_logits.at[jnp.arange(batch_size)[:, None], top_k_indices].set(mask)
        
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
        logits, kv_cache = qwen3_forward_kv(params, next_token[:, None], cfg, kv_cache)
    
    return cur_ids
