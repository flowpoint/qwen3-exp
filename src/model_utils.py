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

from functools import partial, reduce

dtype = jnp.float32

def rms2(norm1, x):
    return jax.vmap(lambda y: rmsnorm_forward(norm1, y))(x)

def feedforward_forward(params, x):
    gate = jax.nn.silu(jnp.einsum('bse,eh->bsh', x, params["gate_proj"]))
    up = jnp.einsum('bse,eh->bsh', x, params["up_proj"])
    return jnp.einsum('bsh,he->bse', gate * up, params["down_proj"])

def rmsnorm_forward(params, x, eps=1e-6):
    orig_dtype = dtype
    x = x.astype(dtype)
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

def apply_rope_with_offset(x, cos, sin, position_offset=0):
    seq_len = x.shape[2]
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    
    positions = jnp.arange(0, 0 + seq_len) + position_offset
    cos_slice = cos[positions, :][None, None, :, :]
    sin_slice = sin[positions, :][None, None, :, :]
    
    rotated = jnp.concatenate([-x2, x1], axis=-1)
    return ((x * cos_slice) + (rotated * sin_slice)).astype(dtype)

def apply_qk_norm(x, norm_params):
    h, s, d = x.shape
    x_reshaped = x.reshape(h * s, d)
    x_normed = rmsnorm_forward(norm_params, x_reshaped)
    return x_normed.reshape(h, s, d)

