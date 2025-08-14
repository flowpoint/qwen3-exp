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
from pdb import set_trace
try:
    from huggingface_hub import snapshot_download
except ImportError:
    snapshot_download = None

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.5'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['JAX_PLATFORMS'] = 'gpu'

device = jax.devices('gpu')[0] if jax.devices('gpu') else jax.devices('cpu')[0]

QWEN3_CONFIG = {
    "vocab_size": 151936, "context_length": 40960, "emb_dim": 1024, "n_heads": 16,
    "n_layers": 28, "hidden_dim": 3072, "head_dim": 128, "qk_norm": True,
    "n_kv_groups": 8, "rope_base": 1000000.0
}
cfg = QWEN3_CONFIG

class Qwen3Tokenizer():
    def __init__(self, tokenizer_file_path="tokenizer.json", repo_id=None):
        if not Path(tokenizer_file_path).is_file() and repo_id and snapshot_download:
            snapshot_download(repo_id=repo_id, local_dir=Path(tokenizer_file_path).parent)
        self.tokenizer = Tokenizer.from_file(tokenizer_file_path)
     
    def encode(self, prompt):
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.format_qwen_chat(messages)
        return self.tokenizer.encode(formatted_prompt).ids
                     
    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)
         
    @staticmethod
    def format_qwen_chat(messages):
        prompt = ""
        for msg in messages:
            prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
        prompt += "<|im_start|>assistant<|think>\n\n<|/think>\n\n"
        return prompt

def download_model_from_hf(repo_id, local_dir="./model_cache"):
    local_dir = Path(local_dir)
    local_dir.mkdir(exist_ok=True)
    model_path = snapshot_download(repo_id=repo_id, local_dir=local_dir / repo_id.replace("/", "_"), local_dir_use_symlinks=False)
    return Path(model_path)

def safe_convert_numpy_to_jax(numpy_array):
    if numpy_array.dtype in [np.float16]:
        numpy_array = numpy_array.astype(np.float32)
    return jnp.array(numpy_array)

def batch_convert_numpy_weights(numpy_weights_dict):
    converted = {key: safe_convert_numpy_to_jax(array) for key, array in numpy_weights_dict.items()}
    return jax.tree.map(lambda x: jax.device_put(x, device), converted)

def cleanup_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def feedforward_forward(params, x):
    gate = jax.nn.silu(jnp.einsum('bse,eh->bsh', x, params["gate_proj"]))
    up = jnp.einsum('bse,eh->bsh', x, params["up_proj"])
    return jnp.einsum('bsh,he->bse', gate * up, params["down_proj"])

def rmsnorm_forward(params, x):
    eps=1e-6
    orig_dtype = jax.numpy.float32 #x.dtype
    x = x.astype(jnp.float32)
    variance = jnp.mean(x ** 2, axis=-1, keepdims=True)
    norm_x = x * jax.lax.rsqrt(variance + eps) * params["scale"]
    return norm_x.astype(jax.numpy.float32)

def compute_rope_params(head_dim, theta_base=10000.0, context_length=4096):
    inv_freq = 1.0 / (theta_base ** (jnp.arange(0, head_dim, 2) / head_dim))
    positions = jnp.arange(context_length)
    angles = jnp.concatenate([positions[:, None] * inv_freq[None, :]] * 2, axis=1)
    return jnp.cos(angles), jnp.sin(angles)

def apply_rope(x, cos, sin):
    cl = cfg['context_length']
    vs = cfg['vocab_size']
    seq_len = cl #x.shape[2]
    x1, x2 = x[..., :vs//2], x[..., vs//2:]
    cos, sin = cos[:seq_len, :][None, None, :, :], sin[:seq_len, :][None, None, :, :]
    rotated = jnp.concatenate([-x2, x1], axis=-1)
    return ((x * cos) + (rotated * sin)).astype(jax.numpy.float32)

def apply_rope_with_offset(x, cos, sin, position_offset):
    cl = cfg['context_length']
    vs = cfg['vocab_size']
    seq_len = x.shape[2]
    x1, x2 = x[..., :vs//2], x[..., vs//2:]
    
    positions = jnp.arange(position_offset, position_offset + seq_len)
    cos_slice = cos[positions, :][None, None, :, :]
    sin_slice = sin[positions, :][None, None, :, :]
    
    rotated = jnp.concatenate([-x2, x1], axis=-1)
    return ((x * cos_slice) + (rotated * sin_slice)).astype(jax.numpy.float32)

def apply_qk_norm(x, norm_params):
    b, h, s, d = x.shape
    x_reshaped = x.reshape(b * h * s, d)
    x_normed = rmsnorm_forward(norm_params, x_reshaped)
    return x_normed.reshape(b, h, s, d)

@jax.jit
def grouped_query_attention_forward_kv(params, x, cos, sin,  kv_cache):
    num_heads, num_kv_groups, head_dim = cfg["n_heads"], cfg["n_kv_groups"], cfg["head_dim"]
    #set_trace()

    cl = cfg['context_length']
    vs = cfg['vocab_size']


    b = 1
    seq = cl #x.shape[2]
    d_in = vs

    b, seq, d_in = x.shape

    group_size = num_heads // num_kv_groups
    
    #if kv_cache is not None and kv_cache["keys"].shape[2] > 0:
    #position_offset = kv_cache["keys"].shape[2]
    #else:
    position_offset = 0
    
    queries = jnp.einsum('bsd,dh->bsh', x, params["W_query"]).reshape(b, seq, cfg['n_heads'], cfg['head_dim']).transpose(0,2,1,3)
    keys = jnp.einsum('bsd,dh->bsh', x, params["W_key"]).reshape(b, seq, cfg['n_kv_groups'], cfg['head_dim']).transpose(0,2,1,3)
    values = jnp.einsum('bsd,dh->bsh', x, params["W_value"]).reshape(b, seq, cfg['n_kv_groups'], cfg['head_dim']).transpose(0,2,1,3)

    queries = apply_qk_norm(queries, params["q_norm"])
    keys = apply_qk_norm(keys, params["k_norm"])

    queries = apply_rope_with_offset(queries, cos, sin, position_offset)
    keys = apply_rope_with_offset(keys, cos, sin, position_offset)
    
    #if kv_cache is not None and kv_cache["keys"].shape[2] > 0:
    keys = jnp.concatenate([kv_cache["keys"], keys], axis=2)
    values = jnp.concatenate([kv_cache["values"], values], axis=2)
    
    new_cache = {"keys": keys, "values": values}
    
    keys_expanded = jnp.repeat(keys, group_size, axis=1)
    values_expanded = jnp.repeat(values, group_size, axis=1)
    
    attn_scores = jnp.einsum('bnqh,bnkh->bnqk', queries, keys_expanded) / jnp.sqrt(head_dim)
    
    #if kv_cache is None or kv_cache["keys"].shape[2] == 0:
    q_len, k_len = queries.shape[2], keys.shape[2]
    causal_mask = jnp.triu(jnp.ones((q_len, k_len)), k=1)
    attn_scores = jnp.where(causal_mask[None, None, :, :], -jnp.inf, attn_scores)
    
    attn_weights = jax.nn.softmax(attn_scores, axis=-1)
    context = jnp.einsum('bnqk,bnkh->bnqh', attn_weights, values_expanded)
    context = context.transpose(0,2,1,3).reshape(b, seq, num_heads * head_dim)
    output = jnp.einsum('bsh,hd->bsd', context, params["out_proj"])
    
    return output, new_cache

#@jax.jit
def transformer_block_forward_kv(params, x, cos, sin, kv_cache):
    shortcut = x
    x = rmsnorm_forward(params["norm1"], x)
    x, new_cache = grouped_query_attention_forward_kv(params["att"], x, cos, sin,  kv_cache)
    x = x + shortcut
    shortcut = x
    x = rmsnorm_forward(params["norm2"], x)
    x = feedforward_forward(params["ff"], x)
    return x + shortcut, new_cache

def init_qwen3_params(key):
    k_emb, k_blocks, k_final_norm, k_out = jax.random.split(key, 4)
    tok_emb = jax.random.normal(k_emb, (cfg["vocab_size"], cfg["emb_dim"])) / jnp.sqrt(cfg["vocab_size"])
    block_keys = jax.random.split(k_blocks, cfg["n_layers"])
    
    def init_block_params(k):
        k_att, k_ff, k_norm1, k_norm2 = jax.random.split(k, 4)
        kq, kk, kv, ko = jax.random.split(k_att, 4)
        k_gate, k_up, k_down = jax.random.split(k_ff, 3)
        
        att_params = {
            "W_query": jax.random.normal(kq, (cfg["emb_dim"], cfg["n_heads"] * cfg["head_dim"])) / jnp.sqrt(cfg["emb_dim"]),
            "W_key": jax.random.normal(kk, (cfg["emb_dim"], cfg["n_kv_groups"] * cfg["head_dim"])) / jnp.sqrt(cfg["emb_dim"]),
            "W_value": jax.random.normal(kv, (cfg["emb_dim"], cfg["n_kv_groups"] * cfg["head_dim"])) / jnp.sqrt(cfg["emb_dim"]),
            "out_proj": jax.random.normal(ko, (cfg["n_heads"] * cfg["head_dim"], cfg["emb_dim"])) / jnp.sqrt(cfg["n_heads"] * cfg["head_dim"]),
        }
        
        att_params["q_norm"] = {"scale": jnp.ones((cfg["head_dim"],))}
        att_params["k_norm"] = {"scale": jnp.ones((cfg["head_dim"],))}
        
        return {
            "att": att_params,
            "ff": {
                "gate_proj": jax.random.normal(k_gate, (cfg["emb_dim"], cfg["hidden_dim"])) / jnp.sqrt(cfg["emb_dim"]),
                "up_proj": jax.random.normal(k_up, (cfg["emb_dim"], cfg["hidden_dim"])) / jnp.sqrt(cfg["emb_dim"]),
                "down_proj": jax.random.normal(k_down, (cfg["hidden_dim"], cfg["emb_dim"])) / jnp.sqrt(cfg["hidden_dim"]),
            },
            "norm1": {"scale": jnp.ones((cfg["emb_dim"],))},
            "norm2": {"scale": jnp.ones((cfg["emb_dim"],))},
        }
    
    trf_blocks = [init_block_params(k) for k in block_keys]
    final_norm = {"scale": jnp.ones((cfg["emb_dim"],))}
    out_head = jax.random.normal(k_out, (cfg["emb_dim"], cfg["vocab_size"])) / jnp.sqrt(cfg["emb_dim"])
    cos, sin = compute_rope_params(cfg["head_dim"], cfg["rope_base"], cfg["context_length"])
    
    params = {"tok_emb": tok_emb, "trf_blocks": trf_blocks, "final_norm": final_norm, "out_head": out_head, "cos": cos, "sin": sin}
    
    return jax.tree.map(lambda x: jax.device_put(x, device), params)

def qwen3_forward_kv(params, x, kv_cache):
    x = params["tok_emb"][x]
    #mask = jnp.triu(jnp.ones((cfg["context_length"], cfg["context_length"]), dtype=bool), k=1)
    
    new_cache = []
    for i, block_params in enumerate(params["trf_blocks"]):
        layer_cache = kv_cache[i]
        x, updated_cache = transformer_block_forward_kv(block_params, x, params["cos"], params["sin"], layer_cache)
        new_cache.append(updated_cache)
    
    x = rmsnorm_forward(params["final_norm"], x)
    logits = jnp.einsum('bse,ev->bsv', x, params["out_head"])
    
    return logits, new_cache

def generate_kv_optimized(model, idx, max_new_tokens, context_size, temperature=0.7, top_k=50, eos_id=None, batch_size=1):
    params = model["params"]
    batch_size = 1
    
    # Keep input on device
    cur_ids = jnp.array([idx] * batch_size)
    key = jax.random.PRNGKey(42)
    
    # Initialize KV cache for batch processing
    kv_cache = [{"keys": jnp.zeros((batch_size, cfg["n_kv_groups"], 0, cfg["head_dim"])), 
                 "values": jnp.zeros((batch_size, cfg["n_kv_groups"], 0, cfg["head_dim"]))} 
                for _ in range(cfg["n_layers"])]
    
    logits, kv_cache = qwen3_forward_kv(params, cur_ids, kv_cache)
    
    for i in tqdm(range(max_new_tokens), desc="Generating"):
        next_token_logits = logits[:, -1, :]
        
        # Vectorized top_k for batch processing
        top_k_logits, top_k_indices = jax.lax.top_k(next_token_logits, top_k)
        mask = jnp.full_like(next_token_logits, -jnp.inf)
        mask = jnp.take_along_axis(mask, top_k_indices, axis=-1)
        mask = jnp.where(jnp.arange(mask.shape[-1])[None, :] < top_k, top_k_logits, -jnp.inf)
        next_token_logits = jnp.full_like(next_token_logits, -jnp.inf)
        next_token_logits = next_token_logits.at[jnp.arange(batch_size)[:, None], top_k_indices].set(mask)
        
        next_token = jnp.argmax(next_token_logits, axis=-1)
        
        # Check EOS for all sequences in batch - keep on device
        if eos_id is not None and jnp.any(next_token == eos_id):
            break
        
        cur_ids = jnp.concatenate([cur_ids, next_token[:, None]], axis=1)
        
        # Process next tokens for entire batch
        logits, kv_cache = qwen3_forward_kv(params, next_token[:, None],kv_cache)
    
    return cur_ids

def assign_layer_weights(block_params, converted_weights):
    weight_map = {
        "self_attn.q_proj.weight": ("att", "W_query", True),
        "self_attn.k_proj.weight": ("att", "W_key", True),
        "self_attn.v_proj.weight": ("att", "W_value", True),
        "self_attn.o_proj.weight": ("att", "out_proj", True),
        "input_layernorm.weight": ("norm1", "scale", False),
        "post_attention_layernorm.weight": ("norm2", "scale", False),
        "mlp.gate_proj.weight": ("ff", "gate_proj", True),
        "mlp.up_proj.weight": ("ff", "up_proj", True),
        "mlp.down_proj.weight": ("ff", "down_proj", True),
    }
    
    #if qk_norm:
    weight_map.update({
        "self_attn.q_norm.weight": ("att", "q_norm", "scale", False),
        "self_attn.k_norm.weight": ("att", "k_norm", "scale", False),
    })
    
    for key, tensor in converted_weights.items():
        if key in weight_map:
            if len(weight_map[key]) == 3:
                section, param, transpose = weight_map[key]
                block_params[section][param] = tensor.T if transpose else tensor
            elif len(weight_map[key]) == 4:
                section, subsection, param, transpose = weight_map[key]
                if subsection in block_params[section]:
                    block_params[section][subsection][param] = tensor.T if transpose else tensor

def load_and_convert_file_weights(file_path, jax_params):
    pt_params = load_file(str(file_path))
    file_weights, layer_weights = {}, defaultdict(dict)
    
    for key, tensor in pt_params.items():
        if key == "model.embed_tokens.weight":
            file_weights["tok_emb"] = tensor
        elif key == "model.norm.weight":
            file_weights["final_norm"] = tensor
        elif key == "lm_head.weight":
            file_weights["out_head"] = tensor
        elif key.startswith("model.layers."):
            parts = key.split(".")
            layer_idx = int(parts[2])
            layer_weights[layer_idx][".".join(parts[3:])] = tensor
    
    if file_weights:
        converted_global = batch_convert_numpy_weights(file_weights)
        if "tok_emb" in converted_global:
            jax_params["tok_emb"] = converted_global["tok_emb"]
        if "final_norm" in converted_global:
            jax_params["final_norm"]["scale"] = converted_global["final_norm"]
        if "out_head" in converted_global:
            jax_params["out_head"] = converted_global["out_head"].T
    
    for layer_idx, weights in layer_weights.items():
        if layer_idx < len(jax_params["trf_blocks"]):
            converted_layer = batch_convert_numpy_weights(weights)
            assign_layer_weights(jax_params["trf_blocks"][layer_idx], converted_layer)
    
    del pt_params
    cleanup_memory()

def load_qwen3_weights_jax_optimized(jax_params, safetensors_files):
    for i, file_path in enumerate(safetensors_files):
        print(f"Loading file {i+1}/{len(safetensors_files)}: {file_path.name}")
        load_and_convert_file_weights(file_path, jax_params)
        cleanup_memory()
    
    if "lm_head.weight" not in [key for file_path in safetensors_files for key in load_file(str(file_path)).keys()]:
        if jax_params["tok_emb"] is not None:
            jax_params["out_head"] = jax_params["tok_emb"].T
    
    return jax_params

if __name__ == "__main__":
    HF_REPO_ID = "Qwen/Qwen3-0.6B"
    
    model_path = download_model_from_hf(HF_REPO_ID)
    safetensors_files = list(Path(model_path).glob("*.safetensors"))
    safetensors_files.sort()
    
    tokenizer_path = model_path / "tokenizer.json"
    tokenizer = Qwen3Tokenizer(str(tokenizer_path) if tokenizer_path.exists() else "tokenizer.json", repo_id=HF_REPO_ID)

    prompt = "Give me a short introduction to large language models."
    input_ids = tokenizer.encode(prompt)
    if len(input_ids) > QWEN3_CONFIG["context_length"]:
        input_ids = input_ids[:QWEN3_CONFIG["context_length"]]
    
    # Keep input on device from start
    input_token_ids = jnp.array(input_ids)
    
    key = jax.random.PRNGKey(0)
    params = init_qwen3_params(key)
    params = load_qwen3_weights_jax_optimized(params, safetensors_files)
    model = {"params": params}
    
    import time
    start_time = time.time()
    
    # Generate with optimized function (batch_size=1 for single sequence)
    output_token_ids = generate_kv_optimized(
        model=model, idx=input_token_ids, max_new_tokens=50,
        context_size=QWEN3_CONFIG["context_length"], top_k=1,
        temperature=0, eos_id=None, batch_size=1
    )
    
    generation_time = time.time() - start_time
    
    # Only move to CPU at the very end for decoding
    output_text = tokenizer.decode(list(output_token_ids[0]))
    print("\n" + "="*20)
    print("GENERATED TEXT :")
    print("="*50)
    print(output_text)
    print(f"Time taken: {generation_time:.2f}s")
    print("="*50)
