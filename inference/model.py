import jax
import jax.numpy as jnp
from tokenizers import Tokenizer
import torch
from safetensors.torch import load_file
import os
from pathlib import Path
import gc
from collections import defaultdict
from tqdm import tqdm

try:
    from huggingface_hub import hf_hub_download, snapshot_download
except ImportError:
    hf_hub_download = None
    snapshot_download = None

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.5'  # Reduce to 50%
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['JAX_PLATFORMS'] = 'gpu'
os.environ['XLA_FLAGS'] = '--xla_gpu_force_compilation_parallelism=1'

if jax.devices('gpu'):
    device = jax.devices('gpu')[0]
else:
    device = jax.devices('cpu')[0]

QWEN3_CONFIG = {
    "vocab_size": 151_936,
    "context_length": 40_960,
    "emb_dim": 1024,
    "n_heads": 16,
    "n_layers": 28,
    "hidden_dim": 3072,
    "head_dim": 128,
    "qk_norm": True,
    "n_kv_groups": 8,
    "rope_base": 1_000_000.0,
    "dtype": torch.bfloat16,
}

class Qwen3Tokenizer():
    def __init__(self, tokenizer_file_path="tokenizer.json", repo_id=None, add_generation_prompt=False, add_thinking=False):
        self.tokenizer_file_path = tokenizer_file_path
        self.add_generation_prompt = add_generation_prompt
        self.add_thinking = add_thinking

        tokenizer_file_path_obj = Path(tokenizer_file_path)
        if not tokenizer_file_path_obj.is_file() and repo_id is not None:
            if hf_hub_download is not None:
                _ = hf_hub_download(
                    repo_id=repo_id,
                    filename=str(tokenizer_file_path_obj.name),
                    local_dir=str(tokenizer_file_path_obj.parent)
                )
        
        self.tokenizer = Tokenizer.from_file(tokenizer_file_path)
     
    def encode(self, prompt):
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.format_qwen_chat(messages, add_generation_prompt=self.add_generation_prompt, add_thinking=self.add_thinking)
        return self.tokenizer.encode(formatted_prompt).ids
                     
    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)
         
    @staticmethod
    def format_qwen_chat(messages, add_generation_prompt=False, add_thinking=False):
        prompt = ""
        for msg in messages:
            prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
        if add_generation_prompt:
            prompt += "<|im_start|>assistant"
            if not add_thinking:
                prompt += "<|think>\n\n<|/think>\n\n"
            else:
                prompt += "\n"    
        return prompt

def download_model_from_hf(repo_id, local_dir="./model_cache"):
    local_dir = Path(local_dir)
    local_dir.mkdir(exist_ok=True)
    
    model_path = snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir / repo_id.replace("/", "_"),
        local_dir_use_symlinks=False
    )
    
    return Path(model_path)

def find_safetensors_files(model_path):
    safetensors_files = list(Path(model_path).glob("*.safetensors"))
    safetensors_files.sort()
    return safetensors_files

def safe_convert_torch_to_jax(torch_tensor):
    if torch_tensor.is_cuda:
        torch_tensor = torch_tensor.cpu()
    
    if torch_tensor.dtype == torch.bfloat16:
        torch_tensor = torch_tensor.to(torch.float32)
    elif torch_tensor.dtype == torch.float16:
        torch_tensor = torch_tensor.to(torch.float32)
    
    numpy_array = torch_tensor.detach().numpy()
    jax_array = jnp.array(numpy_array)
    
    return jax.device_put(jax_array, device)

def batch_convert_weights(torch_weights_dict):
    jax_weights = {}
    for key, tensor in torch_weights_dict.items():
        jax_weights[key] = safe_convert_torch_to_jax(tensor)
    return jax_weights

def cleanup_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def init_feedforward_params(key, emb_dim, hidden_dim):
    k1, k2, k3 = jax.random.split(key, 3)
    params = {
        "gate_proj": jax.device_put(jax.random.normal(k1, (emb_dim, hidden_dim)) / jnp.sqrt(emb_dim), device),
        "up_proj": jax.device_put(jax.random.normal(k2, (emb_dim, hidden_dim)) / jnp.sqrt(emb_dim), device),
        "down_proj": jax.device_put(jax.random.normal(k3, (hidden_dim, emb_dim)) / jnp.sqrt(hidden_dim), device),
    }
    return params

def feedforward_forward(params, x):
    gate = jnp.einsum('bse,eh->bsh', x, params["gate_proj"])
    gate = jax.nn.silu(gate)
    up = jnp.einsum('bse,eh->bsh', x, params["up_proj"])
    out = jnp.einsum('bsh,he->bse', gate * up, params["down_proj"])
    return out

def init_rmsnorm_params(emb_dim, bias=False):
    params = {"scale": jax.device_put(jnp.ones((emb_dim,)), device)}
    if bias:
        params["shift"] = jax.device_put(jnp.zeros((emb_dim,)), device)
    return params

def rmsnorm_forward(params, x, eps=1e-6):
    orig_dtype = x.dtype
    x = x.astype(jnp.float32)
    
    variance = jnp.mean(x ** 2, axis=-1, keepdims=True)
    norm_x = x * jax.lax.rsqrt(variance + eps)
    
    norm_x = norm_x * params["scale"]
    if "shift" in params:
        norm_x = norm_x + params["shift"]
    
    return norm_x.astype(orig_dtype)

def compute_rope_params(head_dim, theta_base=10000.0, context_length=4096, dtype=jnp.float32):
    inv_freq = 1.0 / (theta_base ** (jnp.arange(0, head_dim, 2, dtype=dtype) / head_dim))
    positions = jnp.arange(context_length, dtype=dtype)
    angles = positions[:, None] * inv_freq[None, :]
    angles = jnp.concatenate([angles, angles], axis=1)
    cos = jnp.cos(angles)
    sin = jnp.sin(angles)
    return jax.device_put(cos, device), jax.device_put(sin, device)

def apply_rope(x, cos, sin):
    batch, num_heads, seq_len, head_dim = x.shape
    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]
    cos = cos[:seq_len, :][None, None, :, :]
    sin = sin[:seq_len, :][None, None, :, :]
    rotated = jnp.concatenate([-x2, x1], axis=-1)
    x_rotated = (x * cos) + (rotated * sin)
    return x_rotated.astype(x.dtype)

def init_gqa_params(key, d_in, num_heads, num_kv_groups, head_dim, qk_norm=False):
    kq, kk, kv, ko = jax.random.split(key, 4)
    params = {
        "W_query": jax.device_put(jax.random.normal(kq, (d_in, num_heads * head_dim)) / jnp.sqrt(d_in), device),
        "W_key": jax.device_put(jax.random.normal(kk, (d_in, num_kv_groups * head_dim)) / jnp.sqrt(d_in), device),
        "W_value": jax.device_put(jax.random.normal(kv, (d_in, num_kv_groups * head_dim)) / jnp.sqrt(d_in), device),
        "out_proj": jax.device_put(jax.random.normal(ko, (num_heads * head_dim, d_in)) / jnp.sqrt(num_heads * head_dim), device),
    }
    
    if qk_norm:
        params["q_norm"] = init_rmsnorm_params(head_dim)
        params["k_norm"] = init_rmsnorm_params(head_dim)
    
    return params

def apply_qk_norm(x, norm_params):
    b, h, s, d = x.shape
    x_reshaped = x.reshape(b * h * s, d)
    x_normed = rmsnorm_forward(norm_params, x_reshaped)
    return x_normed.reshape(b, h, s, d)

def grouped_query_attention_forward_simple(params, x, mask, cos, sin, num_heads, num_kv_groups, head_dim, qk_norm=False):
    """Simple attention function without KV caching, similar to PyTorch version."""
    b, seq, d_in = x.shape
    group_size = num_heads // num_kv_groups

    # Compute queries, keys, and values
    queries = jnp.einsum('bsd,dh->bsh', x, params["W_query"]).reshape(b, seq, num_heads, head_dim).transpose(0,2,1,3)
    keys = jnp.einsum('bsd,dh->bsh', x, params["W_key"]).reshape(b, seq, num_kv_groups, head_dim).transpose(0,2,1,3)
    values = jnp.einsum('bsd,dh->bsh', x, params["W_value"]).reshape(b, seq, num_kv_groups, head_dim).transpose(0,2,1,3)

    if qk_norm and "q_norm" in params and "k_norm" in params:
        queries = apply_qk_norm(queries, params["q_norm"])
        keys = apply_qk_norm(keys, params["k_norm"])

    # Apply RoPE
    queries = apply_rope(queries, cos, sin)
    keys = apply_rope(keys, cos, sin)

    # Repeat keys and values for grouped query attention
    keys = jnp.repeat(keys, group_size, axis=1)
    values = jnp.repeat(values, group_size, axis=1)

    # Compute attention
    scale = 1.0 / jnp.sqrt(head_dim)
    attn_scores = jnp.einsum('bnqh,bnkh->bnqk', queries, keys) * scale
    
    # Apply causal mask
    attn_scores = jnp.where(mask, -jnp.inf, attn_scores)
    attn_weights = jax.nn.softmax(attn_scores, axis=-1)
    context = jnp.einsum('bnqk,bnkh->bnqh', attn_weights, values)
    context = context.transpose(0,2,1,3).reshape(b, seq, num_heads * head_dim)
    out = jnp.einsum('bsh,hd->bsd', context, params["out_proj"])
    
    return out

def init_transformer_block_params(key, cfg):
    k_att, k_ff, k_norm1, k_norm2 = jax.random.split(key, 4)
    params = {
        "att": init_gqa_params(k_att, cfg["emb_dim"], cfg["n_heads"], cfg["n_kv_groups"], cfg["head_dim"], cfg["qk_norm"]),
        "ff": init_feedforward_params(k_ff, cfg["emb_dim"], cfg["hidden_dim"]),
        "norm1": init_rmsnorm_params(cfg["emb_dim"]),
        "norm2": init_rmsnorm_params(cfg["emb_dim"]),
    }
    return params

# Simple transformer block without KV caching
def transformer_block_forward_simple(params, x, mask, cos, sin, cfg):
    """Simple transformer block without KV caching, similar to PyTorch version."""
    shortcut = x
    x = rmsnorm_forward(params["norm1"], x)
    
    x = grouped_query_attention_forward_simple(
        params["att"], x, mask, cos, sin, 
        num_heads=cfg["n_heads"], 
        num_kv_groups=cfg["n_kv_groups"], 
        head_dim=cfg["head_dim"], 
        qk_norm=cfg["qk_norm"]
    )
    
    x = x + shortcut

    shortcut = x
    x = rmsnorm_forward(params["norm2"], x)
    x = feedforward_forward(params["ff"], x)
    x = x + shortcut
    
    return x

def init_qwen3_params(key, cfg):
    k_emb, k_blocks, k_final_norm, k_out = jax.random.split(key, 4)
    tok_emb = jax.device_put(jax.random.normal(k_emb, (cfg["vocab_size"], cfg["emb_dim"])) / jnp.sqrt(cfg["vocab_size"]), device)
    block_keys = jax.random.split(k_blocks, cfg["n_layers"])
    trf_blocks = [init_transformer_block_params(k, cfg) for k in block_keys]
    final_norm = init_rmsnorm_params(cfg["emb_dim"])
    out_head = jax.device_put(jax.random.normal(k_out, (cfg["emb_dim"], cfg["vocab_size"])) / jnp.sqrt(cfg["emb_dim"]), device)
    cos, sin = compute_rope_params(head_dim=cfg["head_dim"], theta_base=cfg["rope_base"], context_length=cfg["context_length"], dtype=jnp.float32)
    params = {
        "tok_emb": tok_emb,
        "trf_blocks": trf_blocks,
        "final_norm": final_norm,
        "out_head": out_head,
        "cos": cos,
        "sin": sin,
    }
    return params

# Simple forward pass without KV caching (like PyTorch)
def qwen3_forward_simple(params, x, cfg):
    """Simple forward pass without KV caching, similar to PyTorch version."""
    tok_emb = params["tok_emb"]
    x = tok_emb[x]
    num_tokens = x.shape[1]
    
    # Standard causal mask
    mask = jnp.triu(jnp.ones((num_tokens, num_tokens), dtype=bool), k=1)
    
    for block_params in params["trf_blocks"]:
        x = transformer_block_forward_simple(
            block_params, x, mask, params["cos"], params["sin"], cfg
        )
    
    x = rmsnorm_forward(params["final_norm"], x)
    logits = jnp.einsum('bse,ev->bsv', x, params["out_head"])
    
    return logits

# Simplified generation function like PyTorch
def generate_simple(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    """Simple generation function without KV caching, similar to PyTorch version."""
    params = model["params"]
    cfg = model["cfg"]
    
    cur_ids = jax.device_put(idx, device)
    key = jax.random.PRNGKey(42)
    
    for i in range(max_new_tokens):
        # Use sliding window like PyTorch
        idx_cond = cur_ids[:, -context_size:]
        
        # Get logits
        logits = qwen3_forward_simple(params, idx_cond, cfg)
        next_token_logits = logits[:, -1, :]
        
        # Filter logits with top_k sampling
        if top_k is not None:
            top_k_logits, top_k_indices = jax.lax.top_k(next_token_logits[0], top_k)
            min_val = top_k_logits[-1]
            next_token_logits = jnp.where(next_token_logits < min_val, -jnp.inf, next_token_logits)
        
        # Apply temperature scaling
        if temperature > 0.0:
            next_token_logits = next_token_logits / temperature
            # Apply softmax to get probabilities
            probs = jax.nn.softmax(next_token_logits, axis=-1)
            # Sample from the distribution
            key, subkey = jax.random.split(key)
            next_token = jax.random.categorical(subkey, next_token_logits, axis=-1)
        else:
            # Get the token with highest logits
            next_token = jnp.argmax(next_token_logits, axis=-1)
        
        if eos_id is not None and int(next_token[0]) == eos_id:
            break
        
        # Append sampled token to sequence
        cur_ids = jnp.concatenate([cur_ids, next_token[:, None]], axis=1)
    
    return cur_ids

def assign_layer_weights(block_params, converted_weights, qk_norm=False):
    for key, tensor in converted_weights.items():
        if key == "self_attn.q_proj.weight":
            block_params["att"]["W_query"] = tensor.T
        elif key == "self_attn.k_proj.weight":
            block_params["att"]["W_key"] = tensor.T
        elif key == "self_attn.v_proj.weight":
            block_params["att"]["W_value"] = tensor.T
        elif key == "self_attn.o_proj.weight":
            block_params["att"]["out_proj"] = tensor.T
        elif key == "input_layernorm.weight":
            block_params["norm1"]["scale"] = tensor
        elif key == "post_attention_layernorm.weight":
            block_params["norm2"]["scale"] = tensor
        elif key == "mlp.gate_proj.weight":
            block_params["ff"]["gate_proj"] = tensor.T
        elif key == "mlp.up_proj.weight":
            block_params["ff"]["up_proj"] = tensor.T
        elif key == "mlp.down_proj.weight":
            block_params["ff"]["down_proj"] = tensor.T
        elif key == "self_attn.q_norm.weight" and qk_norm:
            if "q_norm" in block_params["att"]:
                block_params["att"]["q_norm"]["scale"] = tensor
        elif key == "self_attn.k_norm.weight" and qk_norm:
            if "k_norm" in block_params["att"]:
                block_params["att"]["k_norm"]["scale"] = tensor

def load_and_convert_file_weights(file_path, jax_params, cfg):
    pt_params = load_file(str(file_path))
    
    file_weights = {}
    layer_weights = defaultdict(dict)
    
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
            weight_path = ".".join(parts[3:])
            layer_weights[layer_idx][weight_path] = tensor
    
    # Convert and assign global weights
    if file_weights:
        converted_global = batch_convert_weights(file_weights)
        
        if "tok_emb" in converted_global:
            jax_params["tok_emb"] = converted_global["tok_emb"]
            
        if "final_norm" in converted_global:
            jax_params["final_norm"]["scale"] = converted_global["final_norm"]
            
        if "out_head" in converted_global:
            jax_params["out_head"] = converted_global["out_head"].T
    
    # Convert and assign layer weights
    for layer_idx, weights in layer_weights.items():
        if layer_idx < len(jax_params["trf_blocks"]):
            converted_layer = batch_convert_weights(weights)
            assign_layer_weights(jax_params["trf_blocks"][layer_idx], converted_layer, cfg["qk_norm"])
    
    del pt_params
    cleanup_memory()

def load_qwen3_weights_jax_optimized(param_config, jax_params, safetensors_files):
    for i, file_path in enumerate(safetensors_files):
        print(f"Loading file {i+1}/{len(safetensors_files)}: {file_path.name}")
        load_and_convert_file_weights(file_path, jax_params, param_config)
        cleanup_memory()
    
    # Only use tied embeddings if lm_head wasn't loaded
    if "lm_head.weight" not in [key for file_path in safetensors_files 
                                for key in load_file(str(file_path)).keys()]:
        if jax_params["tok_emb"] is not None:
            jax_params["out_head"] = jax_params["tok_emb"].T
    
    return jax_params

def estimate_kv_cache_memory(batch_size, max_seq_len, num_layers, num_kv_groups, head_dim, dtype=jnp.bfloat16):
    """Estimate KV cache memory usage in MB."""
    # Each cache has 2 tensors (K and V) per layer
    # Shape: (batch_size, num_kv_groups, max_seq_len, head_dim)
    elements_per_cache = batch_size * num_kv_groups * max_seq_len * head_dim
    total_elements = elements_per_cache * 2 * num_layers  # 2 for K and V
    
    if dtype == jnp.bfloat16 or dtype == jnp.float16:
        bytes_per_element = 2
    else:  # float32
        bytes_per_element = 4
    
    total_bytes = total_elements * bytes_per_element
    total_mb = total_bytes / (1024 * 1024)
    return total_mb


if __name__ == "__main__":
    HF_REPO_ID = "Qwen/Qwen3-0.6B"
    
    # Download model
    model_path = download_model_from_hf(HF_REPO_ID)
    safetensors_files = find_safetensors_files(model_path)
    
    # Initialize tokenizer
    tokenizer_path = model_path / "tokenizer.json"
    if not tokenizer_path.exists():
        tokenizer = Qwen3Tokenizer("tokenizer.json", repo_id=HF_REPO_ID, add_generation_prompt=True, add_thinking=False)
    else:
        tokenizer = Qwen3Tokenizer(str(tokenizer_path), add_generation_prompt=True, add_thinking=False)

    # Prepare input
    prompt = "Give me a short introduction to large language models."
    input_ids = tokenizer.encode(prompt)
    if len(input_ids) > QWEN3_CONFIG["context_length"]:
        input_ids = input_ids[:QWEN3_CONFIG["context_length"]]
    
  
    
    # Convert to JAX array with proper shape (batch_size, seq_len)
    input_token_ids = jnp.array([input_ids])

    # Initialize model
    cfg = QWEN3_CONFIG
    key = jax.random.PRNGKey(0)
    params = init_qwen3_params(key, cfg)

    if isinstance(safetensors_files, list) and isinstance(safetensors_files[0], str):
        safetensors_files = [Path(f) for f in safetensors_files]
    
    params = load_qwen3_weights_jax_optimized(cfg, params, safetensors_files)
    
    model = {"params": params, "cfg": cfg}
    
    # Test with simple generation (like PyTorch)
 
    import time
    start_time = time.time()
    
    output_token_ids = generate_simple(
        model=model, 
        idx=input_token_ids, 
        max_new_tokens=150,
        context_size=QWEN3_CONFIG["context_length"], 
        top_k=1,
        temperature=0.0,
        eos_id=151645  # <|im_end|> token ID
    )
    
    generation_time = time.time() - start_time
    
    output_text = tokenizer.decode(list(output_token_ids[0]))
    print("\n" + "="*50)
    print("GENERATED TEXT :")
    print("="*50)
    print(output_text)
    print(f"Time taken: {generation_time:.2f}s")
    print("="*50)