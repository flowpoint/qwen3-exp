import jax
import jax.numpy as jnp
from tokenizers import Tokenizer
import torch
from safetensors.torch import load_file
import os
from pathlib import Path
import gc
from collections import defaultdict

try:
    from huggingface_hub import hf_hub_download, snapshot_download
except ImportError:
    hf_hub_download = None
    snapshot_download = None

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['JAX_PLATFORMS'] = 'gpu'

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
                prompt += "\n"
            else:
                prompt += "<|think>\n\n<|/think>\n\n"
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
        "fc1": jax.device_put(jax.random.normal(k1, (emb_dim, hidden_dim)) / jnp.sqrt(emb_dim), device),
        "fc2": jax.device_put(jax.random.normal(k2, (emb_dim, hidden_dim)) / jnp.sqrt(emb_dim), device),
        "fc3": jax.device_put(jax.random.normal(k3, (hidden_dim, emb_dim)) / jnp.sqrt(hidden_dim), device),
    }
    return params

@jax.jit
def feedforward_forward(params, x):
    x_fc1 = jnp.einsum('bse,eh->bsh', x, params["fc1"])
    x_fc2 = jnp.einsum('bse,eh->bsh', x, params["fc2"])
    x = jax.nn.silu(x_fc1) * x_fc2
    out = jnp.einsum('bsh,he->bse', x, params["fc3"])
    return out

def init_rmsnorm_params(emb_dim, bias=False):
    params = {"scale": jax.device_put(jnp.ones((emb_dim,)), device)}
    if bias:
        params["shift"] = jax.device_put(jnp.zeros((emb_dim,)), device)
    return params

@jax.jit
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

@jax.jit
def apply_rope(x, cos, sin):
    batch, num_heads, seq_len, head_dim = x.shape
    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]
    cos = cos[:seq_len, :][None, None, :, :]
    sin = sin[:seq_len, :][None, None, :, :]
    rotated = jnp.concatenate([-x2, x1], axis=-1)
    x_rotated = (x * cos) + (rotated * sin)
    return x_rotated.astype(x.dtype)

def init_gqa_params(key, d_in, num_heads, num_kv_groups, head_dim):
    kq, kk, kv, ko = jax.random.split(key, 4)
    params = {
        "W_query": jax.device_put(jax.random.normal(kq, (d_in, num_heads * head_dim)) / jnp.sqrt(d_in), device),
        "W_key": jax.device_put(jax.random.normal(kk, (d_in, num_kv_groups * head_dim)) / jnp.sqrt(d_in), device),
        "W_value": jax.device_put(jax.random.normal(kv, (d_in, num_kv_groups * head_dim)) / jnp.sqrt(d_in), device),
        "out_proj": jax.device_put(jax.random.normal(ko, (num_heads * head_dim, d_in)) / jnp.sqrt(num_heads * head_dim), device),
    }
    return params

def grouped_query_attention_forward(params, x, mask, cos, sin, num_heads, num_kv_groups, head_dim, q_norm=None, k_norm=None):
    b, seq, d_in = x.shape
    group_size = num_heads // num_kv_groups

    queries = jnp.einsum('bsd,dh->bsh', x, params["W_query"]).reshape(b, seq, num_heads, head_dim).transpose(0,2,1,3)
    keys = jnp.einsum('bsd,dh->bsh', x, params["W_key"]).reshape(b, seq, num_kv_groups, head_dim).transpose(0,2,1,3)
    values = jnp.einsum('bsd,dh->bsh', x, params["W_value"]).reshape(b, seq, num_kv_groups, head_dim).transpose(0,2,1,3)

    if q_norm is not None:
        queries = q_norm(queries)
    if k_norm is not None:
        keys = k_norm(keys)

    queries = apply_rope(queries, cos, sin)
    keys = apply_rope(keys, cos, sin)

    keys = jnp.repeat(keys, group_size, axis=1)
    values = jnp.repeat(values, group_size, axis=1)

    scale = 1.0 / jnp.sqrt(head_dim)
    attn_scores = jnp.einsum('bnqh,bnkh->bnqk', queries, keys) * scale
    attn_scores = jnp.where(mask, -jnp.inf, attn_scores)
    attn_weights = jax.nn.softmax(attn_scores, axis=-1)
    context = jnp.einsum('bnqk,bnkh->bnqh', attn_weights, values)
    context = context.transpose(0,2,1,3).reshape(b, seq, num_heads * head_dim)
    out = jnp.einsum('bsh,hd->bsd', context, params["out_proj"])
    return out

def init_transformer_block_params(key, cfg):
    k_att, k_ff, k_norm1, k_norm2 = jax.random.split(key, 4)
    params = {
        "att": init_gqa_params(k_att, cfg["emb_dim"], cfg["n_heads"], cfg["n_kv_groups"], cfg["head_dim"]),
        "ff": init_feedforward_params(k_ff, cfg["emb_dim"], cfg["hidden_dim"]),
        "norm1": init_rmsnorm_params(cfg["emb_dim"]),
        "norm2": init_rmsnorm_params(cfg["emb_dim"]),
    }
    return params

def transformer_block_forward(params, x, mask, cos, sin, cfg):
    shortcut = x
    x = rmsnorm_forward(params["norm1"], x)
    x = grouped_query_attention_forward(params["att"], x, mask, cos, sin, num_heads=cfg["n_heads"], num_kv_groups=cfg["n_kv_groups"], head_dim=cfg["head_dim"], q_norm=None, k_norm=None)
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

def qwen3_forward(params, x, cfg):
    tok_emb = params["tok_emb"]
    x = tok_emb[x]
    num_tokens = x.shape[1]
    mask = jnp.triu(jnp.ones((num_tokens, num_tokens), dtype=bool), k=1)
    for block_params in params["trf_blocks"]:
        x = transformer_block_forward(block_params, x, mask, params["cos"], params["sin"], cfg)
    x = rmsnorm_forward(params["final_norm"], x)
    logits = jnp.einsum('bse,ev->bsv', x, params["out_head"])
    return logits

def generate(model, idx, max_new_tokens, context_size=None, top_k=1, temperature=0.0, eos_id=None):
    params = model["params"]
    cfg = model["cfg"]
    
    if context_size is None:
        context_size = cfg["context_length"]
    
    cur_ids = jax.device_put(idx, device)
    
    @jax.jit
    def compiled_forward(params, x):
        return qwen3_forward(params, x, cfg)
    
    for i in range(max_new_tokens):
        idx_cond = cur_ids[:, -context_size:]
        logits = compiled_forward(params, idx_cond)
        next_token_logits = logits[:, -1, :]
        
        if temperature > 0.0:
            next_token_logits = next_token_logits / temperature
            if top_k > 1:
                top_k_logits, top_k_indices = jax.lax.top_k(next_token_logits[0], top_k)
                key = jax.random.PRNGKey(i)
                next_token_idx = jax.random.categorical(key, top_k_logits)
                next_token = top_k_indices[next_token_idx][None]
            else:
                key = jax.random.PRNGKey(i)
                next_token = jax.random.categorical(key, next_token_logits, axis=-1)
        else:
            next_token = jnp.argmax(next_token_logits, axis=-1)
        
        cur_ids = jnp.concatenate([cur_ids, next_token[:, None]], axis=1)
        
        if eos_id is not None and int(next_token[0]) == eos_id:
            break
    
    return cur_ids

def assign_layer_weights(block_params, converted_weights):
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
            block_params["ff"]["fc1"] = tensor.T
        elif key == "mlp.up_proj.weight":
            block_params["ff"]["fc2"] = tensor.T
        elif key == "mlp.down_proj.weight":
            block_params["ff"]["fc3"] = tensor.T

def load_and_convert_file_weights(file_path, jax_params, cfg):
    pt_params = load_file(str(file_path))
    
    file_weights = {}
    layer_weights = defaultdict(dict)
    
    for key, tensor in pt_params.items():
        if key == "model.embed_tokens.weight":
            file_weights["tok_emb"] = tensor
        elif key == "model.norm.weight":
            file_weights["final_norm"] = tensor
        elif key.startswith("model.layers."):
            parts = key.split(".")
            layer_idx = int(parts[2])
            weight_path = ".".join(parts[3:])
            layer_weights[layer_idx][weight_path] = tensor
    
    if file_weights:
        converted_global = batch_convert_weights(file_weights)
        
        if "tok_emb" in converted_global:
            jax_params["tok_emb"] = converted_global["tok_emb"]
            
        if "final_norm" in converted_global:
            jax_params["final_norm"]["scale"] = converted_global["final_norm"]
    
    for layer_idx, weights in layer_weights.items():
        if layer_idx < len(jax_params["trf_blocks"]):
            converted_layer = batch_convert_weights(weights)
            assign_layer_weights(jax_params["trf_blocks"][layer_idx], converted_layer)
    
    del pt_params
    cleanup_memory()

def load_qwen3_weights_jax_optimized(param_config, jax_params, safetensors_files):
    for file_path in safetensors_files:
        load_and_convert_file_weights(file_path, jax_params, param_config)
    
    if jax_params["tok_emb"] is not None:
        jax_params["out_head"] = jax_params["tok_emb"].T
    
    return jax_params

if __name__ == "__main__":
    HF_REPO_ID = "Qwen/Qwen3-0.6B"
    
    # Download model
    model_path = download_model_from_hf(HF_REPO_ID)
    safetensors_files = find_safetensors_files(model_path)
    
    # Initialize tokenizer
    tokenizer_path = model_path / "tokenizer.json"
    if not tokenizer_path.exists():
        tokenizer = Qwen3Tokenizer("tokenizer.json", repo_id=HF_REPO_ID, add_generation_prompt=True)
    else:
        tokenizer = Qwen3Tokenizer(str(tokenizer_path), add_generation_prompt=True)

    # Prepare input
    prompt = "Give me a short introduction to large language models."
    input_ids = tokenizer.encode(prompt)
    if len(input_ids) > QWEN3_CONFIG["context_length"]:
        input_ids = input_ids[:QWEN3_CONFIG["context_length"]]
    input_ids = jax.device_put(jnp.array([input_ids]), device)

    # Initialize model
    cfg = QWEN3_CONFIG
    key = jax.random.PRNGKey(0)
    params = init_qwen3_params(key, cfg)

    if isinstance(safetensors_files, list) and isinstance(safetensors_files[0], str):
        safetensors_files = [Path(f) for f in safetensors_files]
    
    # Load weights
    params = load_qwen3_weights_jax_optimized(cfg, params, safetensors_files)
    
    model = {"params": params, "cfg": cfg}
    input_token_ids = jnp.array([input_ids]) if isinstance(input_ids, list) else input_ids
    
    # Generate text
    output_token_ids = generate(
        model=model, 
        idx=input_token_ids, 
        max_new_tokens=100,
        context_size=QWEN3_CONFIG["context_length"], 
        top_k=50,
        temperature=0.7
    )
    
    output_text = tokenizer.decode(list(output_token_ids[0]))
    print("\n" + "="*50)
    print("GENERATED TEXT:")
    print("="*50)
    print(output_text)
    print("="*50)