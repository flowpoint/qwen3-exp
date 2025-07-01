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
    "n_kv_groups": 8, "rope_base": 1000000.0, "dtype": torch.bfloat16,
}

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
    return jax.device_put(jnp.array(numpy_array), device)


def batch_convert_numpy_weights(numpy_weights_dict):
    return {key: safe_convert_numpy_to_jax(array) for key, array in numpy_weights_dict.items()}


def cleanup_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def feedforward_forward(params, x):
    gate = jax.nn.silu(jnp.einsum('bse,eh->bsh', x, params["gate_proj"]))
    up = jnp.einsum('bse,eh->bsh', x, params["up_proj"])
    return jnp.einsum('bsh,he->bse', gate * up, params["down_proj"])

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
    return jax.device_put(jnp.cos(angles), device), jax.device_put(jnp.sin(angles), device)

def apply_rope(x, cos, sin):
    seq_len = x.shape[2]
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    cos, sin = cos[:seq_len, :][None, None, :, :], sin[:seq_len, :][None, None, :, :]
    rotated = jnp.concatenate([-x2, x1], axis=-1)
    return ((x * cos) + (rotated * sin)).astype(x.dtype)

def apply_qk_norm(x, norm_params):
    b, h, s, d = x.shape
    x_reshaped = x.reshape(b * h * s, d)
    x_normed = rmsnorm_forward(norm_params, x_reshaped)
    return x_normed.reshape(b, h, s, d)

def grouped_query_attention_forward_simple(params, x, mask, cos, sin, num_heads, num_kv_groups, head_dim, qk_norm=False):
    b, seq, d_in = x.shape
    group_size = num_heads // num_kv_groups
    
    queries = jnp.einsum('bsd,dh->bsh', x, params["W_query"]).reshape(b, seq, num_heads, head_dim).transpose(0,2,1,3)
    keys = jnp.einsum('bsd,dh->bsh', x, params["W_key"]).reshape(b, seq, num_kv_groups, head_dim).transpose(0,2,1,3)
    values = jnp.einsum('bsd,dh->bsh', x, params["W_value"]).reshape(b, seq, num_kv_groups, head_dim).transpose(0,2,1,3)

    if qk_norm and "q_norm" in params and "k_norm" in params:
        queries = apply_qk_norm(queries, params["q_norm"])
        keys = apply_qk_norm(keys, params["k_norm"])

    queries, keys = apply_rope(queries, cos, sin), apply_rope(keys, cos, sin)
    keys, values = jnp.repeat(keys, group_size, axis=1), jnp.repeat(values, group_size, axis=1)
    
    attn_scores = jnp.einsum('bnqh,bnkh->bnqk', queries, keys) / jnp.sqrt(head_dim)
    attn_scores = jnp.where(mask, -jnp.inf, attn_scores)
    attn_weights = jax.nn.softmax(attn_scores, axis=-1)
    context = jnp.einsum('bnqk,bnkh->bnqh', attn_weights, values)
    context = context.transpose(0,2,1,3).reshape(b, seq, num_heads * head_dim)
    return jnp.einsum('bsh,hd->bsd', context, params["out_proj"])

def transformer_block_forward_simple(params, x, mask, cos, sin, cfg):
    shortcut = x
    x = rmsnorm_forward(params["norm1"], x)
    x = grouped_query_attention_forward_simple(params["att"], x, mask, cos, sin, cfg["n_heads"], cfg["n_kv_groups"], cfg["head_dim"], cfg["qk_norm"])
    x = x + shortcut
    shortcut = x
    x = rmsnorm_forward(params["norm2"], x)
    x = feedforward_forward(params["ff"], x)
    return x + shortcut

def init_qwen3_params(key, cfg):
    k_emb, k_blocks, k_final_norm, k_out = jax.random.split(key, 4)
    tok_emb = jax.device_put(jax.random.normal(k_emb, (cfg["vocab_size"], cfg["emb_dim"])) / jnp.sqrt(cfg["vocab_size"]), device)
    block_keys = jax.random.split(k_blocks, cfg["n_layers"])
    
    def init_block_params(k):
        k_att, k_ff, k_norm1, k_norm2 = jax.random.split(k, 4)
        kq, kk, kv, ko = jax.random.split(k_att, 4)
        k_gate, k_up, k_down = jax.random.split(k_ff, 3)
        
        att_params = {
            "W_query": jax.device_put(jax.random.normal(kq, (cfg["emb_dim"], cfg["n_heads"] * cfg["head_dim"])) / jnp.sqrt(cfg["emb_dim"]), device),
            "W_key": jax.device_put(jax.random.normal(kk, (cfg["emb_dim"], cfg["n_kv_groups"] * cfg["head_dim"])) / jnp.sqrt(cfg["emb_dim"]), device),
            "W_value": jax.device_put(jax.random.normal(kv, (cfg["emb_dim"], cfg["n_kv_groups"] * cfg["head_dim"])) / jnp.sqrt(cfg["emb_dim"]), device),
            "out_proj": jax.device_put(jax.random.normal(ko, (cfg["n_heads"] * cfg["head_dim"], cfg["emb_dim"])) / jnp.sqrt(cfg["n_heads"] * cfg["head_dim"]), device),
        }
        
        if cfg["qk_norm"]:
            att_params["q_norm"] = {"scale": jax.device_put(jnp.ones((cfg["head_dim"],)), device)}
            att_params["k_norm"] = {"scale": jax.device_put(jnp.ones((cfg["head_dim"],)), device)}
        
        return {
            "att": att_params,
            "ff": {
                "gate_proj": jax.device_put(jax.random.normal(k_gate, (cfg["emb_dim"], cfg["hidden_dim"])) / jnp.sqrt(cfg["emb_dim"]), device),
                "up_proj": jax.device_put(jax.random.normal(k_up, (cfg["emb_dim"], cfg["hidden_dim"])) / jnp.sqrt(cfg["emb_dim"]), device),
                "down_proj": jax.device_put(jax.random.normal(k_down, (cfg["hidden_dim"], cfg["emb_dim"])) / jnp.sqrt(cfg["hidden_dim"]), device),
            },
            "norm1": {"scale": jax.device_put(jnp.ones((cfg["emb_dim"],)), device)},
            "norm2": {"scale": jax.device_put(jnp.ones((cfg["emb_dim"],)), device)},
        }
    
    trf_blocks = [init_block_params(k) for k in block_keys]
    final_norm = {"scale": jax.device_put(jnp.ones((cfg["emb_dim"],)), device)}
    out_head = jax.device_put(jax.random.normal(k_out, (cfg["emb_dim"], cfg["vocab_size"])) / jnp.sqrt(cfg["emb_dim"]), device)
    cos, sin = compute_rope_params(cfg["head_dim"], cfg["rope_base"], cfg["context_length"])
    
    return {"tok_emb": tok_emb, "trf_blocks": trf_blocks, "final_norm": final_norm, "out_head": out_head, "cos": cos, "sin": sin}

def qwen3_forward_simple(params, x, cfg):
    x = params["tok_emb"][x]
    mask = jnp.triu(jnp.ones((x.shape[1], x.shape[1]), dtype=bool), k=1)
    
    for block_params in params["trf_blocks"]:
        x = transformer_block_forward_simple(block_params, x, mask, params["cos"], params["sin"], cfg)
    
    x = rmsnorm_forward(params["final_norm"], x)
    return jnp.einsum('bse,ev->bsv', x, params["out_head"])

def generate_simple(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    params, cfg = model["params"], model["cfg"]
    cur_ids = jax.device_put(idx, device)
    key = jax.random.PRNGKey(42)
    
    for i in tqdm(range(max_new_tokens),desc="Generating tokens"):
        idx_cond = cur_ids[:, -context_size:]
        logits = qwen3_forward_simple(params, idx_cond, cfg)
        next_token_logits = logits[:, -1, :]
        
        if top_k is not None:
            top_k_logits, _ = jax.lax.top_k(next_token_logits[0], top_k)
            next_token_logits = jnp.where(next_token_logits < top_k_logits[-1], -jnp.inf, next_token_logits)
        
        if temperature > 0.0:
            next_token_logits = next_token_logits / temperature
            key, subkey = jax.random.split(key)
            next_token = jax.random.categorical(subkey, next_token_logits, axis=-1)
        else:
            next_token = jnp.argmax(next_token_logits, axis=-1)
        
        if eos_id is not None and int(next_token[0]) == eos_id:
            break
        
        cur_ids = jnp.concatenate([cur_ids, next_token[:, None]], axis=1)
    
    return cur_ids

def assign_layer_weights(block_params, converted_weights, qk_norm=False):
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
    
    if qk_norm:
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

def load_and_convert_file_weights(file_path, jax_params, cfg):
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
            assign_layer_weights(jax_params["trf_blocks"][layer_idx], converted_layer, cfg["qk_norm"])
    
    del pt_params
    cleanup_memory()

def load_qwen3_weights_jax_optimized(param_config, jax_params, safetensors_files):
    for i, file_path in enumerate(safetensors_files):
        print(f"Loading file {i+1}/{len(safetensors_files)}: {file_path.name}")
        load_and_convert_file_weights(file_path, jax_params, param_config)
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
    
    input_token_ids = jnp.array([input_ids])
    
    cfg = QWEN3_CONFIG
    key = jax.random.PRNGKey(0)
    params = init_qwen3_params(key, cfg)
    params = load_qwen3_weights_jax_optimized(cfg, params, safetensors_files)
    model = {"params": params, "cfg": cfg}
    
    import time
    start_time = time.time()
    
    output_token_ids = generate_simple(
        model=model, idx=input_token_ids, max_new_tokens=50,
        context_size=QWEN3_CONFIG["context_length"], top_k=1,
        temperature=0.0, eos_id=151645
    )
    
    generation_time = time.time() - start_time
    output_text = tokenizer.decode(list(output_token_ids[0]))
    print("\n" + "="*50)
    print("GENERATED TEXT :")
    print("="*50)
    print(output_text)
    print(f"Time taken: {generation_time:.2f}s")
    print("="*50)