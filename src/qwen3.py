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

from model import *
try:
    from huggingface_hub import snapshot_download
except ImportError:
    snapshot_download = None


os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.95'
os.environ['JAX_ENABLE_COMPILATION_CACHE'] = 'true'
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['JAX_PLATFORMS'] = 'gpu'

'''
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_triton_gemm_any=True '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
)
'''

'''
if jax.default_backend() == 'gpu':
    device =  jax.devices('gpu')[0]
else:
    device =  jax.devices('cpu')[0]
'''

#device = jax.devices('gpu')[0] if jax.devices('gpu') else jax.devices('cpu')[0]
dtype = dtype

QWEN3_CONFIG = cfg

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
    return jax.tree.map(lambda x: jax.device_put(x.astype(dtype), device), converted)

def cleanup_memory():
    '''
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    '''
    gc.collect()


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

def init_qwen3_params(key, cfg):
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
        
        if cfg["qk_norm"]:
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

def load_qwen3_weights_jax_optimized(param_config, jax_params, safetensors_files):
    for i, file_path in enumerate(safetensors_files):
        print(f"Loading file {i+1}/{len(safetensors_files)}: {file_path.name}")
        load_and_convert_file_weights(file_path, jax_params, param_config)
        cleanup_memory()
    
    if "lm_head.weight" not in [key for file_path in safetensors_files for key in load_file(str(file_path)).keys()]:
        if jax_params["tok_emb"] is not None:
            jax_params["out_head"] = jax_params["tok_emb"].T
    
    return jax_params

#if __name__ == "__main__":
def run():
    HF_REPO_ID = "Qwen/Qwen3-0.6B"
    
    model_path = download_model_from_hf(HF_REPO_ID)
    safetensors_files = list(Path(model_path).glob("*.safetensors"))
    safetensors_files.sort()
    
    tokenizer_path = model_path / "tokenizer.json"
    tokenizer = Qwen3Tokenizer(str(tokenizer_path) if tokenizer_path.exists() else "tokenizer.json", repo_id=HF_REPO_ID)

    max_new_tokens = 16
    pref_mul = 20_000
    #pref_mul = 200
    #pref_mul = 1
    prompt = "Give me a short introduction to large language models."*pref_mul
    input_ids = tokenizer.encode(prompt)
    if len(input_ids) > QWEN3_CONFIG["context_length"]:
        input_ids = input_ids[:QWEN3_CONFIG["context_length"]]
    
    # Keep input on device from start
    input_token_ids = jnp.array(input_ids)
    
    cfg = QWEN3_CONFIG
    cfg.pop('dtype') # needed because jax dtype object cant be passed to jit
    key = jax.random.PRNGKey(0)
    params = init_qwen3_params(key, cfg)
    params = load_qwen3_weights_jax_optimized(cfg, params, safetensors_files)
    #import pickle
    #pickle.dumps(params, 'params.pickle')
    model = {"params": params, "cfg": cfg}
    
    import time
    start_time = time.time()
    
    # Generate with optimized function (batch_size=1 for single sequence)
    output_token_ids = generate_kv_optimized(
        model=model, idx=input_token_ids, max_new_tokens=max_new_tokens,
        context_size=QWEN3_CONFIG["context_length"], top_k=1,
        temperature=0, eos_id=None
    )

    time.sleep(10)
    
    generation_time = time.time() - start_time
    
    # Only move to CPU at the very end for decoding
    output_text = tokenizer.decode(list(output_token_ids[0]))
    print("\n" + "="*50)
    print("GENERATED TEXT :")
    print("="*50)
    print(output_text)
    print(f"Time taken: {generation_time:.2f}s")
    print("="*50)

def inf():
    HF_REPO_ID = "Qwen/Qwen3-0.6B"
    
    model_path = download_model_from_hf(HF_REPO_ID)
    safetensors_files = list(Path(model_path).glob("*.safetensors"))
    safetensors_files.sort()
    
    tokenizer_path = model_path / "tokenizer.json"
    tokenizer = Qwen3Tokenizer(str(tokenizer_path) if tokenizer_path.exists() else "tokenizer.json", repo_id=HF_REPO_ID)

    pref_mul = 20_000
    #pref_mul = 200
    #pref_mul = 1
    prompt = "Give me a short introduction to large language models."*pref_mul
    input_ids = tokenizer.encode(prompt)
    if len(input_ids) > QWEN3_CONFIG["context_length"]:
        input_ids = input_ids[:QWEN3_CONFIG["context_length"]]
    
    # Keep input on device from start
    input_token_ids = jnp.array(input_ids)
    
    cfg.pop('dtype') # needed because jax dtype object cant be passed to jit
    cfg = QWEN3_CONFIG
    key = jax.random.PRNGKey(0)
    params = init_qwen3_params(key, cfg)
    params = load_qwen3_weights_jax_optimized(cfg, params, safetensors_files)
    #import pickle
    #pickle.dumps(params, 'params.pickle')
    model = {"params": params, "cfg": cfg}
    
    import time
    start_time = time.time()
    
    # Generate with optimized function (batch_size=1 for single sequence)
    kv_cache, compiled_pre, compiled_gen = generate_kv_optimized_programs(
        model=model, idx=input_token_ids, max_new_tokens=20,
        context_size=QWEN3_CONFIG["context_length"], top_k=1,
        temperature=0, eos_id=None
    )
    #tokenizer.tokenizer.enable_padding(length=40960)
    tokenizer.tokenizer.enable_padding(length=128)

    while 1:
        prompt = input('> ')
        #prompt = prompt.ljust(20000)
        input_ids = tokenizer.encode(prompt)
        if len(input_ids) > QWEN3_CONFIG["context_length"]:
            input_ids = input_ids[:QWEN3_CONFIG["context_length"]]
        # Keep input on device from start
        #input_ids = [input_ids + ' '*20000]
        input_token_ids = jnp.array(input_ids)
        cur_ids = jnp.array([input_token_ids])
        output_token_ids, kv_cache = infer(params, cur_ids, cfg, kv_cache, compiled_pre, compiled_gen)
    
        # Only move to CPU at the very end for decoding
        output_text = tokenizer.decode(list(output_token_ids[0]))
        print("\n" + "="*50)
        print("GENERATED TEXT :")
        print("="*50)
        print(output_text)
        #print(f"Time taken: {generation_time:.2f}s")
        print("="*50)


if __name__ == "__main__":
    run()
