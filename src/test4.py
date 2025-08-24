from qwen3 import *
from model import *

QWEN3_CONFIG = cfg
context_size = cfg['context_length']

HF_REPO_ID = "Qwen/Qwen3-0.6B"

model_path = download_model_from_hf(HF_REPO_ID)
safetensors_files = list(Path(model_path).glob("*.safetensors"))
safetensors_files.sort()

tokenizer_path = model_path / "tokenizer.json"
tokenizer = Qwen3Tokenizer(str(tokenizer_path) if tokenizer_path.exists() else "tokenizer.json", repo_id=HF_REPO_ID)

pref_mul = 20_000
#pref_mul = 1
prompt = "Give me a short introduction to large language models."*pref_mul
input_ids = tokenizer.encode(prompt)
if len(input_ids) > QWEN3_CONFIG["context_length"]:
    input_ids = input_ids[:QWEN3_CONFIG["context_length"]]

# Keep input on device from start
input_token_ids = jnp.array(input_ids)

cfg = QWEN3_CONFIG
key = jax.random.PRNGKey(0)
params = init_qwen3_params(key, cfg)
params = load_qwen3_weights_jax_optimized(cfg, params, safetensors_files)
#import pickle
#pickle.dumps(params, 'params.pickle')
model = {"params": params, "cfg": cfg}

params, cfg = model["params"], model["cfg"]
cfg.pop('dtype')
import operator

# Keep input on device
cur_ids = jnp.array([input_ids])
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

# prefill1
#logits, kv_cache, position_offset = qwen3_forward_kv_pre(params, cur_ids, cfg, kv_cache, position_offset)

x = cur_ids

def qwen3_forward_kv_pre(params, x, cfg, kv_cache, position_offset, seq_chunk_size=1):
    """Chunked version of qwen3_forward_kv_pre with sequence dimension processing"""
    x = params["tok_emb"][x]
    x, new_cache, position_offset_new = chunk_seq(cfg['context_length'], seq_chunk_size,params, kv_cache, position_offset,x)
    
    x = rmsnorm_forward(params["final_norm"], x)
    logits = jnp.einsum('bse,ev->bsv', x, params["out_head"])
    #logits = get_logits(cfg, x, params)
    
    return logits, new_cache, position_offset 

'''
a = qwen3_forward_kv_pre_unchunk(params, x, cfg, kv_cache, position_offset)
print(a)
'''
b = qwen3_forward_kv_pre(params, x, cfg, kv_cache, position_offset)
print(b)
'''

print(f" comparison")

for ai, bi in zip(a,b):
    print(ai == bi)
'''
