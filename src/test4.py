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

#pref_mul = 20_000
pref_mul = 1
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
position_offset = 0

# prefill1
logits2, kv_cache2, position_offset2 = qwen3_forward_kv(params, cur_ids, cfg, kv_cache, position_offset,pre=True)

'''
logits, kv_cache, position_offset = qwen3_forward_kv(params, cur_ids, cfg, kv_cache, position_offset,pre=False)
#for i in range(cur_ids.shape[1]):
#logits, kv_cache, position_offset = qwen3_forward_kv(params, cur_ids, cfg, kv_cache, position_offset)
print('---')
print(kv_cache2 == kv_cache)
print(logits2 == logits)
print(position_offset2 == position_offset)
'''

'''
x = cur_ids

a = qwen3_forward_kv_pre_unchunk(params, x, cfg, kv_cache, position_offset)
print(a)
b = qwen3_forward_kv_pre(params, x, cfg, kv_cache, position_offset)
print(b)

'''

#def grouped_query_attention_forward_kv(num_heads, num_kv_groups, head_dim, cos, sin, params, kv_cache, qk_norm, position_offset, x):

num_heads = cfg['n_heads']
num_kv_groups = cfg['n_kv_groups']
head_dim = cfg['head_dim']
cos, sin = params['cos'], params['sin']
qk_norm = True
x = jnp.ones([1,26],dtype=jnp.int64)
x = params["tok_emb"][x]

attn_params = params['trf_blocks'][0]['att']
print('-----')

layer_cache = {"keys":kv_cache['keys'][:,0], "values":kv_cache["values"][:,0]}

out1 = grouped_query_attention_forward_kv_pre(num_heads, num_kv_groups, head_dim, cos, sin, attn_params, layer_cache, qk_norm, position_offset, x)

out2 = grouped_query_attention_forward_kv(num_heads, num_kv_groups, head_dim, cos, sin, attn_params, layer_cache, qk_norm, position_offset, x)

out1 == out2

#grouped_query_attention_forward_kv(cfg["n_heads"], cfg["n_kv_groups"], cfg["head_dim"], cos, sin, params["att"], kv_cache, cfg["qk_norm"], position_offset, x)
