# qwen3 jax implementation

> [!IMPORTANT] 
>  a clean, pure jax implementation of Qwen3-0.6B for inference with optimized memory usage and KV caching.

## output 

![img](output.png)

## features

- pure jax implementation with no PyTorch dependencies for inference
- optimized KV caching for efficient text generation

## start here

```bash
# Clone the repository
git clone <your-repo-url>
cd qwen3-exp

# install dependencies
pip install -U "jax[cuda12]" tokenizers torch safetensors huggingface-hub tqdm numpy

# run inference
python src/qwen3.py
```

## usage

The implementation automatically downloads the Qwen3-0.6B model from Hugging Face and runs inference:

```python
from qwen3 import Qwen3Tokenizer, generate_kv_optimized, load_qwen3_weights_jax_optimized

# initialize tokenizer and model
tokenizer = Qwen3Tokenizer(repo_id="Qwen/Qwen3-0.6B")
model = load_model()  # Loads and converts weights to JAX

# generate text
prompt = "Give me a short introduction to large language models."
output = generate_kv_optimized(model, prompt, max_new_tokens=50)
```

## requirements

- Python 3.8+
- JAX/JAXLib (GPU support recommended)
- tokenizers
- safetensors
- huggingface-hub
- numpy
- tqdm

## license

MIT License - see [LICENSE](LICENSE) for details.

## reference

Based on the implementation from [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/11_qwen3) by Sebastian Raschka.