import pytest
import jax
import jax.numpy as jnp
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
from pathlib import Path
import os
from qwen3 import Qwen3Tokenizer, download_model_from_hf, load_qwen3_weights_jax_optimized, init_qwen3_params, generate_kv_optimized, QWEN3_CONFIG

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

def test_qwen3_jax_vs_hf():
    HF_REPO_ID = "Qwen/Qwen3-0.6B"

    # Download model and tokenizer
    model_path = download_model_from_hf(HF_REPO_ID)
    safetensors_files = list(Path(model_path).glob("*.safetensors"))
    safetensors_files.sort()

    tokenizer_path = model_path / "tokenizer.json"
    tokenizer = Qwen3Tokenizer(str(tokenizer_path) if tokenizer_path.exists() else "tokenizer.json", repo_id=HF_REPO_ID)

    # Prompt for testing
    prompt = "Give me a short introduction to large language models."
    input_ids = tokenizer.encode(prompt)
    if len(input_ids) > QWEN3_CONFIG["context_length"]:
        input_ids = input_ids[:QWEN3_CONFIG["context_length"]]

    # HuggingFace model
    hf_model = AutoModelForCausalLM.from_pretrained(HF_REPO_ID, torch_dtype=torch.bfloat16).to(torch_device)
    hf_tokenizer = AutoTokenizer.from_pretrained(HF_REPO_ID)

    # Ensure same input
    hf_input_ids = torch.tensor([input_ids], dtype=torch.long).to(torch_device)

    # JAX model
    cfg = QWEN3_CONFIG
    key = jax.random.PRNGKey(0)
    params = init_qwen3_params(key, cfg)
    params = load_qwen3_weights_jax_optimized(cfg, params, safetensors_files)
    model = {"params": params, "cfg": cfg}

    # Generate with HuggingFace model
    hf_model.eval()
    with torch.no_grad():
        hf_output = hf_model.generate(hf_input_ids, max_new_tokens=50, do_sample=False, top_k=1, temperature=0.0)

    # Generate with JAX model
    jax_output = generate_kv_optimized(
        model=model, idx=jnp.array(input_ids), max_new_tokens=50,
        context_size=QWEN3_CONFIG["context_length"], top_k=1,
        temperature=0, eos_id=None
    )

    # Convert JAX output to numpy for comparison
    jax_output_np = np.array(jax_output[0])

    # Compare outputs
    assert np.array_equal(hf_output[0].cpu().numpy(), jax_output_np), "Outputs do not match"

if __name__ == "__main__":
    pytest.main([__file__])

