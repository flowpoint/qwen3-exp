def manual_vmap(fn, *args):
    dim1 = args[0].shape[0]
    outputs = []
    for i in range(dim1):
        inputs_i = [arg[i] for arg in args]  # Grab the i-th slice from each arg
        out = fn(*inputs_i)
        outputs.append(out)

    return jnp.stack(outputs)

def manual_vmap(att_head, queries, keys_expanded, values_expanded, pre, position_offset):    
    xr = []
    for query, key, value in zip(queries, keys_expanded, values_expanded):
        x = att_head(query, key, value, pre, position_offset)
        xr.append(x)
    return jnp.stack(xr)

def get_logits_old(cfg, x, params):
    logits_chunks = []
    seq_chunk_size = 1024
    for chunk_start in range(0, cfg["context_length"], seq_chunk_size):
        chunk_end = min(chunk_start + seq_chunk_size, cfg["context_length"])
        x_chunk = x[:, chunk_start:chunk_end]
        print(x_chunk.shape)
        print(params['out_head'].shape)
        logits_chunk = jnp.einsum('bse,ev->bsv', x_chunk, params["out_head"])
        logits_chunks.append(logits_chunk)
    
    logits = jnp.concatenate(logits_chunks, axis=1)
    return logits

def get_logits(cfg, x, params, vocab_chunk_size=128):
    """Chunked version of get_logits across the vocab/embedding dimension"""
    logits_chunks = []
    seq_chunk_size = 1
    
    for chunk_start in range(0, cfg["context_length"], seq_chunk_size):
        chunk_end = min(chunk_start + seq_chunk_size, cfg["context_length"])
        x_chunk = x[:, chunk_start:chunk_end]
        
        vocab_chunks = []
        vocab_size = params['out_head'].shape[0]
        
        for vocab_chunk_start in range(0, vocab_size, vocab_chunk_size):
            vocab_chunk_end = min(vocab_chunk_start + vocab_chunk_size, vocab_size)
            #print('--')
            #print(params['out_head'].shape)
            out_head_chunk = params["out_head"][:, vocab_chunk_start:vocab_chunk_end]
            #print(x_chunk.shape)
            #print(out_head_chunk.shape)
            logits_chunk = jnp.einsum('bse,ev->bsv', x_chunk, out_head_chunk)
            vocab_chunks.append(logits_chunk)
        
        # Concatenate vocab chunks for this sequence chunk
        logits_seq_chunk = jnp.concatenate(vocab_chunks, axis=2)
        logits_chunks.append(logits_seq_chunk)
    
    logits = jnp.concatenate(logits_chunks, axis=1)
    return logits


def gen2(f, logits, kv_cache, position_offset,  max_new_tokens):
    [logits, kv_cache, position_offset], seq = jax.lax.scan(
            f, 
            init=[logits, kv_cache, position_offset], length=max_new_tokens,
            unroll=False, #20,
            )
    return [logits, kv_cache, position_offset], seq


def steptop(params, cfg):
    #@jax.jit
    def step(args,_):#logits, kv_cache, position_offset, cur_ids):
        logits, kv_cache, position_offset = args

        next_token_logits = logits[:, -1, :]
        next_token = jnp.argmax(next_token_logits, axis=-1)

        '''
        if eos_id is not None and jnp.any(next_token == eos_id):
            break
        '''
        
        #cur_ids = jnp.concatenate([cur_ids, next_token[:, None]], axis=1)
        
        # Process next tokens for entire batch
        logits, kv_cache, position_offset = qwen3_forward_kv(params, next_token[:, None], cfg, kv_cache, position_offset)
        return [logits, kv_cache, position_offset ], next_token
        #return [logits, kv_cache, position_offset, cur_ids], None
    return step
