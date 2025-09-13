
'''
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
'''
