from pdb import set_trace
import jax
import jax.numpy as jnp


#@jax.jit
def tiled_matmul_scan(A, B):
    m, k = A.shape
    k, n = B.shape
    tile_size = m
    #m, k = 40960, 128
    #k, n = 128, 40960
    #k0s = jnp.arange(0, k, tile_size)
    set_trace()
    C = jnp.zeros((m, n))
    num_tiles = m // tile_size
    A = jnp.reshape(A, [num_tiles, m // num_tiles])
    B = jnp.reshape(B, [tile_size, n // tile_size])
    xs = (A, B) #k0s,

    def update_C(C, x):
        A_block, B_block = x
        #A_block = A[:, k0:k0 + tile_size]
        #B_block = B[k0:k0 + tile_size, :]
        return C + A_block @ B_block, None

    return jax.lax.scan(update_C, C, k0s)[0]

#@jax.jit
def tiled_matmul_v(x, y):
    tile_size = 1024
    
    def scan_body(carry, y_tile):
        set_trace()
        acc = carry
        result_tile = jnp.einsum('se,ev->sv', x, y_tile)
        acc2 = acc + result_tile
        return acc2, None
    
    y_tiles = y.reshape(y.shape[0] // tile_size, tile_size, y.shape[1])
    init_acc = jnp.zeros((x.shape[0], y.shape[1]))
    final_result, _ = jax.lax.scan(scan_body, init_acc, y_tiles)
    return final_result

def tiled_matmul(x, y):
    tile_size = 1024
    
    def scan_body(carry, x_tile):
        acc = carry
        result_tile = jnp.matmul(x_tile, y)
        set_trace()
        acc = jnp.concatenate([acc, result_tile], axis=0)
        return acc, None
    
    x_tiles = x.reshape(x.shape[0] // tile_size, tile_size, x.shape[1])
    init_acc = jnp.zeros((0, y.shape[1]))
    final_result, _ = jax.lax.scan(scan_body, init_acc, x_tiles)
    return final_result


seqlen = 2048
x = jnp.ones([seqlen,1024])
y = jnp.ones([1024,150000])

logits = jnp.einsum('se,ev->sv', x, y)
print(logits)
print(logits.shape)
l3 = jnp.matmul(x, y)
print(jnp.allclose(logits, l3))
'''

l2 = tiled_matmul(x,y)
print(l2)
'''
