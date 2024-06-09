import jax
import jax.numpy as np
from jax import lax, nn

@jax.jit
def normalize(x):
  mean = x.mean()
  std = x.std()
  y = (x - mean) / std
  return y

@jax.jit
def mix_weight(x, w):
  out = np.einsum('...k, ik, i...k', x, w)
  return out

@jax.jit
def mix(a, b, w):
  inv_w = 1 - w
  w_a = mix_weight(a, w)
  w_b = mix_weight(b, inv_w)
  out = w_a + w_b
  return out

@jax.jit
def multi_weight_only(x, w):
  out = np.einsum('i...l, ikl -> i...k', x, w)
  return out

@jax.jit
def weight_only(x, w):
  out = np.einsum('...l, kl -> ...k', x, w)
  return out

@jax.jit
def dense(x, w, b):
  out = weight_only(x, w) + b
  return out

@jax.jit
def dense_norm(x, w, b):
  out = dense(x, w, b)
  norm_out = normalize(out)
  return norm_out

@jax.jit
def rkv_layer(x, last_x, mix_w, rkv_w):
  mix_x = mix(x, last_x, mix_w)
  r_x, k_x, v = multi_weight_only(mix_x, rkv_w)

  k = np.exp(k_x)
  r = np.exp(-np.exp(r_x))

  kv = k * v
  k_kv = np.stack((k, kv))

  return k_kv, r

@jax.jit
def mem_out(mem, r, out_w):
  r_mem = mem[0] / mem[1] * r
  out = weight_only(r_mem, out_w)
  return out

@jax.jit
def serial_memory(x, last_x, last_mem, mix_w, rkv_w, out_w, raw_decay):
  k_kv, r = rkv_block(x, last_x, mix_w, rkv_w)

  decay = np.exp(-np.exp(raw_decay))
  mem = last_mem * decay + k_kv

  out = mem_out(mem, r, out_w)
  return out

@jax.jit
def mem_scan(carry, k_kv):
  last_mem, decay = carry
  mem = last_mem * decay + k_kv
  return (mem, decay), mem

@jax.jit
def memory(x, last_x, last_mem, mix_w, rkv_w, out_w, raw_decay):
  k_kv, r = rkv_block(x, last_x, mix_w, rkv_w)

  decay = np.exp(-np.exp(raw_decay))
  tmp_k_kv = np.moveaxis(k_kv, 2, 0)
  tmp_mem = lax.scan(mem_scan, (last_mem, decay), tmp_k_kv)
  new_mem = tmp_mem[-1]
  mem = np.moveaxis(tmp_mem, 0, 2)

  out = mem_out(mem, r, out_w)
  return out, x, new_mem

@jax.jit
def memory_block(x, state, params):
  x = dense_norm(x=x, **params['dense_norm'])
  dx, state['last_x'], state['last_mem'] = memory(x=x, **state, **params['memory'])
  out = x + dx
  x = nn.gelu(dense(x, **params['dense']))
  return x

if __name__ == '__main__':
  from time import time
  k = jax.random.key(0)

  i = jax.random.uniform(k, (512, 32))
  w = jax.random.uniform(k, (64, 32))
  b = jax.random.uniform(k, (64,))
  o = dense_norm(i, w, b)
  
  trials = 10000
  start = time()
  for i in range(trials):
    i = jax.random.uniform(k, (512, 32))
    w = jax.random.uniform(k, (64, 32))
    b = jax.random.uniform(k, (64,))
    o = dense_norm(i, w, b)
  print((time() - start) / trials)
