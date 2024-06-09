import torch as pt

@pt.compile
def normalize(x):
  mean = x.mean()
  std = x.std()
  y = (x - mean) / std
  return y

@pt.compile
def mix_weight(x, w):
  out = pt.einsum('...k, ik, i...k', x, w)
  return out

@pt.compile
def mix(a, b, w):
  inv_w = 1 - w
  w_a = mix_weight(a, w)
  w_b = mix_weight(b, inv_w)
  out = w_a + w_b
  return out

@pt.compile
def multi_weight_only(x, w):
  out = pt.einsum('i...l, ikl -> i...k', x, w)
  return out

@pt.compile
def weight_only(x, w):
  out = pt.einsum('...l, kl -> ...k', x, w)
  return out

@pt.compile
def dense_norm(x, w, b):
  out = pt.nn.functional.linear(x, w, b)
  norm_out = normalize(out)
  return norm_out

@pt.compile
def rkv_layer(x, last_x, mix_w, rkv_w):
  mix_x = mix(x, last_x, mix_w)
  r_x, k_x, v = multi_weight_only(mix_x, rkv_w)

  k = pt.exp(k_x)
  r = pt.exp(-pt.exp(r_x))

  kv = k * v
  k_kv = pt.stack((k, kv))

  return k_kv, r

@pt.compile
def mem_out_block(mem, r, out_w):
  r_mem = mem[0] / mem[1] * r
  out = weight_only(r_mem, out_w)
  return out


@pt.compile
def serial_memory(x, last_x, last_mem, mix_w, rkv_w, out_w, raw_decay):
  k_kv, r = rkv_block(x, last_x, mix_w, rkv_w)

  decay = pt.exp(-pt.exp(raw_decay))
  mem = last_mem * decay + k_kv

  out = mem_out_block(mem, r, out_w)
  return out, x, mem

@pt.jit.script
def mem_scan(last_mem, decay, k_kv):
  mem = pt.zeros_like(k_kv)
  t_len = mem.shape[2]
  for i in range(t_len):
    last_mem = last_mem * decay + k_kv[:, :, i]
    mem[:, :, i] = last_mem
  return mem, last_mem

@pt.compile
def parallel_memory(x, raw_last_x, last_mem, mix_w, rkv_w, out_w, raw_decay):
  last_x = pt.zeros_like(x)
  last_x[:, 0] = raw_last_x
  last_x[:, 1:] = x[:, :-1]
  new_x = x[:, -1]
  k_kv, r = rkv_block(x, last_x, mix_w, rkv_w)

  decay = pt.exp(-pt.exp(raw_decay))
  mem, new_mem = mem_scan(last_mem, decay, k_kv)

  out = mem_out_block(mem, r, out_w)
  return out, new_x, new_mem

class Memory(pt.nn.Module):
  def __init__(self, in_len, mem_len=None, out_len=None, serial=False, last_x=None, last_mem=None, norm_w=None, mix_w=None, rkv_w=None, decay=None, out_w=None, dense_w=None):
    super().__init__()
    
    if mem_len == None:
      mem_len = in_len
    if out_len == None:
      out_len = in_len
    self.in_len = in_len
    self.mem_len = mem_len
    self.out_len = out_len

    if norm_w == None:
      norm_w = pt.nn.Parameter(pt.randn(in_len, in_len))
    if mix_w == None:
      mix_w = pt.nn.Parameter(pt.randn(3, in_len))
    if rkv_w == None:
      rkv_w = pt.nn.Parameter(pt.randn(3, mem_len, in_len))
    if decay == None:
      decay = pt.nn.Parameter(pt.randn(mem_len))
    if out_w == None:
      out_w = pt.nn.Parameter(pt.randn(in_len, mem_len))
    if dense_w == None:
      dense_w = pt.nn.Parameter(pt.randn(out_len, in_len))
    self.norm_w = norm_w
    self.mix_w = mix_w
    self.rkv_w = rkv_w
    self.decay_w = decay_w
    self.out_w = out_w
    self.dense_w = dense_w
    
    self.memory = serial_memory if serial else memory
    self.serial = serial
    if last_x == None:
      last_x = pt.zeros(1, in_len)
    self.last_x = last_x
    if last_mem == None:
      last_mem = pt.zeros(2, 1, mem_len)
    self.last_mem = last_mem
  
  def forward(self, x):
