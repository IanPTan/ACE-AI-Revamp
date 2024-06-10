import torch as pt

@pt.compile
def normalize(x):
  mean = x.mean()
  std = x.std()
  y = (x - mean) / std
  return y

@pt.compile
def mix_weight(x, w):
  out = pt.einsum('...k, ik -> i...k', x, w)
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
def rkv_block(x, last_x, mix_w, rkv_w):
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
def serial_memory(x, last_x, last_mem, mix_w, rkv_w, out_w, decay):
  k_kv, r = rkv_block(x, last_x, mix_w, rkv_w)

  mem_decay = pt.exp(-pt.exp(decay))
  mem = last_mem * mem_decay + k_kv

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
def parallel_memory(x, last_x, last_mem, mix_w, rkv_w, out_w, decay):
  all_last_x = pt.zeros_like(x)
  all_last_x[:, 0] = last_x
  all_last_x[:, 1:] = x[:, :-1]
  new_x = x[:, -1]
  k_kv, r = rkv_block(x, all_last_x, mix_w, rkv_w)

  mem_decay = pt.exp(-pt.exp(decay))
  mem, new_mem = mem_scan(last_mem, mem_decay, k_kv)

  out = mem_out_block(mem, r, out_w)
  return out, new_x, new_mem

class DenseNorm(pt.nn.Module):
  def __init__(self, in_len, out_len):
    super(DenseNorm, self).__init__()
    self.dense = pt.nn.Linear(in_len, out_len)

  def forward(self, x):
    x = self.dense(x)
    x = normalize(x)
    return x

class MemoryBlock(pt.nn.Module):
  def __init__(self, in_len, mem_len=None, out_len=None, serial=False, last_x=None, last_mem=None, mix_w=None, rkv_w=None, decay=None, out_w=None):
    super(MemoryBlock, self).__init__()
    
    self.in_len = in_len
    self.mem_len = mem_len if mem_len else self.in_len
    self.out_len = out_len if out_len else self.in_len


    self.mix_w = mix_w if mix_w else pt.nn.Parameter(pt.randn(3, self.in_len))
    self.rkv_w = rkv_w if rkv_w else pt.nn.Parameter(pt.randn(3, self.mem_len, self.in_len))
    self.decay = decay if decay else pt.nn.Parameter(pt.randn(self.mem_len))
    self.out_w = out_w if out_w else pt.nn.Parameter(pt.randn(self.in_len, self.mem_len))

    self.dense_norm = DenseNorm(self.in_len, self.in_len)
    self.dense = pt.nn.Linear(self.in_len, self.out_len)
    self.gelu = pt.nn.GELU()
    
    self.memory = serial_memory if serial else parallel_memory
    self.register_buffer('last_x', last_x if last_x else pt.zeros(1, self.in_len))
    self.register_buffer('last_mem', last_mem if last_mem else pt.zeros(2, 1, self.mem_len))
  
  def forward(self, x):
    x = self.dense_norm(x)
    dx, self.last_x, self.last_mem = self.memory(x, last_x=self.last_x, last_mem=self.last_mem, mix_w=self.mix_w, rkv_w=self.rkv_w, out_w=self.out_w, decay=self.decay)
    x = x + dx
    x = self.gelu(x)
    x = self.dense(x)
    return x

  def reset(self):
    self.last_x = pt.zeros(1, self.in_len, device=self.last_x.device)
    self.last_mem = pt.zeros(2, 1, self.mem_len, device=self.last_mem.device)

  def set_serial(self, serial):
    self.memory = serial_memory if serial else parallel_memory

  def get_state(self):
    return {
        'last_x': self.last_x,
        'last_mem': self.last_mem,
        }

if __name__ == '__main__':
  print('Running test...')
  d = pt.device('cuda')
  l = MemoryBlock(8, serial=0).to(d)
  i = pt.randn(256, 16, 8, device=d)
  o = l(i)
