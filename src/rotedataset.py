import numpy as np
import torch as pt
from torch.utils.data import Dataset
import h5py as hp



def gen_seq(opt_num, seq_len, ind):

  seq = pt.zeros(seq_len * 2, opt_num + 1)
  seq_i = pt.zeros(seq_len, dtype=pt.int64)

  if ind == 0:
    seq[:seq_len, 0] = 1
  else:
    for i in range(seq_len):
      factor = (opt_num) ** (seq_len - i - 1)
      seq_i[i] = ind // factor
      seq[i, seq_i[i]] = 1
      ind = ind % factor

  seq[seq_len, -1] = 1
  seq[seq_len + 1:] = seq[:seq_len - 1]

  return seq, seq_i

class RoteDataset(Dataset):

  def __init__(self, opt_num, seq_len):

    self.opt_num = opt_num
    self.seq_len = seq_len

  def __len__(self):

    return self.opt_num ** self.seq_len

  def __getitem__(self, ind):

    input, target = gen_seq(self.opt_num, self.seq_len, ind)

    return input, target
  
  def gen_mask(self):

    mask = pt.zeros(self.seq_len * 2)
    mask[self.seq_len:] = 1

    return mask
