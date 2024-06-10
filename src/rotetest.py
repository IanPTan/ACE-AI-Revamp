import matplotlib.pyplot as plt
import numpy as np
import torch as pt
from torch.utils.data import DataLoader, random_split
from time import time
from rotedataset import RoteDataset
from model import MemoryBlock


device = pt.device('cuda' if pt.cuda.is_available() else 'cpu')

def test(ds, m, ind, full=False):

  model.reset()
  i, t = ds[ind]
  i, t = i.to(device), t.to(device)
  p = m(i)[5:]
  o = pt.argmax(p, dim=-1)

  if full:
    return o, t, p
  else:
    print(f'Sequence comparison for datapoint {ind}:')
    for it, io in zip(t, o):
      mark = '✔' if it == io else 'WRONG'
      print(f'\t{it}\t{io}\t{mark}')

class Model(pt.nn.Module):

  def __init__(self, in_len, out_len, mem_len, lay_len, serial=False):
    super(Model, self).__init__()
    self.serial = serial
    self.in_len = in_len
    self.mem_len = mem_len

    self.model = pt.nn.Sequential(
        pt.nn.Linear(in_len, mem_len),
        *[MemoryBlock(mem_len, serial=serial) for _ in range(lay_len)],
        pt.nn.Linear(mem_len, out_len),
        pt.nn.Softmax(dim=-1),
        )

  def forward(self, x):
    return self.model(x)

  def reset(self):
    for layer in self.model:
      if isinstance(layer, MemoryBlock):
        layer.reset()

  def set_serial(self, serial):
    for layer in self.model:
      if isinstance(layer, MemoryBlock):
        layer.set_serial(serial)

epochs = 500
opt_num = 3
seq_len = 5
test_ratio = 0.1
batch_size = 219
hidden_size = 32
lay_len = 2
backup_interval = 100
trials = 10

dataset = RoteDataset(opt_num, seq_len)

test_size = int(test_ratio * len(dataset))
train_size = len(dataset) - test_size

train_batch_len = train_size // batch_size + bool(train_size % batch_size)
test_batch_len = test_size // test_size + bool(test_size % batch_size)

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

models = []
times = []
all_trial_losses = []
all_trial_test_losses = []
# pt.autograd.set_detect_anomaly(True)
for trial in range(trials):

  model = Model(opt_num + 1, opt_num, hidden_size, lay_len).to(device)
  criterion = pt.nn.CrossEntropyLoss()
  optimizer = pt.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

  all_losses = pt.zeros(epochs)
  all_test_losses = pt.zeros(epochs)

  print(f'Starting training trial {trial + 1}/{trials}, using {device}.')

  start = time()
  for epoch in range(epochs):

    losses = pt.zeros(train_batch_len)
    test_losses = pt.zeros(test_batch_len)

    model.train()
    for i, (inputs, targets) in enumerate(train_loader):
      inputs, targets = inputs.to(device), targets.to(device)
      exit()
      optimizer.zero_grad()
      model.reset()
      outputs = model(inputs)[:, seq_len:].permute(0, 2, 1)
      loss = criterion(outputs, targets)
      losses[i] = loss
      loss.backward()
      optimizer.step()

    model.eval()
    with pt.no_grad():
      for i, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        model.reset()
        outputs = model(inputs)[:, seq_len:].permute(0, 2, 1)
        loss = criterion(outputs, targets)
        test_losses[i] = loss

    all_losses[epoch] = pt.mean(losses)
    all_test_losses[epoch] = pt.mean(test_losses)

    if (epoch + 1) % backup_interval == 0:
      print(f'Epoch {epoch + 1}/{epochs}, time: {time() - start:.3f}, Loss: {pt.mean(losses)}, Val Loss: {pt.mean(test_losses)}')
      pt.save(model.state_dict(), 'backup.ckpt')

  models.append(model.state_dict())
  times.append(time() - start)
  all_trial_losses.append(all_losses.detach())
  all_trial_test_losses.append(all_test_losses.detach())
  print(f'Finished! ({times[-1]:.3f}s)')

pt.save(model.state_dict(), 'backup.ckpt')
times = np.array(times)
print(f'Finished all trials in {times.sum():.3f}s with average time {times.mean():.3f}s.')
for all_losses in all_trial_losses:
  plt.plot(all_losses)
plt.show()
for all_test_losses in all_trial_test_losses:
  plt.plot(all_test_losses)
plt.show()
