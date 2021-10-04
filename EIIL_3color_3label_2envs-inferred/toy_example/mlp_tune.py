import torch
from torch import nn
import pickle
import os
from torchvision import datasets

'''  if flags is None:
    assert results_dir is not None, "flags and results_dir cannot both be None."
    flags = pickle.load(open(os.path.join(results_dir, 'flags.p'), 'rb'))'''


def load_mlp(hidden_dim = 390):

  class MLP(nn.Module):
    def __init__(self):
      super(MLP, self).__init__()
      lin1 = nn.Linear(1 * 28 * 28, hidden_dim)
      lin2 = nn.Linear(hidden_dim, hidden_dim)
      lin3 = nn.Linear(hidden_dim, 10)
      for lin in [lin1, lin2, lin3]:
        nn.init.xavier_uniform_(lin.weight)
        nn.init.zeros_(lin.bias)
      self._main = nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True), lin3)
    def forward(self, input):
      out = input.view(input.shape[0], 1 * 28 * 28)
      out = self._main(out)
      return out
  mlp = MLP()
  if torch.cuda.is_available():
    mlp = mlp.cuda()
  mlp.eval()

  return mlp

'''  if results_dir is not None:
    mlp.load_state_dict(torch.load(os.path.join(results_dir, basename)))
    print('Model params loaded from %s.' % results_dir)
  else:
    print('Model built with randomly initialized parameters.')'''

flags = {'hidden_dim': 390, 'grayscale_model' : None}
mlp_pre = load_mlp().cuda()
mlp_pre.train()

mnist = datasets.MNIST('~/datasets/mnist', train=True, download=True)
mnist_train = (mnist.data[:50000].cuda(), mnist.targets[:50000].cuda())
mnist_val = (mnist.data[50000:].cuda(), mnist.targets[50000:].cuda())
logits = mlp_pre(mnist_train[0].float())
print(logits, type(logits), logits.size())

soft = nn.Softmax(dim=1)
input = torch.randn(2,3)
output = soft(input)
print(output)
logsoft = nn.LogSoftmax(dim=1)
print(logsoft(logits))
print('logsoft shape', logsoft(logits).size())
nnn = nn.NLLLoss(reduction='mean')
result = nnn(logsoft(logits), mnist_train[1].long()) + torch.tensor(100)
print('result= ', result)
print(type(result), result.size())

aaaa = torch.rand(100,3).view(3,-1)
print(aaaa.size())