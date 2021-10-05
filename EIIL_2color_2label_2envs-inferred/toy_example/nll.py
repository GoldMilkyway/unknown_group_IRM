import torch
from torch import nn

def nll(logits, y, reduction='mean'):
  logsoft = nn.LogSoftmax(dim=1)
  lossnll = nn.NLLLoss(reduction=reduction)
  return lossnll(logsoft(logits), y.squeeze().long())

def nll2(logits, y, reduction='mean'):
    return nn.functional.binary_cross_entropy_with_logits(logits, y, reduction=reduction)

logits = torch.tensor([[0.,10.],[1.,0.323]])
lables = torch.tensor([[0.],[1.]])

logits2 = torch.tensor([[0.,],[0.323]])
lables2 = torch.tensor([[0],[1]]).float()
print(logits.size(), lables.size())
print(nll(logits, lables).requires_grad_(), nll(logits, lables).dtype)
print(logits2.size(), lables.size())
print(nll2(logits2, lables2))