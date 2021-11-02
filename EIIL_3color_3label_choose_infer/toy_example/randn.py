import torch
from torch import nn

length = 10
class_num = 3
log = torch.rand([10,1]).float()
logits = torch.rand([10,3])
labels = torch.tensor([0,1,2,1,0,1,1,0,2,0])
labelss = labels[:,None]

print(labelss.size())
print((labels.size()))

def nll(logits, y, reduction='mean'):
  return nn.functional.binary_cross_entropy_with_logits(logits, y, reduction=reduction)

def nll2(logits, y, reduction='mean'):
  return nn.functional.cross_entropy(logits,y, reduction=reduction)

nll_value = nll(log, labelss.float())
nll2_value = nll2(logits, labels)
print(nll_value, 'pr')
env_w = torch.randn([len(logits), class_num])
print(nll_value, nll2_value)

aaa = torch.tensor([0,1,2,3,4,5])
print(torch.sort(aaa, dim=0, descending=True))