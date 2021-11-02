import torch
from torch import nn

def mean_accuracy(logits, y):
    with torch.no_grad():
        soft = nn.Softmax(dim=1)
        exps = soft(logits)
        y_pred = torch.argmax(exps, dim=1)
        return ((y_pred - y).abs() < 1e-2).float().mean()

so = nn.Softmax(dim=1)
aa = torch.tensor([[0.123,0.234,0.123]])
soaa = so(aa)
print(soaa)
print(torch.argmax(soaa, dim=1))

aa = 0
bb = 1

if aa:
    print('hello')
if bb:
    print('bye')