import torch

aa = torch.rand(2)
bb = torch.rand((2,1))

print(aa, aa.size())
print(bb, bb.size())
print(aa-bb)