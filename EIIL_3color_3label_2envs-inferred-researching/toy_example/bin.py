import torch
import numpy as np
aaa = torch.tensor([0,1,0,1,1])
for i in range(len(aaa)):
    if aaa[i]:
        print(1)

bb = torch.tensor([0,1,2,3,4,5,6])
print(np.random.choice(bb))


for i in [0,1]:
    print(i)
indice = torch.rand(75).view(1,3,5,5)

print(aaa[0].item())