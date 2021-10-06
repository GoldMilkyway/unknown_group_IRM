import torch

aa = torch.tensor([0,1,2,3,4,5,6,7,8,9])
print(len(aa))

for i in range(len(aa)):
    if aa[i] < 3:
        aa[i] = 2
    elif aa[i] > 6:
        aa[i] = 0
    else:
        aa[i] = 1
print(aa)
print(type(aa[0].item()))
aa = aa.float()

print(type(aa[0].item()))

print()