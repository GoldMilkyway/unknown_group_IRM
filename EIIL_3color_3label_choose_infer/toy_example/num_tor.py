import numpy as np
import torch

num = np.ndarray([1,2,3])
idx = torch.tensor([0,1,0])
boo = torch.tensor([True, False])
print(boo.numpy())
print(torch.argmax(idx))