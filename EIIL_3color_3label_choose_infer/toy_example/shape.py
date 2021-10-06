import torch


scale = torch.tensor(1.).cuda().requires_grad_()
print([scale], type([scale]))