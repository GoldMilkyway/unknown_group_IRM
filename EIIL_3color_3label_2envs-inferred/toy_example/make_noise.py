import torch
import numpy as np

label_name = torch.tensor([0,1,2])
labels = torch.tensor([0,1,2,1,2,0,0,1,2,2])
label_noise = torch.tensor([0,0,1,1,1,1,1,0,0,0])

for idx in range(len(labels)):
    if label_noise[idx]:
        labels[idx] = np.random.choice(np.delete(label_name.numpy(), labels[idx].numpy()))
print(labels)
