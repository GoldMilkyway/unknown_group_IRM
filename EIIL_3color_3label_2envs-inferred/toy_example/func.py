import torch
import numpy as np

label_name = torch.tensor([0,1,2])
labels = torch.tensor([0,1,2,1,2,0,0,1,2,2])
label_noise = torch.tensor([0,0,1,1,1,1,1,0,0,0])

def make_noise(labels, noise, pure_size):

    lab = labels.clone()
    for idx in range(len(labels)):
        noise_name = np.arange(0, pure_size)
        if noise[idx]:
            aa = np.delete(noise_name, lab[idx].long())
            lab[idx] = np.random.choice(aa)
    return lab

a = make_noise(labels, label_noise, 3)
print(labels)
print(a)