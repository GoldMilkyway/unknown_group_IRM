import torch
import numpy as np

label_name = torch.tensor([0,1,2])
labels = torch.tensor([0,1,2,1,2,0,0,1,2,2])
label_noise = torch.tensor([0,0,1,1,1,1,1,0,0,0])


def make_noise(labels, noise, pure_size):
    '''
    labels : torch.tensor; labels
    noise : torch.tensor; label_noise or color_noise
    pure_size : int; noise 결과의 모든 경우의 수 (가능한 label 개수, color 개수)
    label 개수와 color 개수가 다르면 다른 함수 필요
    '''
    lab = torch.tensor([])
    for idx in range(len(labels)):
        noise_name = np.arange(0, pure_size)
        if noise[idx]:
            aa = np.delete(noise_name, labels[idx].long())
            labels[idx] = np.random.choice(aa)
        print(noise_name)
        print('noised', labels)
    labelss = labels.clone()
    print(noise_name)
    labels = torch.tensor([0,1,2,1,2,0,0,1,2,2])
    return lab
print(labels)
a = make_noise(labels, label_noise, 3)
print(a)
print(labels)

