import torch
images = torch.rand(5*3*3*3).view(5,3,3,3)
colors = torch.tensor([0,1,2,0,1])
collors = colors +1
images[range(5), colors, :, :] *= 0
'''
def give_color():
'''
print(images)
print(collors%3)
print((collors+1)%3)