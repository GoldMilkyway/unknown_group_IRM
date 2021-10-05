import torch
import matplotlib.pyplot as plt
import attr
import numpy as np
from torchvision import datasets
imagess = torch.rand(1*3*14*14).view(1,3,14,14)
ima = torch.cat([imagess[0,0,:,:], imagess[0,1],imagess[0,2]])


def get_envs(cuda=True, flags=None):
    if flags is None:  # configure data generation like in original IRM paper
        @attr.s
        class DefaultFlags(object):
            """Specify spurrious correlations as original IRM paper."""
            train_env_1__color_noise = attr.ib(default=0.2)
            train_env_2__color_noise = attr.ib(default=0.1)
            test_env__color_noise = attr.ib(default=0.9)
            label_noise = attr.ib(default=0.25)

        flags = DefaultFlags()

    def _make_environment(images, labels, e):

        # NOTE: low e indicates a spurious correlation from color to (noisy) label

        def torch_bernoulli(p, size):
            return (torch.rand(size) < p).float()

        '''
        def torch_xor(a, b):
          return (a-b).abs() # Assumes both inputs are either 0 or 1
        '''

        ###
        def make_noise(labels, noise, pure_size):
            '''
            labels : torch.tensor; labels
            noise : torch.tensor; label_noise or color_noise
            pure_size : int; noise 결과의 모든 경우의 수 (가능한 label 개수, color 개수)
            label 개수와 color 개수가 다르면 다른 함수 필요
            '''
            noise_name = np.arange(0, pure_size)
            print(type(labels), type(noise), type(pure_size),type(labels[0].long()))
            for idx in range(len(labels)):
                if noise[idx]:
                    labels[idx] = np.random.choice(np.delete(noise_name, labels[idx].long()))
            return labels

        ###

        samples = dict()
        # 2x subsample for computational convenience
        images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit; flip label with probability 0.25

        ### labels 할당 수정 (2개 -> 3개)
        for i in range(len(labels)):
            if labels[i] < 3:
                labels[i] = 2
            elif labels[i] > 6:
                labels[i] = 0
            else:
                labels[i] = 1
        labels = labels.float()
        ###

        samples.update(preliminary_labels=labels)
        label_noise = torch_bernoulli(flags.label_noise, len(labels))
        labels = make_noise(labels, label_noise, 3)  ###
        samples.update(final_labels=labels)
        samples.update(label_noise=label_noise)
        # Assign a color based on the label; flip the color with probability e
        color_noise = torch_bernoulli(e, len(labels))
        colors = make_noise(labels, color_noise, 3)  ###
        samples.update(colors=colors)
        samples.update(color_noise=color_noise)
        # Apply the color to the image by zeroing out the other color channel
        images = torch.stack([images, images, images], dim=1)  ###
        images[range(len(images)), ((colors + 1) % 3).long(), :, :] *= 0  ###
        images[range(len(images)), ((colors + 2) % 3).long(), :, :] *= 0  ###
        ### images[torch.tensor(range(len(images))), (1-colors).long(), :, :] *= 0
        images = (images.float() / 255.)
        print('label', labels.size())
        #labels = labels[:, None]
        if cuda and torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        samples.update(images=images, labels=labels)
        return samples

    mnist = datasets.MNIST('~/datasets/mnist', train=True, download=True)
    mnist_train = (mnist.data[:50000], mnist.targets[:50000])
    mnist_val = (mnist.data[50000:], mnist.targets[50000:])

    rng_state = np.random.get_state()
    np.random.shuffle(mnist_train[0].numpy())
    np.random.set_state(rng_state)
    np.random.shuffle(mnist_train[1].numpy())

    envs = [
        _make_environment(mnist_train[0][::2], mnist_train[1][::2], flags.train_env_1__color_noise),
        _make_environment(mnist_train[0][1::2], mnist_train[1][1::2], flags.train_env_2__color_noise),
        _make_environment(mnist_val[0], mnist_val[1], flags.test_env__color_noise)
    ]
    return envs
env = get_envs()
ima = torch.cat([env[0]['images'][0,0],env[0]['images'][0,1],env[0]['images'][0,2]]).cpu()
ima1 = torch.cat([env[0]['images'][1,0],env[0]['images'][1,1],env[0]['images'][1,2]]).cpu()
ima2 = torch.cat([env[0]['images'][2,0],env[0]['images'][2,1],env[0]['images'][2,2]]).cpu()
ima3 = torch.cat([env[0]['images'][3,0],env[0]['images'][3,1],env[0]['images'][3,2]]).cpu()
ima4 = torch.cat([env[0]['images'][4,0],env[0]['images'][4,1],env[0]['images'][4,2]]).cpu()
ima5 = torch.cat([env[0]['images'][5,0],env[0]['images'][5,1],env[0]['images'][5,2]]).cpu()
imaa = torch.cat([ima,ima1,ima2, ima3, ima4, ima5], dim = 1)
plt.imshow(imaa)
plt.show()

