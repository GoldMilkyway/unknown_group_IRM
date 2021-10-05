"""Build environments."""
import attr
import numpy as np
import torch
from torchvision import datasets


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
      lab = labels.clone()
      for idx in range(len(lab)):
        noise_name = np.arange(0, pure_size)
        if noise[idx]:
          lab[idx] = np.random.choice(np.delete(noise_name, lab[idx].long()))
      return lab
    ###

    samples = dict()
    # 2x subsample for computational convenience
    images = images.reshape((-1, 28, 28))[:, ::2, ::2]
    # Assign a binary label based on the digit; flip label with probability 0.25

    ### labels 할당 수정 (2개)
    for i in range(len(labels)):
      if labels[i] < 5:
        labels[i] = 1
      else:
        labels[i] = 0
    ###

    samples.update(preliminary_labels=labels)
    label_noise = torch_bernoulli(flags.label_noise, len(labels))
    labels = make_noise(labels, label_noise, 2) ###
    samples.update(final_labels=labels)
    samples.update(label_noise=label_noise)
    # Assign a color based on the label; flip the color with probability e
    color_noise = torch_bernoulli(e, len(labels))
    colors = make_noise(labels, color_noise, 2) ###
    samples.update(colors=colors)
    samples.update(color_noise=color_noise)
    # Apply the color to the image by zeroing out the other color channel
    images = torch.stack([images, images], dim=1) ###
    images[range(len(images)), ((colors+1)%2).long(), :, :] *= 0 ###
    ### images[torch.tensor(range(len(images))), (1-colors).long(), :, :] *= 0
    images = (images.float() / 255.)
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


def get_envs_with_indices():
  """Return IRM envs but with indices and environment indicators."""
  envs = get_envs()
  examples_so_far = 0
  for i, env in enumerate(envs):
    num_examples = len(env['images'])
    env['idx'] = idx = torch.tensor(
         np.arange(examples_so_far, examples_so_far + num_examples),
         dtype=torch.int32
    )
    examples_so_far += num_examples
    # here "env" is a label indicating which env each example belongs to
    env['env'] = torch.tensor(i * np.ones_like(env['idx']), dtype=torch.uint8)
  return envs


def split_by_noise(env, noise_var='label'):
  assert noise_var in ('label', 'color'), 'Unexpected noise variable.'
  noise_name = '%s_noise' % noise_var
  clean_idx = (env[noise_name] == 0.)
  noisy_idx = (env[noise_name] == 1.)
  from copy import deepcopy
  clean_env, noisy_env = deepcopy(env), deepcopy(env)
  for k, v in clean_env.items():
    if v.numel() > 1:
      clean_env[k] = v[clean_idx]
  for k, v in noisy_env.items():
    if v.numel() > 1:
      noisy_env[k] = v[noisy_idx]
  return clean_env, noisy_env
