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

    ### Noise 생성기
    def torch_bernoulli(noise_value, size):
      return (torch.rand(size) < noise_value).float()
    ###

    ### label과 color개수가 두 개 일때 사용 (현재 코드 X) --> 대신 make_noise사용
    def torch_xor(a, b):
      return (a-b).abs() # Assumes both inputs are either 0 or 1
    ###

    ### 랜덤 Noise 발생기
    def make_noise(labels, noise, pure_label=2):
      """
      labels : torch.tensor; labels
      noise : torch.tensor; label_noise or color_noise
      pure_size : int; noise 결과의 모든 경우의 수 (가능한 label 개수 또는 color 개수)
      label 개수와 color 개수가 서로 다르면 다른 함수 필요 (두 개수가 같다는 전제의 함수)
      """
      lab = labels.clone()
      for idx in range(len(lab)):
        noise_name = np.arange(0, pure_label)
        if noise[idx]:
          lab[idx] = np.random.choice(np.delete(noise_name, lab[idx].long()))
      return lab
    ###

    samples = dict()
    # 2x subsample for computational convenience
    images = images.reshape((-1, 28, 28))[:, ::2, ::2]
    # Assign a binary label based on the digit; flip label with probability 0.25

    ### labels 할당 수정
    for i in range(len(labels)):
      if labels[i] < 5:
        labels[i] = 0     # 0,1,2,3,4 - label: 0
      else:
        labels[i] = 1     # 5,6,7,8,9 - label: 1
    ###

    samples.update(preliminary_labels=labels)
    label_noise = torch_bernoulli(flags.label_noise, len(labels))
    labels = make_noise(labels, label_noise) ### xor --> make_noise
    #labels = labels  ### 추가
    samples.update(final_labels=labels)
    samples.update(label_noise=label_noise)
    # Assign a color based on the label; flip the color with probability e
    color_noise = torch_bernoulli(e, len(labels))
    colors = make_noise(labels, color_noise) ### xor --> make_noise
    #colors = colors.float()  ### 추가
    samples.update(colors=colors)
    samples.update(color_noise=color_noise)

    ### color_noise를 적용
    # Apply the color to the image by zeroing out the other color channel
    images = torch.stack([images, images], dim=1) ### stack 2개 --> 3개 (2color -> 3color)
    images[range(len(images)), ((colors+1)%2).long(), :, :] *= 0 ### 3COLOR 중 하나를 제거
    images = (images.float() / 255.) # image의 픽셀값들을 0~1 사이의 값으로 만듦
    #labels = labels[:, None] ### BCE가 아니라 CE를 쓸 것이기에 삭제 따라서 label은 1-dimension
    if cuda and torch.cuda.is_available():
      images = images.cuda()
      labels = labels.cuda()
    samples.update(images=images, labels=labels)
    return samples # dictionary

  mnist = datasets.MNIST('~/datasets/mnist', train=True, download=True)
  mnist_train = (mnist.data[:50000], mnist.targets[:50000]) ### 50000->50001 3env로 나누어 떨어지기 위함
  mnist_val = (mnist.data[50000:], mnist.targets[50000:])  ### 10000->9999 3env로 나누어 떨어지기 위함

  rng_state = np.random.get_state()
  np.random.shuffle(mnist_train[0].numpy())
  np.random.set_state(rng_state)
  np.random.shuffle(mnist_train[1].numpy())

  envs = [
    _make_environment(mnist_train[0][::2], mnist_train[1][::2], flags.train_env_1__color_noise),
    _make_environment(mnist_train[0][1::2], mnist_train[1][1::2], flags.train_env_2__color_noise),
    _make_environment(mnist_val[0], mnist_val[1], flags.test_env__color_noise)
  ]
  ### input envs 개수를 2개에서 3개로 늘림 color_noise는 (0.2, 0.1) -> (0.1, 0.2, 0.3)으로 교체
  ### images : env1,2,3 = 16667 x 3 x 14 x 14 labels : env1,2,3 = 16667
  ### images : val = 9999 x 3 x 14 x 14 labels : val = 9999
  return envs # list of 4 dictionary

### 필요없음
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

### 필요없음
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
