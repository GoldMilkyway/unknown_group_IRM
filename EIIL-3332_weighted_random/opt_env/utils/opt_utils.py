import numpy as np
import torch
from torch import autograd
from torch import nn
from torch import optim
from tqdm import tqdm

#def nll(logits, y, reduction='mean'):
#  return nn.functional.binary_cross_entropy_with_logits(logits, y, reduction=reduction)

def nll(logits, y, reduction='mean'):
  return nn.functional.cross_entropy(logits, y, reduction=reduction)

#def mean_accuracy(logits, y):
#  preds = (logits > 0.).float()
#  return ((preds - y).abs() < 1e-2).float().mean()
def mean_accuracy(logits, y):
  softmax = nn.Softmax(dim=1)
  prob = softmax(logits)
  y_pred = torch.argmax(prob, dim=1).float()
  return ((y_pred - y).abs() < 1e-2).float().mean()

def penalty(logits, y):
  scale = torch.tensor(1.).cuda().requires_grad_()
  loss = nll(logits * scale, y)
  grad = autograd.grad(loss, [scale], create_graph=True)[0]
  return torch.sum(grad**2)

### Input envs -> Infer envs
def split_data_opt(envs, model, flags, n_steps=10000, n_samples=-1, lr=0.001,
                    batch_size=None, join=True, no_tqdm=False): ### flags 추가
  """Learn soft environment assignment."""

  if join:  # assumes first two entries in envs list are the train sets to joined
    print('pooling envs')
    # pool all training envs (defined as each env in envs[:-1])
    joined_train_envs = dict()
    for k in envs[0].keys():
      if envs[0][k].numel() > 1:  # omit scalars previously stored during training
        joined_values = torch.cat((envs[0][k][:n_samples],
                                   envs[1][k][:n_samples],
                                   envs[2][k][:n_samples]), ### envs[2]추가
                                  0)
        joined_train_envs[k] = joined_values ### 나누어져있던 input_envs들을 모두 합침
    print('size of pooled envs: %d' % len(joined_train_envs['images']))
  else:
    if not isinstance(envs, dict):
      raise ValueError(('When join=False, first argument should be a dict'
                        ' corresponding to the only environment.'
                       ))
    print('splitting data from single env of size %d' % len(envs['images']))
    joined_train_envs = envs

  scale = torch.tensor(1.).cuda().requires_grad_()
  if batch_size:
    logits = []
    i = 0
    num_examples = len(joined_train_envs['images'])
    while i < num_examples:
      images = joined_train_envs['images'][i:i+64] ### batch_size = 64
      images = images.cuda()
      logits.append(model(images).detach())
      i += 64
    logits = torch.cat(logits)
  else:
    logits = model(joined_train_envs['images'])
    logits = logits.detach()

  loss = nll(logits * scale, joined_train_envs['labels'].cuda(), reduction='none')
  ### infered envs 개수 : 2 ########################################inferred envs 개수 바꾸는 곳
  if flags.eiil: ### eiil과 eiil_infer구분
    env_w = torch.randn(len(logits), 2).cuda().requires_grad_() ## 학습시킬 대상 : soft-environment assignment
    softmax = nn.Softmax(dim=1) ###
    #prob_env = softmax(env_w) ### = soft-assignment of inferred environment
    ### requires_grad_()를 env_w에서 prob_env로 옮김 뒤에 lossa를 계산 시 backpropagation을 제대로 사용하기 위함
    optimizer = optim.Adam([env_w], lr=lr)

    ### envs inferring 하는 과정 2개와 3개를 시도해볼 것 현재 : 2개
    with tqdm(total=n_steps, position=1, bar_format='{desc}', desc='AED Loss: ', disable=no_tqdm) as desc:
      for i in tqdm(range(n_steps), disable=no_tqdm):
        # penalty for env a
        lossa = (loss.squeeze() * softmax(env_w)[:,0]).mean() ### env1의 prob
        grada = autograd.grad(lossa, [scale], create_graph=True)[0]
        penaltya = torch.sum(grada**2)
        # penalty for env b
        lossb = (loss.squeeze() * softmax(env_w)[:,1]).mean() ### env2의 prob
        gradb = autograd.grad(lossb, [scale], create_graph=True)[0]
        penaltyb = torch.sum(gradb**2)
        # negate
        npenalty = - torch.stack([penaltya, penaltyb]).mean()
        # step
        optimizer.zero_grad()
        npenalty.backward(retain_graph=True)
        optimizer.step()
        desc.set_description('AED Loss: %.8f' % npenalty.cpu().item())

    print('Final AED Loss: %.8f' % npenalty.cpu().item())

    # split envs based on env_w threshold
    new_envs = []
    ##idx0 = (env_w.sigmoid()>.5) ### BCE 기준
    ##idx1 = (env_w.sigmoid()<=.5) ### BCE 기준
    idx0 = []
    idx1 = [] ### idxn의 개수는 infer할 environment 개수
    for row in range(len(env_w)): ## 각 data마다
      if torch.argmax(env_w[row])==0:
        idx0.append(row)   ## index=0 이 제일 큰 것
      if torch.argmax(env_w[row])==1:
        idx1.append(row)   ## index=1 이 제일 큰것  # train envs
    ###
    # NOTE: envs include original data indices for qualitative investigation
    for _idx in (idx0, idx1):
      new_env = dict()
      for k, v in joined_train_envs.items():
        if k == 'paths':  # paths is formatted as a list of str, not ndarray or tensor
          v_ = np.array(v)
          v_ = v_[_idx.cpu().numpy()]
          v_ = list(v_)
          new_env[k] = v_
        else:
          new_env[k] = v[_idx]
      new_envs.append(new_env)

  if flags.eiil_infer: ### eiil과 eiil_infer구분
    env_w = torch.randn(len(logits), 3).cuda().requires_grad_()  ## 학습시킬 대상 : soft-environment assignment
    softmax = nn.Softmax(dim=1)  ###
    #prob_env = softmax(env_w)  ### = soft-assignment of inferred environment
    ### requires_grad_()를 env_w에서 prob_env로 옮김 뒤에 lossa를 계산 시 backpropagation을 제대로 사용하기 위함
    optimizer = optim.Adam([env_w], lr=lr)

    ### envs inferring 하는 과정 2개와 3개를 시도해볼 것 현재 : 3개
    with tqdm(total=n_steps, position=1, bar_format='{desc}', desc='AED Loss: ', disable=no_tqdm) as desc:
      for i in tqdm(range(n_steps), disable=no_tqdm):
        # penalty for env a
        lossa = (loss.squeeze() * softmax(env_w)[:, 0]).mean()  ### env1의 prob
        grada = autograd.grad(lossa, [scale], create_graph=True)[0]
        penaltya = torch.sum(grada ** 2)
        # penalty for env b
        lossb = (loss.squeeze() * softmax(env_w)[:, 1]).mean()  ### env2의 prob
        gradb = autograd.grad(lossb, [scale], create_graph=True)[0]
        penaltyb = torch.sum(gradb ** 2)
        # penalty for env c ### 추가
        lossc = (loss.squeeze() * softmax(env_w)[:, 2]).mean()  ### env3의 prob
        gradc = autograd.grad(lossc, [scale], create_graph=True)[0]
        penaltyc = torch.sum(gradc ** 2)
        # negate
        npenalty = - torch.stack([penaltya, penaltyb, penaltyc]).mean() ### penaltyc 추가
        # step
        optimizer.zero_grad()
        npenalty.backward(retain_graph=True)
        optimizer.step()
        desc.set_description('AED Loss: %.8f' % npenalty.cpu().item())

    print('Final AED Loss: %.8f' % npenalty.cpu().item())

    # split envs based on env_w threshold
    new_envs = []
    idx0 = []
    idx1 = []  ### idxn의 개수는 infer할 environment 개수
    idx2 = []  ### infer 3를 위해 추가
    for row in range(len(env_w)):
      if torch.argmax(env_w[row]) == 0:
        idx0.append(row)
      if torch.argmax(env_w[row]) == 1:
        idx1.append(row)
      if torch.argmax(env_w[row]) == 2: ###추가
        idx2.append(row) ### 추가
    ###
    # NOTE: envs include original data indices for qualitative investigation
    for _idx in (idx0, idx1,idx2):
      new_env = dict()
      for k, v in joined_train_envs.items():
        if k == 'paths':  # paths is formatted as a list of str, not ndarray or tensor
          v_ = np.array(v)
          v_ = v_[_idx.cpu().numpy()]
          v_ = list(v_)
          new_env[k] = v_
        else:
          new_env[k] = v[_idx]
      new_envs.append(new_env)

  #print('size of env0: %d' % len(new_envs[0]['images']))
  #print('size of env1: %d' % len(new_envs[1]['images']))

  if join:  #NOTE: assume the user includes test set as part of arguments only if join=True
    new_envs.append(envs[-1])
    #print('size of env2: %d' % len(new_envs[2]['images']))
  return new_envs


def train_irm_batch(model, envs, flags):
  """Batch version of the IRM algo for CMNIST expers."""
  def _pretty_print(*values):
    col_width = 13
    def format_val(v):
      if not isinstance(v, str):
        v = np.array2string(v, precision=5, floatmode='fixed')
      return v.ljust(col_width)
    str_values = [format_val(v) for v in values]
    print("   ".join(str_values))

  #if flags.color_based_eval:  # skip IRM and evaluate color-based model
  #  from opt_env.utils.model_utils import ColorBasedClassifier
  #  model = ColorBasedClassifier()
  #if not flags.color_based_eval:
  optimizer = optim.Adam(model.parameters(), lr=flags.lr)
  if flags.eiil:
    print('env1: ', len(envs[0]['images']),
          'env2: ', len(envs[1]['images']),
          'env3: ', len(envs[2]['images']))
    for step in range(flags.steps):
      for env in envs:
        logits = model(env['images'])
        env['nll'] = nll(logits, env['labels'])
        env['acc'] = mean_accuracy(logits, env['labels'])
        env['penalty'] = penalty(logits, env['labels'])

      train_nll = torch.stack([envs[0]['nll'], envs[1]['nll']]).mean()
      train_acc = torch.stack([envs[0]['acc'], envs[1]['acc']]).mean()
      train_penalty = torch.stack([envs[0]['penalty'], envs[1]['penalty']]).mean()

      weight_norm = torch.tensor(0.).cuda()
      for w in model.parameters():
        weight_norm += w.norm().pow(2)
      loss = train_nll.clone()
      loss += flags.l2_regularizer_weight * weight_norm
      penalty_weight = (flags.penalty_weight
                        if step >= flags.penalty_anneal_iters else 1.0)
      loss += penalty_weight * train_penalty
      if penalty_weight > 1.0:
        # Rescale the entire loss to keep gradients in a reasonable range
        loss /= penalty_weight

      # if not flags.color_based_eval:
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      test_acc = envs[2]['acc']
      if step % 100 == 0:
        _pretty_print(
          np.int32(step),
          train_nll.detach().cpu().numpy(),
          train_acc.detach().cpu().numpy(),
          train_penalty.detach().cpu().numpy(),
          test_acc.detach().cpu().numpy()
        )

  else:
    print('env1: ', len(envs[0]['images']),
          'env2: ', len(envs[1]['images']),
          'env3: ', len(envs[2]['images']),
          'test: ', len(envs[3]['images']))
    for step in range(flags.steps):
      for env in envs:
        logits = model(env['images'])
        env['nll'] = nll(logits, env['labels'])
        env['acc'] = mean_accuracy(logits, env['labels'])
        env['penalty'] = penalty(logits, env['labels'])



      train_nll = torch.stack([envs[0]['nll'], envs[1]['nll'], envs[2]['nll']]).mean()
      train_acc = torch.stack([envs[0]['acc'], envs[1]['acc'], envs[2]['acc']]).mean()
      train_penalty = torch.stack([envs[0]['penalty'], envs[1]['penalty'], envs[2]['penalty']]).mean()

      weight_norm = torch.tensor(0.).cuda()
      for w in model.parameters():
        weight_norm += w.norm().pow(2)
      loss = train_nll.clone()
      loss += flags.l2_regularizer_weight * weight_norm
      penalty_weight = (flags.penalty_weight
          if step >= flags.penalty_anneal_iters else 1.0)
      loss += penalty_weight * train_penalty
      if penalty_weight > 1.0:
        # Rescale the entire loss to keep gradients in a reasonable range
        loss /= penalty_weight

      #if not flags.color_based_eval:
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      test_acc = envs[3]['acc']
      if step % 100 == 0:
        _pretty_print(
          np.int32(step),
          train_nll.detach().cpu().numpy(),
          train_acc.detach().cpu().numpy(),
          train_penalty.detach().cpu().numpy(),
          test_acc.detach().cpu().numpy()
        )

  final_train_acc = train_acc.detach().cpu().numpy()
  final_test_acc = test_acc.detach().cpu().numpy()
  return model, final_train_acc, final_test_acc

