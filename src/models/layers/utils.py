from functools import partial
import torch.nn as nn
from collections import namedtuple
import torch

from builder.utils import get_submodule_by_name as utils_get_submodule_by_name

def get_act(act=True, *args, **kwargs):
    if act is None or act is False: return None
    elif isinstance(act, nn.Module): return act
    elif act is True: return nn.ReLU(*args, **kwargs)
    return get_module(act, *args, search_path='torch.nn', **kwargs)

def get_norm(norm, *args, **kwargs):
    return get_module(norm, *args, search_path='torch.nn', **kwargs)

def get_module(module, *args, search_path=('torch.nn',), **kwargs):
    if module is None or module is False: return None
    elif isinstance(module, str):
        module = utils_get_submodule_by_name(module, search_path=search_path)
    elif isinstance(module, dict):
        module_args = module.get('args', {})
        for k in kwargs.keys(): module_args.pop(k, None)
        module = partial(utils_get_submodule_by_name(module['submodule_name'], search_path=search_path), **module_args)
    try:
        return module(*args, **kwargs)
    except TypeError as e:
        raise(e)
#        raise(TypeError(f"No Implementation for module as {module}"))



def autopad(k, p=None, d=1):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = (k-1)*d // 2 if isinstance(k, int) else [(x-1)*d // 2 for x in k]  # auto-pad
    return p

def sample_gumbel(shape, device, eps=1e-20):
    while True:
      gumbel = -torch.empty(shape, device=device).exponential_().log()
      if torch.isinf(gumbel).any() or torch.isnan(gumbel).any(): continue
      else: break
    return gumbel
#    U = torch.rand(shape, device=device)
#    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature=1.):
    y = logits + sample_gumbel(logits.size(), logits.device)
    return nn.functional.softmax(y / temperature, dim=-1)


def gumbel_softmax_old(logits, temperature=1, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)

    if not hard:
        return y

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard

#useage: gumbel_softmax(F.log_softmax(alpha, dim=-1), hard=True))
def gumbel_softmax(logits, temperature=1, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    while True:
      y = gumbel_softmax_sample(logits, temperature)
      if torch.isinf(y).any() or torch.isnan(y).any(): continue
      else: break

    if not hard:
        return y

    shape = y.size()
    _, ind = y.max(dim=-1, keepdim=True)
    y_hard = torch.zeros_like(y)
    y_hard.scatter_(-1, ind, 1.)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = y_hard - y.detach() + y
    return y_hard


submodule_map = {}
def get_layer(layer_name):
    return utils_get_submodule_by_name(layer_name, search_path=['src.models.layers', 'src.models', 'torch.nn'], loaded_submodule=submodule_map)

#    submodule_name = submodule_name.split('.')
#    if len(submodule_name) == 1:
#        submodule = utils_get_submodule(submodule_name[0], '.models.layers', package_path='src', loaded_submodule=submodule_map)
#    elif len(submodule_name)>=2:
#        submodule = utils_get_submodule(submodule_name[-1], '.'.join(submodule_name[0:-1]), package_path=None, loaded_submodule=submodule_map)

#    return submodule
