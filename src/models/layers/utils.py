import torch.nn as nn
from collections import namedtuple

from builder.utils import get_submodule_by_name as utils_get_submodule_by_name

OP = namedtuple('OP', ['OPtype', 'args'])

darts_candidate_op = (
       OP(OPtype='ConvBNAct', args=dict(kernel=3, dilation=1, pad=None, group=1, bn=True, act=nn.ReLU())),
       OP(OPtype='ConvBNAct', args=dict(kernel=5, dilation=1, pad=None, group=1, bn=True, act=nn.ReLU())),
       OP(OPtype='ConvBNAct', args=dict(kernel=3, dilation=2, pad=None, group=1, bn=True, act=nn.ReLU())),
       OP(OPtype='ConvBNAct', args=dict(kernel=5, dilation=2, pad=None, group=1, bn=True, act=nn.ReLU())),
       OP(OPtype='PoolBNAct', args=dict(pool='max', kernel=3, pad=None, bn=True, act=nn.ReLU())),
       OP(OPtype='PoolBNAct', args=dict(pool='avg', kernel=3, pad=None, bn=True, act=nn.ReLU())),
       OP(OPtype=nn.Identity, args={}),
                       )

eautodet_candidate_op = (
       OP(OPtype='SepConvBNAct_search', args=dict(
                       candidate_op=[(1,1), (3,1), (5,1), (3,2)], 
                       candidate_ch=[1.], 
                       gumbel_op=False, gumbel_channel=True,
                       bn=True, act=nn.SiLU(),
                       independent_ch_arch_param=False,
                       independent_op_arch_param=False)
         ),
)

def get_act(act=True):
    if act is None or act is False: return None
    elif act is True: return nn.ReLU()
    elif isinstance(act, nn.Module): return act
    elif isinstance(act, str): 
        return utils_get_submodule_by_name(act)()
    else:
        raise(TypeError(f"No Implementation for act func as {act}"))


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
    return utils_get_submodule_by_name(layer_name, search_path=['src.models.layers', 'torch.nn'], loaded_submodule=submodule_map)

#    submodule_name = submodule_name.split('.')
#    if len(submodule_name) == 1:
#        submodule = utils_get_submodule(submodule_name[0], '.models.layers', package_path='src', loaded_submodule=submodule_map)
#    elif len(submodule_name)>=2:
#        submodule = utils_get_submodule(submodule_name[-1], '.'.join(submodule_name[0:-1]), package_path=None, loaded_submodule=submodule_map)

#    return submodule
