import torch
import torch.nn as nn

from .base import OP_CFG
from builder.utils import get_submodule_by_name

def get_search_space(ss):
    if isinstance(ss, str):
        try:
            return globals()[ss]
        except KeyError:
            return get_submodule_by_name(ss)
        except Exception as e:
            raise(e)
    else: return ss

darts = (
       OP_CFG(submodule_name='ConvBNAct', args=dict(kernel=3, dilation=1, pad=None, group=1, bn=True, act='torch.nn.ReLU')),
       OP_CFG(submodule_name='ConvBNAct', args=dict(kernel=5, dilation=1, pad=None, group=1, bn=True, act='torch.nn.ReLU')),
       OP_CFG(submodule_name='ConvBNAct', args=dict(kernel=3, dilation=2, pad=None, group=1, bn=True, act='torch.nn.ReLU')),
       OP_CFG(submodule_name='ConvBNAct', args=dict(kernel=5, dilation=2, pad=None, group=1, bn=True, act='torch.nn.ReLU')),
       OP_CFG(submodule_name='PoolBNAct', args=dict(pool='max', kernel=3, pad=None, bn=True, act='torch.nn.ReLU')),
       OP_CFG(submodule_name='PoolBNAct', args=dict(pool='avg', kernel=3, pad=None, bn=True, act='torch.nn.ReLU')),
       OP_CFG(submodule_name='src.models.layers.darts_cell.darts_identity', args=dict(affine=True, act='torch.nn.ReLU')),
                       )

eautodet = (
       OP_CFG(submodule_name='SepConvBNAct_search', args=dict(
                       candidate_op=[(1,1), (3,1), (5,1), (3,2)], 
                       candidate_ch=[1.], 
                       gumbel_op=False, gumbel_channel=True,
                       bn=True, act='torch.nn.SiLU',
                       independent_ch_arch_param=False,
                       independent_op_arch_param=False)
         ),
)
