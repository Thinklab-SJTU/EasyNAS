import torch
import torch.nn as nn
from easydict import EasyDict as edict

from builder.utils import get_submodule_by_name
from .base import DiscreteSpace

def get_search_space(ss):
    if isinstance(ss, str):
        try:
            return DiscreteSpace(candidates=globals()[ss], num_reserve=1)
#            return globals()[ss]
        except KeyError:
            return get_submodule_by_name(ss)
        except Exception as e:
            raise(e)
    else: return ss

darts_conv = (
       edict(submodule_name='ConvBNAct', args=dict(kernel=3, dilation=1, pad=None, group=1, bn=dict(submodule_name='torch.nn.BatchNorm2d', args=dict(affine=False)), act='torch.nn.ReLU')),
       edict(submodule_name='ConvBNAct', args=dict(kernel=5, dilation=1, pad=None, group=1, bn=dict(submodule_name='torch.nn.BatchNorm2d', args=dict(affine=False)), act='torch.nn.ReLU')),
       edict(submodule_name='ConvBNAct', args=dict(kernel=3, dilation=2, pad=None, group=1, bn=dict(submodule_name='torch.nn.BatchNorm2d', args=dict(affine=False)), act='torch.nn.ReLU')),
       edict(submodule_name='ConvBNAct', args=dict(kernel=5, dilation=2, pad=None, group=1, bn=dict(submodule_name='torch.nn.BatchNorm2d', args=dict(affine=False)), act='torch.nn.ReLU')),
       edict(submodule_name='PoolBNAct', args=dict(pool='max', kernel=3, pad=None, bn=dict(submodule_name='torch.nn.BatchNorm2d', args=dict(affine=False)), act='torch.nn.ReLU')),
       edict(submodule_name='PoolBNAct', args=dict(pool='avg', kernel=3, pad=None, bn=dict(submodule_name='torch.nn.BatchNorm2d', args=dict(affine=False)), act='torch.nn.ReLU')),
       edict(submodule_name='src.models.layers.darts_cell.darts_identity', args=dict(bn=dict(submodule_name='torch.nn.BatchNorm2d', args=dict(affine=False)), act='torch.nn.ReLU')),
                       )

darts = (
       edict(submodule_name='Zero', args={}),
       [edict(submodule_name='torch.nn.ReLU', args=dict(inplace=False)), edict(submodule_name='SepConvBNAct', args=dict(kernel=3, dilation=1, pad=None, group=1, bn=dict(submodule_name='torch.nn.BatchNorm2d', args=dict(affine=False)), act='torch.nn.ReLU', num_pair=1)), edict(submodule_name='SepConvBNAct', args=dict(kernel=3, dilation=1, pad=None, group=1, bn=dict(submodule_name='torch.nn.BatchNorm2d', args=dict(affine=False)), act=False, num_pair=1))],
       [edict(submodule_name='torch.nn.ReLU', args=dict(inplace=False)), edict(submodule_name='SepConvBNAct', args=dict(kernel=5, dilation=1, pad=None, group=1, bn=dict(submodule_name='torch.nn.BatchNorm2d', args=dict(affine=False)), act='torch.nn.ReLU', num_pair=1)), edict(submodule_name='SepConvBNAct', args=dict(kernel=5, dilation=1, pad=None, group=1, bn=dict(submodule_name='torch.nn.BatchNorm2d', args=dict(affine=False)), act=False, num_pair=1))],
       [edict(submodule_name='torch.nn.ReLU', args=dict(inplace=False)), edict(submodule_name='SepConvBNAct', args=dict(kernel=3, dilation=2, pad=None, group=1, bn=dict(submodule_name='torch.nn.BatchNorm2d', args=dict(affine=False)), act=False, num_pair=1))],
       [edict(submodule_name='torch.nn.ReLU', args=dict(inplace=False)), edict(submodule_name='SepConvBNAct', args=dict(kernel=5, dilation=2, pad=None, group=1, bn=dict(submodule_name='torch.nn.BatchNorm2d', args=dict(affine=False)), act=False, num_pair=1))],
       edict(submodule_name='PoolBNAct', args=dict(pool='max', kernel=3, pad=None, bn=dict(submodule_name='torch.nn.BatchNorm2d', args=dict(affine=False)), act=False)),
       edict(submodule_name='PoolBNAct', args=dict(pool='avg', kernel=3, pad=None, bn=dict(submodule_name='torch.nn.BatchNorm2d', args=dict(affine=False)), act=False)),
       edict(submodule_name='src.models.layers.darts_cell.darts_identity', args=dict(bn=dict(submodule_name='torch.nn.BatchNorm2d', args=dict(affine=False)), act=edict(submodule_name='torch.nn.ReLU', args=dict(inplace=False)))),
                       )

darts_bk = (
       edict(submodule_name='SepConvBNAct', args=dict(kernel=3, dilation=1, pad=None, group=1, bn=dict(submodule_name='torch.nn.BatchNorm2d', args=dict(affine=False)), act=False, num_pair=2)),
#       [edict(submodule_name='SepConvBNAct', args=dict(kernel=3, dilation=1, pad=None, group=1, bn=dict(submodule_name='torch.nn.BatchNorm2d', args=dict(affine=False)), act='torch.nn.ReLU', num_pair=1)), edict(submodule_name='SepConvBNAct', args=dict(kernel=3, dilation=1, pad=None, group=1, bn=dict(submodule_name='torch.nn.BatchNorm2d', args=dict(affine=False)), act=False, num_pair=1))],
       edict(submodule_name='SepConvBNAct', args=dict(kernel=5, dilation=1, pad=None, group=1, bn=dict(submodule_name='torch.nn.BatchNorm2d', args=dict(affine=False)), act=False, num_pair=2)),
       edict(submodule_name='SepConvBNAct', args=dict(kernel=3, dilation=2, pad=None, group=1, bn=dict(submodule_name='torch.nn.BatchNorm2d', args=dict(affine=False)), act=False, num_pair=1)),
       edict(submodule_name='SepConvBNAct', args=dict(kernel=5, dilation=2, pad=None, group=1, bn=dict(submodule_name='torch.nn.BatchNorm2d', args=dict(affine=False)), act=False, num_pair=1)),
       edict(submodule_name='PoolBNAct', args=dict(pool='max', kernel=3, pad=None, bn=dict(submodule_name='torch.nn.BatchNorm2d', args=dict(affine=False)), act=False)),
       edict(submodule_name='PoolBNAct', args=dict(pool='avg', kernel=3, pad=None, bn=dict(submodule_name='torch.nn.BatchNorm2d', args=dict(affine=False)), act=False)),
       edict(submodule_name='src.models.layers.darts_cell.darts_identity', args=dict(bn=dict(submodule_name='torch.nn.BatchNorm2d', args=dict(affine=False)), act=False)),
                       )

mergenas = (
       edict(submodule_name='Zero', args={}),
       edict(submodule_name='SepConvBNAct_search', args=dict(
                       candidate_op=[(3,1), (5,1), (3,2), (5,2)], 
                       candidate_ch=[1.], 
                       gumbel_op=False, gumbel_channel=False,
                       bn=dict(submodule_name='torch.nn.BatchNorm2d', args=dict(affine=False)), 
                       act=edict(submodule_name='torch.nn.ReLU', args=dict(inplace=False)),
                       act_first=True,
                       bias=False,
                       merge_kernel=True,
                       independent_ch_arch_param=False,
                       independent_op_arch_param=False)),
       edict(submodule_name='PoolBNAct', args=dict(pool='max', kernel=3, pad=None, bn=dict(submodule_name='torch.nn.BatchNorm2d', args=dict(affine=False)), act=False)),
       edict(submodule_name='PoolBNAct', args=dict(pool='avg', kernel=3, pad=None, bn=dict(submodule_name='torch.nn.BatchNorm2d', args=dict(affine=False)), act=False)),
       edict(submodule_name='src.models.layers.darts_cell.darts_identity', args=dict(bn=dict(submodule_name='torch.nn.BatchNorm2d', args=dict(affine=False)), act=edict(submodule_name='torch.nn.ReLU', args=dict(inplace=False)))),
                       )

eautodet = (
       edict(submodule_name='SepConvBNAct_search', args=dict(
                       candidate_op=[(1,1), (3,1), (5,1), (3,2)], 
                       candidate_ch=[1.], 
                       gumbel_op=False, gumbel_channel=True,
                       bn=dict(submodule_name='torch.nn.BatchNorm2d', args=dict(affine=False)), act='torch.nn.SiLU',
                       independent_ch_arch_param=False,
                       independent_op_arch_param=False)
         ),
)
