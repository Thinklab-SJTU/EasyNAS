import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
#from mish_cuda import MishCuda as Mish

from .base import SearchModule
from .search_space import get_search_space, darts
from .common import ConvBNAct, FactorizedReduce
from .search_common import AFF
from .utils import get_act

class darts_identity(nn.Module):
    def __init__(self, in_channel, out_channel, stride, affine=True, act=True):
        super(darts_identity, self).__init__()
        if stride == 1:
            self.op = nn.Identity()
        else: self.op = FactorizedReduce(in_channel, out_channel, stride, affine, act)
    
    def forward(self, x):
        return self.op(x)


class Cell(nn.Module):
  def __init__(self, in_channel, out_channel, strides, 
               cell_ops, edges, multiplier,
               act=nn.ReLU(), bn=True):
      super(Cell, self).__init__()
      self._steps = len(op)
      self.edges = edges
      self._multiplier = multiplier
      C = out_channel // multiplier

      reduction = True
      for s in strides:
          if s==1: 
              reduction = False
              break
      self.preprocess = nn.ModuleList([])
      for cin, s in zip(in_channel, strides):
          self.preprocess.append(FactorizedReduce(cin, C, stride=2, act=act) if not reduction and s==2 else ConvBNAct(cin, C, kernel=1, stride=1, act=act, bn=True))

      self._ops = nn.ModuleList()
      tmp_cins, tmp_strides = [C for _ in range(len(in_channel))], strides.copy() if reduction else [1 for _ in range(len(strides))]
      for i in range(self._steps):
          cell_ops[i]['args'].update(
              in_channel=[tmp_cins[e] for e in edges[i]],
              out_channel=C,
              strides=[tmp_strides[e] for e in edges[i]],
          )
          self._ops.append(get_layer(cell_ops[i]['submodule_name'])(**cell_ops[i]['args']))
          tmp_cins.append(C)
          strides.append(1)

  def forward(self, inputs):
      xs = []
      for x, pre_op in zip(inputs, self.preprocess):
          xs.append(pre_op(x))

      for op, edge in enumerate(self._ops, self.edges):
          xs.append(op([xs[e] for e in edge]))
      return torch.cat(xs[-self._multiplier:], dim=1)



class Cell_search(SearchModule):
    def __init__(self, in_channel, out_channel, strides, 
                 steps=4, multiplier=4,
                 candidate_op=darts, gumbel_op=False, gumbel_edge=False, 
                 act=nn.ReLU(), bn=True,
                 independent_ch_arch_param=True, independent_op_arch_param=True, independent_edge_arch_param=True):

        super(Cell_search, self).__init__()
        self._steps = steps
        self._multiplier = multiplier
        C = out_channel // multiplier

        reduction = True
        for s in strides:
            if s==1: 
                reduction = False
                break
        self.preprocess = nn.ModuleList([])
        for cin, s in zip(in_channel, strides):
            self.preprocess.append(FactorizedReduce(cin, C, stride=2, act=act) if not reduction and s==2 else ConvBNAct(cin, C, kernel=1, stride=1, act=act, bn=True))

        candidate_op = get_search_space(candidate_op)
        self._ops = nn.ModuleList()
        tmp_cins, tmp_strides = [C for _ in range(len(in_channel))], strides.copy() if reduction else [1 for _ in range(len(strides))]
        for i in range(self._steps):
            self._ops.append(AFF(in_channel=tmp_cins,
                                 out_channel=C,
                                 strides=tmp_strides,
                                 candidate_op=candidate_op,
                                 gumbel_op=gumbel_op,
                                 gumbel_edge=gumbel_edge,
                                 act=act, bn=bn,
                                 ))
            tmp_cins.append(C)
            tmp_strides.append(1)

    def forward(self, inputs):
        xs = []
        for x, pre_op in zip(inputs, self.preprocess):
            xs.append(pre_op(x))

        for op in self._ops:
            xs.append(op(xs))
        return torch.cat(xs[-self._multiplier:], dim=1)


    def discretize(self, cfg, op_alphas=None, ch_alphas=None, edge_alphas=None, num_reserved_op=1, num_reserved_edge=2):
        args = {'cell_ops': [], 'edges': []}
        for i in range(self._steps):
            op = self._ops[i].discretize()
            edge = op.pop('input_idx')
            args['cell_ops'].append(op)
            args['edges'].append(edge)
        new_cfg = self.init_output_yaml(cfg, outOp_name="Cell", input_idx=cfg['input_idx'], **args)
        return new_cfg



