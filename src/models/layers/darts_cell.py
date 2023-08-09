import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
#from mish_cuda import MishCuda as Mish

from .base import SearchModule
from .search_space import get_search_space, darts
from .common import ConvBNAct, FactorizedReduce
from .search_common import AFF
from .utils import get_act, get_layer

def darts_identity(in_channel, out_channel, stride, bn=dict(name='torch.nn.BatchNorm2d', args=dict(affine=False)), act=True):
    if stride == 1:
        return nn.Identity()
    else: 
        if act:
            return nn.Sequential(
                    get_act(act),
                    FactorizedReduce(in_channel, out_channel, stride, bn, False)
                    )
        else:
            return FactorizedReduce(in_channel, out_channel, stride, bn, False)


class Cell(nn.Module):
  def __init__(self, in_channel, out_channel, strides, 
               cell_ops, edges, multiplier=4,
               act=nn.ReLU(), bn=dict(name='torch.nn.BatchNorm2d', args=dict(affine=True)),
               drop_path_prob=0.2):
      super(Cell, self).__init__()
      self._steps = len(cell_ops)
      self.edges = edges
      self._multiplier = multiplier
      C = out_channel // multiplier
      self.strides = strides

      reduction = True
      for s in strides:
          if s==1: 
              reduction = False
              break
      self.preprocess = nn.ModuleList([])
      for cin, s in zip(in_channel, strides):
          pre_op = nn.Sequential(
                  get_act(act),
                  FactorizedReduce(cin, C, stride=2, act=False, bn=bn) if not reduction and s==2 else ConvBNAct(cin, C, kernel=1, stride=1, act=False, bn=bn),
                  )
          self.preprocess.append(pre_op)
#          self.preprocess.append(FactorizedReduce(cin, C, stride=2, act=act, bn=bn) if not reduction and s==2 else ConvBNAct(cin, C, kernel=1, stride=1, act=act, bn=bn))

      self._ops = nn.ModuleList()
      tmp_cins, tmp_strides = [C for _ in range(len(in_channel))], strides.copy() if reduction else [1 for _ in range(len(strides))]
      for i in range(self._steps):
          cell_ops[i]['args'].update(
              in_channel=[tmp_cins[e] for e in edges[i]],
              out_channel=C,
              strides=[tmp_strides[e] for e in edges[i]],
              act=False,
              bn=False,
              drop_path_prob=drop_path_prob
          )
          self._ops.append(get_layer(cell_ops[i]['submodule_name'])(**cell_ops[i]['args']))
          tmp_cins.append(C)
          tmp_strides.append(1)

  def forward(self, inputs):
      xs = []
      for x, pre_op in zip(inputs, self.preprocess):
          xs.append(pre_op(x))

      for op, edge in zip(self._ops, self.edges):
          xs.append(op([xs[e] for e in edge]))
      return torch.cat(xs[-self._multiplier:], dim=1)



class Cell_search(SearchModule):
    def __init__(self, in_channel, out_channel, strides, 
                 steps=4, multiplier=4,
                 candidate_op=darts, gumbel_op=False, gumbel_edge=False, 
                 act=nn.ReLU(), bn=dict(name='torch.nn.BatchNorm2d', args=dict(affine=False)),
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
            pre_op = nn.Sequential(
                    get_act(),
                    FactorizedReduce(cin, C, stride=2, act=False, bn=bn) if not reduction and s==2 else ConvBNAct(cin, C, kernel=1, stride=1, act=False, bn=bn),
                    )
            self.preprocess.append(pre_op)
#            self.preprocess.append(FactorizedReduce(cin, C, stride=2, act=act, bn=bn) if not reduction and s==2 else ConvBNAct(cin, C, kernel=1, stride=1, act=act, bn=bn))

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
                                 act=False, bn=False,
                                 independent_edge_arch_param=independent_edge_arch_param,
                                 independent_op_arch_param=independent_op_arch_param,
                                 independent_ch_arch_param=independent_ch_arch_param,
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
        args = dict(multiplier=self._multiplier, cell_ops=[], edges=[], 
                drop_path_prob=0.2,
                bn=dict(submodule_name='torch.nn.BatchNorm2d', args=dict(affine=True)),
                )
        for i in range(self._steps):
            op = self._ops[i].discretize()
            # set affine as True for each BN
            for edge_op in op['args']['ops']:
                if isinstance(edge_op, (dict)) and edge_op.get('args', {}).get('bn', False):
                    if edge_op['submodule_name'] == 'PoolBNAct': # when DARTS retrains, pooling has no BN
                        edge_op['args']['bn'] = False
                    else: # when DARTS retrain, affine in BN is set as True
                        edge_op['args']['bn'] = dict(submodule_name='torch.nn.BatchNorm2d', args=dict(affine=True))
                else:
                    for sub_op in edge_op:
                        if 'bn' in sub_op.get('args', {}):
                            if sub_op['submodule_name'] == 'PoolBNAct': # when DARTS retrains, pooling has no BN
                                sub_op['args']['bn'] = False
                            else: # when DARTS retrain, affine in BN is set as True
                                sub_op['args']['bn'] = dict(submodule_name='torch.nn.BatchNorm2d', args=dict(affine=True))
            edge = op.pop('input_idx')
            args['cell_ops'].append(op)
            args['edges'].append(edge)
        new_cfg = self.init_output_yaml(cfg, outOp_name="Cell", input_idx=cfg['input_idx'], **args)
        return new_cfg



