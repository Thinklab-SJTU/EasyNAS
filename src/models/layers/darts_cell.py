from easydict import EasyDict as edict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
#from mish_cuda import MishCuda as Mish

from .base import SearchModule
from src.search_space.cell_space import get_search_space, darts
from .common import ConvBNAct, SepConvBNAct, FactorizedReduce
from .search_common import AtomSearchModule
from .utils import get_act, get_layer

###################
# operation
###################

class Zero(nn.Module):
  def __init__(self, stride):
    super(Zero, self).__init__()
    self.stride = stride

  def forward(self, x):
    if self.stride == 1:
      return x.mul(0.)
    return x[:,:,::self.stride,::self.stride].mul(0.)

class darts_identity(nn.Module):
  def __init__(self, in_channel, out_channel, stride, bn=dict(submodule_name='torch.nn.BatchNorm2d', args=dict(affine=False)), act=dict(submodule_name='torch.nn.ReLU', args=dict(inplace=False))):
    super(darts_identity, self).__init__()
    if stride == 1:
        self.act = None
        self.op = nn.Identity()
    else: 
        self.act = get_act(act)
        self.op = FactorizedReduce(in_channel, out_channel, stride, bn, False)
  def forward(self, x):
    if self.act: x = self.act(x)
    return self.op(x)

class darts_conv(nn.Module):
  def __init__(self, in_channel, out_channel, kernel_dilation, stride=1, affine=False, num_repeat=2, separable=True):
    super(darts_conv, self).__init__()
    if separable: my_conv = SepConvBNAct
    else: my_conv = ConvBNAct
    kernel, dilation = kernel_dilation
    self.op1 = nn.Sequential()
    out_channels = [in_channel] * (num_repeat-1) + [out_channel]
    for i in range(num_repeat):
        self.op1.add_module('relu%d'%i, nn.ReLU(inplace=False))
        self.op1.add_module('conv%d'%i, my_conv(in_channel, out_channels[i], kernel, dilation, stride=stride if i==0 else 1, pad=None, group=1, bn=dict(submodule_name='torch.nn.BatchNorm2d', args=dict(affine=affine)), act=False, bias=False))

  def forward(self, x):
    return self.op1(x)

class mergenas_conv(nn.Module):
  def __init__(self, in_channel, out_channel, candidate_op=[(3,1),(5,1),(3,2),(5,2)], stride=1, affine=False, num_repeat=1, separable=True):
    super(mergenas_conv, self).__init__()
    if separable: my_conv = SepConvBNAct_search
    else: my_conv = ConvBNAct_search
    self.op1 = nn.Sequential()
    for i in range(num_repeat):
        self.op1.add_module('relu%d'%i, nn.ReLU(inplace=False))
        self.op1.add_module('conv%d'%i, my_conv(in_channel, out_channel, candidate_op=candidate_op, stride=stride if i==0 else 1, pad=None, group=1, bn=dict(submodule_name='torch.nn.BatchNorm2d', args=dict(affine=affine)), act=False), merge_kernel=True)

  def forward(self, x):
    return self.op1(x)


###################
# Cell
###################
class Cell_search(SearchModule):
    def __init__(self, in_channel, out_channel, strides, candidate_edge, candidate_op,  
                 multiplier=4,
                 bn=dict(name='torch.nn.BatchNorm2d', args=dict(affine=False)), 
                 gumbel_edge=False,
                 label=None,
                 drop_path_prob=0.
                ):

        super(Cell_search, self).__init__()
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
                    nn.ReLU(inplace=False),
                    FactorizedReduce(cin, C, stride=2, act=False, bn=bn) if not reduction and s==2 else ConvBNAct(cin, C, kernel=1, stride=1, act=False, bn=bn),
                    )
            self.preprocess.append(pre_op)
#            self.preprocess.append(FactorizedReduce(cin, C, stride=2, act=act, bn=bn) if not reduction and s==2 else ConvBNAct(cin, C, kernel=1, stride=1, act=act, bn=bn))

        self._ops = nn.ModuleList()
        tmp_cins, tmp_strides = [C for _ in range(len(in_channel))], strides.copy() if reduction else [1 for _ in range(len(strides))]
        self.candidate_edge = candidate_edge
        for edge, op in zip(candidate_edge, candidate_op):
            _cins = [tmp_cins[e] for e in edge]
            _strides = [tmp_strides[e] for e in edge]
            self._ops.append(AtomSearchModule(in_channel=_cins,
                                 out_channel=C,
                                 strides=_strides,
                                 input_idx=edge,
                                 candidate_op=op,
                                 act=False, bn=False,
                                 drop_path_prob=drop_path_prob
                                 ))
            tmp_cins.append(C)
            tmp_strides.append(1)

    def forward(self, inputs):
        xs = []
        for x, pre_op in zip(inputs, self.preprocess):
            xs.append(pre_op(x))

        for edge, op in zip(self.candidate_edge, self._ops):
            tmp_xs = [xs[e] for e in edge]
            xs.append(op(tmp_xs))
        return torch.cat(xs[-self._multiplier:], dim=1)




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

class Cell_search_bk(SearchModule):
    def __init__(self, in_channel, out_channel, strides, candidate_op, input_idx,  
                 steps=4, multiplier=4,
                 act=nn.ReLU(), bn=dict(name='torch.nn.BatchNorm2d', args=dict(affine=False)), 
                 gumbel_edge=False,
                 label=None
                ):

        super(Cell_search_bk, self).__init__()
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

        self._ops = nn.ModuleList()
        tmp_cins, tmp_strides = [C for _ in range(len(in_channel))], strides.copy() if reduction else [1 for _ in range(len(strides))]
        for i in range(self._steps):
            space = list(range(len(tmp_cins)))
            norm_fn = 'gumbel_softmax' if gumbel_op else 'softmax'
            sampler_cfg = {'submodule_name': 'WeightedSampler', 'args': {'norm_fn': norm_fn}}
            e_label = 'edge%d'%i if label is None else label + 'edge%d'%i
            input_idx = SearchSpace(space=space, sampler_cfg=sampler_cfg, label=e_label, num_reserve=2, reserve_replace=False)
            e_op_label = 'edge%d_op'%i if label is None else label + 'edge%d_op'%i
            e_candidate_op = []
            for j in range(len(tmp_cins)):
                e_candidate_op.append(candidate_op.new_space(label=e_op_label+'%d'%j))
            e_candidate_op = SearchSpace(space=e_candidate_op, label=e_label, num_reserve=2, reserve_replace=False)
            self._ops.append(AtomSearchModule(in_channel=tmp_cins,
                                 out_channel=C,
                                 strides=tmp_strides,
                                 input_idx=input_idx,
                                 candidate_op=e_candidate_op,
                                 act=False, bn=False,
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
            for edge_idx, edge_op in enumerate(op['args']['ops']):
                if isinstance(edge_op, (dict)):
                    if edge_op.get('args', {}).get('bn', False):
                        if edge_op['submodule_name'] == 'PoolBNAct': # when DARTS retrains, pooling has no BN
                            edge_op['args']['bn'] = False
                        else: # when DARTS retrain, affine in BN is set as True
                            edge_op['args']['bn'] = dict(submodule_name='torch.nn.BatchNorm2d', args=dict(affine=True))
                    if edge_op.get('args', {}).get('act', False) and 'Conv' in edge_op['submodule_name']:
                       edge_op['args']['act'] = False
                       op['args']['ops'][edge_idx] = [edict(submodule_name='torch.nn.ReLU', args=dict(inplace=False)), edge_op]
                else:
                    j, no_act_before = 0, True
                    while j < len(edge_op):
                        sub_op = edge_op[j]
                        if 'ReLU' in sub_op['submodule_name']: no_act_before = False
                        if sub_op.get('args', {}).get('bn', False):
                            if sub_op['submodule_name'] == 'PoolBNAct': # when DARTS retrains, pooling has no BN
                                sub_op['args']['bn'] = False
                            else: # when DARTS retrain, affine in BN is set as True
                                sub_op['args']['bn'] = dict(submodule_name='torch.nn.BatchNorm2d', args=dict(affine=True))
                        if no_act_before and sub_op.get('args', {}).get('act', False) and 'Conv' in sub_op['submodule_name']:
                                sub_op['args']['act'] = False
                                edge_op.insert(j, edict(submodule_name='torch.nn.ReLU', args=dict(inplace=False)))
                        j += 1

            edge = op.pop('input_idx')
            args['cell_ops'].append(op)
            args['edges'].append(edge)
        new_cfg = self.init_output_yaml(cfg, outOp_name="Cell", input_idx=cfg['input_idx'], **args)
        return new_cfg



