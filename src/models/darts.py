import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype

import numpy as np


class MixedOp(nn.Module):

  def __init__(self, C, stride, PRIMITIVES):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in PRIMITIVES:
      op = OPS[primitive](C, stride, False)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
      self._ops.append(op)

  def forward(self, x, weights):
#    return sum(w * op(x) for w, op in zip(weights, self._ops))
    return sum(w * op(x) if op is not None else w*0 for w, op in zip(weights, self._ops))


class Cell(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell, self).__init__()
    self.reduction = reduction
    self.primitives = self.PRIMITIVES['primitives_reduct' if reduction else 'primitives_normal']

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
    self._steps = steps
    self._multiplier = multiplier

    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()

    edge_index = 0
    for i in range(self._steps):
      for j in range(2+i):
        stride = 2 if reduction and j < 2 else 1
        op = MixedOp(C, stride, self.primitives[edge_index])
        self._ops.append(op)
        edge_index += 1

  def forward(self, s0, s1, weights, drop_prob=0.):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    offset = 0
    for i in range(self._steps):
      if drop_prob > 0. and self.training:
        s = sum(drop_path(self._ops[offset+j](h, weights[offset+j]), drop_prob) for j, h in enumerate(states))
      else:
        s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
      offset += len(states)

      if len(s.shape) < 4:
        if self.reduction: 
          H,W = s0.shape[2:4]
          s = torch.zeros([s0.shape[0], s0.shape[1], int(H//2), int(W//2)], device=s0.device, dtype=s0.dtype) 
        else: 
          s = torch.zeros_like(s0)

      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):

  def __init__(self, C, num_classes, layers, criterion, primitives, steps=4, multiplier=4, stem_multiplier=3, alpha_weights=None, drop_path_prob=0.0):
    super(Network, self).__init__()
    self.alpha_weights = alpha_weights
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier
    self.drop_path_prob = drop_path_prob

    nn.Module.PRIMITIVES = primitives

    C_curr = stem_multiplier*C
    self.stem = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )
 
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(layers):
      if i in [layers//3, 2*layers//3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier*C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

    self._initialize_alphas()

    self.prune_masks = [torch.ones_like(self.alphas_normal).bool(), torch.ones_like(self.alphas_reduce).bool()] # 0-prune; 1-reserve

  def new(self):
    model_new = Network(self._C, self._num_classes, self._layers, self._criterion, self.PRIMITIVES, drop_path_prob=self.drop_path_prob).cuda()
    model_new.prune_masks = self.prune_masks
    model_new.prune(self.prune_masks)
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, input):
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      if self.alpha_weights is None:
        if cell.reduction:
          weights = F.softmax(self.alphas_reduce, dim=-1)
        else:
          weights = F.softmax(self.alphas_normal, dim=-1)
      else:
        raise(ValueError("Why you want to set alphas manually?"))
        print(self.alpha_weights['alphas_normal'])
        print(self.alpha_weights['alphas_reduce'])
        if cell.reduction:
          weights = self.alpha_weights['alphas_reduce']
        else:
          weights = self.alpha_weights['alphas_normal']
      
      s0, s1 = s1, cell(s0, s1, weights, self.drop_path_prob)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits

  def _loss(self, input, target):
    logits = self(input)
    return self._criterion(logits, target) 

  def _initialize_alphas(self):
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    num_ops = len(self.PRIMITIVES['primitives_normal'][0])

    self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self._arch_parameters = [
      self.alphas_normal,
      self.alphas_reduce,
    ]

  def arch_parameters(self):
    return self._arch_parameters

  def genotype_ori(self):

    def _parse(weights, normal=True):
      PRIMITIVES = self.PRIMITIVES['primitives_normal' if normal else 'primitives_reduct']

      gene = []
      n = 2
      start = 0
      for i in range(self._steps):
        end = start + n
        W = weights[start:end].copy()

        try:
          edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES[x].index('none')))[:2]
        except ValueError: # This error happens when the 'none' op is not present in the ops
          edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x]))))[:2]

        for j in edges:
          k_best = None
          for k in range(len(W[j])):
            if 'none' in PRIMITIVES[j]:
              if k != PRIMITIVES[j].index('none'):
                if k_best is None or W[j][k] > W[j][k_best]:
                  k_best = k
            else:
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((PRIMITIVES[start+j][k_best], j))
        start = end
        n += 1
      return gene

    gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy(), True)
    gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy(), False)

    concat = list(range(2+self._steps-self._multiplier, self._steps+2))
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype


  def genotype(self, no_restrict=False):

    def _parse(weights, prune_mask, no_restrict=False, normal=True):
      PRIMITIVES = self.PRIMITIVES['primitives_normal' if normal else 'primitives_reduct']

      gene = []
      if no_restrict:
        activate = [1,1] + [0]*self._steps
        n = 2
        start = 0
        node_end_row = [] # [2,5,9,14]
        node_start_row = [] # [0,2,5,9]
        for i in range(self._steps): 
          end = start + n
          node_start_row.append(start)
          node_end_row.append(end)
          start = end
          n += 1
        R,C = weights.shape
        weights = (weights * prune_mask).reshape(-1)
        kept = int(min(prune_mask.sum().item(), self.num_kept_max))
        idxes = np.sort(weights.argsort()[-kept:])
        for idx in idxes:
          pos_r = int(idx / C)
          pos_c = int(idx - pos_r*C)
          if prune_mask[pos_r, pos_c] == 0: 
            print("idx %d-%d has been pruned, so we donot keep this op"%(pos_r, pos_c))
            continue
#            raise(ValueError("Code goes wrong"))
          for i, end_row in enumerate(node_end_row):
            if pos_r < end_row:
              dst_node = i+2
              break
          src_node = pos_r - node_start_row[dst_node-2]
          if not activate[src_node]: continue
          activate[dst_node] = 1
          gene.append((PRIMITIVES[pos_r][pos_c], src_node, dst_node))
        concat = (np.where(activate[-self._multiplier:])[0] + 2).tolist()
            
      else:
        n = 2
        start = 0
        for i in range(self._steps):
          end = start + n
          W = weights[start:end].copy()
          W = W * prune_mask[start:end]
          try:
            edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES[x].index('none')))[:2]
          except ValueError: # This error happens when the 'none' op is not present in the ops
            edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x]))))[:2]
          for j in edges:
            k_best = None
            for k in range(len(W[j])):
              if 'none' in PRIMITIVES[j]:
                if k != PRIMITIVES[j].index('none'):
                  if k_best is None or W[j][k] > W[j][k_best]:
                    k_best = k
              else:
                if k_best is None or W[j][k] > W[j][k_best]:
                  k_best = k
            gene.append((PRIMITIVES[start+j][k_best], j))
          start = end
          n += 1
        concat = list(range(2+self._steps-self._multiplier, self._steps+2))
      return gene, concat

    gene_normal, concat_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy(), self.prune_masks[0].data.cpu().numpy(), no_restrict, True)
    gene_reduce, concat_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy(), self.prune_masks[1].data.cpu().numpy(), no_restrict, False)

    genotype = Genotype(
      normal=gene_normal, normal_concat=concat_normal,
      reduce=gene_reduce, reduce_concat=concat_reduce
    )
    return genotype

  def prune(self, prune_masks):
    self.prune_masks = prune_masks
    mask_normal, mask_reduce = prune_masks

    for idx, cell in enumerate(self.cells):
        mask = mask_reduce if cell.reduction else mask_normal
        r = 0
        for i in range(self._steps):
          for j in range(2+i):
            stride = 2 if cell.reduction and j < 2 else 1
            for c in range(mask.shape[1]):
              if mask[r,c] == 0:
                cell._ops[r]._ops[c] = None
            r = r+1
    torch.cuda.empty_cache()

