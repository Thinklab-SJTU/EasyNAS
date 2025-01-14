import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .utils import autopad, gumbel_softmax, get_act, get_norm
from .base import OpBuilder 

__all__ = ["DWConvBNAct", "PoolBNAct", "ConvBNAct", "SepConvBNAct", "Identity", "FuseLayer", "FactorizedReduce"]


def DWConvBNAct(in_channel, out_channel, kernel=1, dilation=1, stride=1, group=1, act=True):
    # Depthwise convolution
    return ConvBNAct(in_channel, out_channel, kernel, d=dilation, s=stride, g=group, act=act)


class PoolBNAct(nn.Module):
    def __init__(self, kernel, out_channel=None, stride=1, pool='max', pad=None, bn=dict(submodule_name='torch.nn.BatchNorm2d', args=dict(affine=True)), act=nn.ReLU(), **kwargs): 
        super(PoolBNAct, self).__init__()
        if bn: assert out_channel is not None

        if isinstance(pool, nn.Module): 
            pool_op = pool
        elif pool == 'max':
            pool_op = nn.MaxPool2d
        elif pool == 'avg':
            pool_op = nn.AvgPool2d
        else:
            raise(ValueError(f"No implementation for pool as {pool}"))

        self.pool = pool_op(kernel_size=kernel, stride=stride, padding=autopad(kernel, pad), **kwargs)
        self.bn = get_norm(bn, out_channel)
        self.act = get_act(act)

    def forward(self, x):
        x = self.pool(x)
        if self.bn: x = self.bn(x)
        if self.act: x = self.act(x)
        return x

class GlobalPoolBNAct(nn.Module):
    def __init__(self, out_channel=None, pool='avg', bn=None, act=None): 
        super(GlobalPoolBNAct, self).__init__()
        self.pool = pool
        if bn: assert out_channel is not None
        self.bn = get_norm(bn, out_channel)
        self.act = get_act(act)

    def forward(self, x):
        if self.pool == 'avg':
            x = x.mean(dim=[2,3], keepdim=True)
        elif self.pool == 'max':
            x = x.max(dim=-1, keepdim=True).max(dim=-2, keepdim=True)
        if self.bn: x = self.bn(x)
        if self.act: x = self.act(x)
        x = x.view(x.size(0), -1)
        return x

class LinearBNAct(nn.Module):
    # Standard convolution
    def __init__(self, in_channel, out_channel, act=None, bn=None, bias=False):  # ch_in, ch_out, kernel, dilation, stride, padding, groups
        super(LinearBNAct, self).__init__()
        self.linear = nn.Linear(in_channel, out_channel, bias=bias)
        self.bn = get_norm(bn, out_channel)
        self.act = get_act(act)

    def forward(self, x):
        if x.dim() > 2: x = x.view(x.size(0), -1)
        x = self.linear(x)
        if self.bn: x = self.bn(x)
        if self.act: x = self.act(x)
        return x


class ConvBNAct(nn.Module):
    def __init__(self, in_channel, out_channel, kernel=1, dilation=1, stride=1, pad=None, group=1, bn=dict(submodule_name='torch.nn.BatchNorm2d', args=dict(affine=True)), act=nn.ReLU(), bias=False):  
        super(ConvBNAct, self).__init__()
        if isinstance(kernel, list): kernel = kernel[0]
        if isinstance(dilation, list): dilation = dilation[0]
        self.conv = nn.Conv2d(in_channel, out_channel, kernel, stride, autopad(kernel, pad, dilation), dilation=dilation, groups=group, bias=bias)
        self.bn = get_norm(bn, out_channel)
        self.act = get_act(act)

    def forward(self, x):
        x = self.conv(x)
        if self.bn: x = self.bn(x)
        if self.act: x = self.act(x)
        return x

class SepConvBNAct(nn.Module):
    def __init__(self, in_channel, out_channel, kernel=1, dilation=1, stride=1, pad=None, group=1, bn=dict(name='torch.nn.BatchNorm2d', args=dict(affine=True)), act=nn.ReLU(), bias=False, num_pair=1):
        super(SepConvBNAct, self).__init__()
        if isinstance(kernel, list): kernel = kernel[0]
        if isinstance(dilation, list): dilation = dilation[0]
        self.op = nn.Sequential()
        for i in range(num_pair-1):
            self.op.add_module( 
                f'{i}_dw',
                nn.Conv2d(in_channel, in_channel, kernel, stride if i==0 else 1, autopad(kernel, pad, dilation), dilation=dilation, groups=in_channel, bias=bias)
            )
            self.op.add_module( 
                f'{i}_pw',
                nn.Conv2d(in_channel, in_channel, 1, 1, padding=0, dilation=1, groups=1, bias=bias)
            )
            if bn: 
                self.op.add_module(
                    f'{i}_bn',
                    get_norm(bn, in_channel)
                    )
            self.op.add_module(
                f'{i}_act',
                nn.ReLU() 
                )

        self.op.add_module( 
            f'{num_pair-1}_dw',
            nn.Conv2d(in_channel, in_channel, kernel, stride if num_pair==1 else 1, autopad(kernel, pad, dilation), dilation=dilation, groups=in_channel, bias=bias)
        )
        self.op.add_module( 
            f'{num_pair-1}_pw',
            nn.Conv2d(in_channel, out_channel, 1, 1, padding=0, dilation=1, groups=1, bias=bias)
        )
        if bn: 
            self.op.add_module(
                f'{num_pair-1}_bn',
                get_norm(bn, out_channel)
                )

        self.act = get_act(act)

    def forward(self, x):
        x = self.op(x)
        if self.act: x = self.act(x)
        return x


class FuseLayer(nn.Module):
    # Feature Fusion
    def __init__(self, in_channel, out_channel, strides, ops, act=nn.ReLU(), bn=dict(name='torch.nn.BatchNorm2d', args=dict(affine=True)), drop_path_prob=0., fuse_edge_func=sum, auto_refine=False, adjust_ch_op=None, upsample_op=None):
        super(FuseLayer, self).__init__()
        self.check_valid(in_channel, strides, ops)

        op_builder = OpBuilder(auto_refine=auto_refine, adjust_ch_op=adjust_ch_op, upsample_op=upsample_op)

        self.edges = nn.ModuleList([])
        for cin, s, op in zip(in_channel, strides, ops):
            self.edges.append(op_builder.build_sequence_op(op, cin, out_channel, s))

        self.act = get_act(act)
        self.bn = get_norm(bn, out_channel)
        self.drop_path = DropPath(drop_path_prob)
        self.fuse_edge_func = fuse_edge_func

    def check_valid(self, in_channel, strides, ops):
        if isinstance(ops, list):
            assert(len(in_channel)==len(strides))
            assert(len(in_channel)==len(ops))

    def forward(self, xs):
        out = self.fuse_edge_func(op(x) if isinstance(op, nn.Identity) else self.drop_path(op(x)) for op, x in zip(self.edges, xs))
        if self.bn: out = self.bn(out)
        if self.act: out = self.act(out)
 
        return out


class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, in_channel, out_channel, kernels=(5, 9, 13), expansion=0.5, bn=torch.nn.BatchNorm2d, act=nn.SiLU):
        super(SPP, self).__init__()
        c_ = int(in_channel * expansion)  # hidden channels
        self.cv1 = ConvBNAct(in_channel, c_, kernel=1, dilation=1, stride=1, bn=bn, act=act)
        self.cv2 = ConvBNAct(c_ * (len(kernels) + 1), out_channel, kernel=1, dilation=1, stride=1, bn=bn, act=act)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in kernels])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))



class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, in_channel, out_channel, kernel=1, stride=1, pad=None, group=1, act=nn.ReLU()):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = ConvBNAct(in_channel * 4, out_channel, kernel, 1, stride, pad, group, act=act, bn=dict(submodule_name='torch.nn.BatchNorm2d', args=dict(affine=True)))
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        # return self.conv(self.contract(x))


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert (H / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(N, C, H // s, s, W // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(N, C * s * s, H // s, W // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(N, s, s, C // s ** 2, H, W)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(N, C // s ** 2, H * s, W * s)  # x(1,16,160,160)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)

class FactorizedReduce(nn.Module):
  def __init__(self, in_channel, out_channel, stride=2, bn=dict(name='torch.nn.BatchNorm2d', args=dict(affine=True)), act=True):
    super(FactorizedReduce, self).__init__()
    assert out_channel % 2 == 0
    self.conv_1 = nn.Conv2d(in_channel, out_channel // 2, 1, stride=stride, padding=0, bias=False)
    self.conv_2 = nn.Conv2d(in_channel, out_channel // 2, 1, stride=stride, padding=0, bias=False) 
    self.bn = get_norm(bn, out_channel)
    self.act = get_act(act)
    self.stride=stride

  def forward(self, x):
    out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
    if self.bn: out = self.bn(out)
    if self.act: out = self.act(out)
    return out

class Zero(nn.Module):
  def __init__(self, stride):
    super(Zero, self).__init__()
    self.stride = stride

  def forward(self, x):
    if self.stride == 1:
      return x.mul(0.)
    return x[:,:,::self.stride,::self.stride].mul(0.)


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def drop_path(self, x, drop_prob: float = 0.):
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
#        random_tensor = torch.bernoulli(torch.ones(shape, dtype=x.dtype, device=x.device)*keep_prob)
        x = x.div(keep_prob).mul_(random_tensor)
        return x

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        return self.drop_path(x, self.drop_prob)

#class MyAct(nn.Module):
#    def __init__(self):
#        super(MyAct, self).__init__()
#        self.act = nn.ReLU(inplace=False)
#        print(self.act.inplace)
#    def forward(self, x):
#        print(x.sum())
#        out = self.act(x)
#        print(self.act.inplace)
##        out = nn.functional.relu(x)
#        print(x.sum(), out.sum())
#        return out
##        return nn.functional.relu(x)


