import math
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .utils import autopad, gumbel_softmax
from .base import OpLayer

__all__ = ["DWConvBNAct", "PoolBNAct", "ConvBNAct", "SepConvBNAct", "Identity"]


def DWConvBNAct(in_channel, out_channel, kernel=1, dilation=1, stride=1, group=1, act=True):
    # Depthwise convolution
    return ConvBNAct(in_channel, out_channel, kernel, d=dilation, s=stride, g=group, act=act)


class PoolBNAct(nn.Module):
    def __init__(self, kernel, stride=1, pool='max', pad=None, bn=True, act=nn.ReLU): 
        super(PoolBNAct, self).__init__()
        if isinstance(pool, nn.Module): 
            pool_op = pool
        elif pool == 'max':
            pool_op = nn.MaxPool2d
        elif pool == 'avg':
            pool_op = nn.AvgPool2d
        else:
            raise(ValueError(f"No implementation for pool as {pool}"))

        self.pool = pool_op(kernel_size=kernel, stride=stride, padding=autopad(kernel, pad))
        self.bn = nn.BatchNorm2d(out_channel) if bn else None
        self.act = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else None)

    def forward(self, x):
        x = self.pool(x)
        if self.bn: x = self.bn(x)
        if self.act: x = self.act(x)
        return x

class GlobalPoolBNAct(PoolBNAct):
    def forward(self, x):
        x = self.pool(x)
        if self.bn: x = self.bn(x)
        if self.act: x = self.act(x)
        x = x.view(x.size(0), -1)
        return x

class LinearAct(nn.Module):
    # Standard convolution
    def __init__(self, in_channel, out_channel, act=None, bias=False):  # ch_in, ch_out, kernel, dilation, stride, padding, groups
        super(LinearAct, self).__init__()
        self.linear = nn.Linear(in_channel, out_channel, bias=bias)
        self.act = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else None)
#        self.act = Mish() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        x = self.linear(x)
        if self.act: x = self.act(x)
        return x


class ConvBNAct(nn.Module):
    # Standard convolution
    def __init__(self, in_channel, out_channel, kernel=1, dilation=1, stride=1, pad=None, group=1, bn=True, act=nn.ReLU, bias=False):  # ch_in, ch_out, kernel, dilation, stride, padding, groups
        super(ConvBNAct, self).__init__()
        if isinstance(kernel, list): kernel = kernel[0]
        if isinstance(dilation, list): dilation = dilation[0]
        self.conv = nn.Conv2d(in_channel, out_channel, kernel, stride, autopad(kernel, pad, dilation), dilation=dilation, groups=group, bias=bias)
        self.bn = nn.BatchNorm2d(out_channel) if bn else None
        self.act = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else None)
#        self.act = Mish() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        x = self.conv(x)
        if self.bn: x = self.bn(x)
        if self.act: x = self.act(x)
        return x

class SepConvBNAct(nn.Module):
    # Standard convolution
    def __init__(self, in_channel, out_channel, kernel=1, dilation=1, stride=1, pad=None, group=1, bn=True, act=nn.ReLU, bias=False):  # ch_in, ch_out, kernel, dilation, stride, padding, groups
        super(SepConv, self).__init__()
        if isinstance(kernel, list): kernel = kernel[0]
        if isinstance(dilation, list): dilation = dilation[0]
        self.dwconv = nn.Conv2d(in_channel, out_channel, kernel, stride, autopad(kernel, pad, dilation), dilation=dilation, groups=in_channel, bias=bias)
        self.pwconv = nn.Conv2d(in_channel, out_channel, 1, 1, padding=0, dilation=1, groups=1, bias=bias)
        self.bn = nn.BatchNorm2d(out_channel) if bn else None
        self.act = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else None)

    def forward(self, x):
        x = self.pwconv(self.dwconv(x))
        if self.bn: x = self.bn(x)
        if self.act: x = self.act(x)
        return x


class SingleOpLayer(OpLayer):
    def __init__(self, in_channel, out_channel, stride, op, act=nn.ReLU, bn=True):
        super(SingleOpLayer, self).__init__()
        self.adjust_ch_op = OP(OPtype='ConvBNAct', args=dict(kernel=1, dilation=1, stride=1, bn=False, act=None))
        refined_op = self.refine_C_stride(op, in_channel, out_channel, stride)
        self.op = self.build_op(refined_op)
        self.act = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else None)
        self.bn = nn.BatchNorm2d(self.cout) if bn else None

    def forward(self, x):
       out = self.op(x)
       if self.bn: out = self.bn(out)
       if self.act: out = self.act(out)
       return out
        
class FuseLayer(OpLayer):
    # Feature Fusion
    def __init__(self, in_channels, out_channel, strides, ops, act=nn.ReLU, bn=True, fuse_op=None):
        super(FuseLayer, self).__init__()
        self.check_valid(in_channels, strides, ops)

        refined_ops = self.refine_C_stride(ops, in_channels, out_channel, strides)
        self.op = self.build_op(refined_ops)
        self.act = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else None)
        self.bn = nn.BatchNorm2d(self.cout) if bn else None

        if fuse_op is not None:
            raise(ValueError("FuseLayer has not supported other fuse type except sum."))

    def check_valid(self, in_channels, strides, ops)
        assert(len(in_channels)==len(strides))
        assert(len(in_channels)==len(ops))

    def forward(self, xs):
       out = 0.
       for idx, (op, x) in enumerate(zip(self.op, xs)):
         if op is not None:
           out += op(x)
       if self.bn: out = self.bn(out)
       if self.act: out = self.act(out)

       return out


class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, in_channel, out_channel, kernels=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = in_channel // 2  # hidden channels
        self.cv1 = Conv(in_channel, c_, k=1, d=1, s=1)
        self.cv2 = Conv(c_ * (len(k) + 1), out_channel, k=1, d=1, s=1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in kernels])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))



class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, in_channel, out_channel, kernel=1, stride=1, pad=None, group=1, act=nn.ReLU):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = ConvBNAct(in_channel * 4, out_channel, kernel, 1, stride, pad, group, act, bn=True)
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



