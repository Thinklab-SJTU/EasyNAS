import math
from collections import abc
import torch
import torch.nn as nn
import torch.nn.functional as F

from .search_common import ConvBNAct_search, SepConvBNAct_search, check_nesting
from .yolov5 import YOLODetect
from .base import SearchModule
from .utils import gumbel_softmax

class YOLOBottleneck_search(SearchModule):
    # Standard bottleneck
    def __init__(self, in_channel, out_channel, candidate_op=[(3,1),(5,1),(3,2)], candidate_ch=[1.], shortcut=True, group=1, expansion=0.5, gumbel_channel=False, separable=False, merge_kernel=True):  # ch_in, ch_out, shortcut, groups, expansion
        super(YOLOBottleneck_search, self).__init__()
        candidate_ch = check_nesting(candidate_ch, 1)
        candidate_op = check_nesting(candidate_op, 2)
        self.gumbel_channel = gumbel_channel
        self.expansion = expansion

        c_ = int(out_channel * expansion)  # hidden channels
        c_max = int(c_ * max(candidate_ch))
        self.cv1 = ConvBNAct_search(in_channel, c_max, candidate_op=[(1,1)], candidate_ch=candidate_ch, stride=1, gumbel_channel=gumbel_channel, act=nn.SiLU, bn=nn.BatchNorm2d, merge_kernel=merge_kernel)
        if separable: my_conv = SepConvBNAct_search
        else: my_conv = ConvBNAct_search
        self.cv2 = my_conv(c_max, out_channel, candidate_op, candidate_ch=[1.], stride=1, group=group, gumbel_channel=gumbel_channel, act=nn.SiLU, bn=nn.BatchNorm2d, merge_kernel=merge_kernel)
        self.add = shortcut and in_channel == out_channel

    def forward(self, x, op_alphas=None, ch_alphas=None):
        if self.gumbel_channel:
          cout = x.size(1)
          out = self.cv2(self.cv1(x), op_alphas, ch_alphas)
          return x + out[:,:cout,:,:] if self.add else out
        else:
          return x + self.cv2(self.cv1(x), op_alphas, ch_alphas) if self.add else self.cv2(self.cv1(x), op_alphas, ch_alphas)

class YOLOC3_search(SearchModule):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, in_channel, out_channel, num_repeat=1, candidate_op=[(3,1),(5,1),(3,2)], candidate_ch=[1.], shortcut=True, group=1, expansion=0.5, e_bottleneck=1., search_out_channel=None, gumbel_channel=False, separable=False, merge_kernel=True):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(YOLOC3_search, self).__init__()
        candidate_ch = check_nesting(candidate_ch, 1)
        candidate_op = check_nesting(candidate_op, 2)
        if search_out_channel==True:
            self.search_out_channel = candidate_ch
        elif search_out_channel in [False, None]:
            self.search_out_channel = [1.]
        elif isinstance(search_out_channel, abc.Iterable):
            self.search_out_channel = search_out_channel
        elif isinstance(search_out_channel, float):
            self.search_out_channel = [search_out_channel]
        else:
            raise(ValueError("search_out_channel has to be bool or None or an iterable instance of float"))
        self.out_channel = out_channel
        self.candidate_ch = candidate_ch
        self.candidate_op = candidate_op
        self.e_bottleneck = [e_bottleneck for _ in range(num_repeat)] if isinstance(e_bottleneck, float) else e_bottleneck

        out_channel = out_channel * max(self.search_out_channel)
        c_ = int(out_channel * expansion)  # hidden channels
        self.cv1 = ConvBNAct_search(in_channel, c_, candidate_op=[(1,1)], candidate_ch=self.search_out_channel, stride=1, gumbel_channel=gumbel_channel, independent_ch_arch_param=False, merge_kernel=merge_kernel, bn=nn.BatchNorm2d, act=nn.SiLU)
        self.cv2 = ConvBNAct_search(in_channel, c_, candidate_op=[(1,1)], candidate_ch=self.search_out_channel, stride=1, gumbel_channel=gumbel_channel, independent_ch_arch_param=False, merge_kernel=merge_kernel, bn=nn.BatchNorm2d, act=nn.SiLU)

        self.gumbel_channel = gumbel_channel and len(self.search_out_channel) > 1
        if self.gumbel_channel :
            self.cv3 = nn.ModuleList([ConvBNAct_search(c_, out_channel, candidate_op=[(1,1)], candidate_ch=self.search_out_channel, stride=1, gumbel_channel=gumbel_channel, act=False, bn=False, independent_ch_arch_param=False) for _ in range(2)])  
            self.cv3_bn = nn.ModuleList([nn.BatchNorm2d(int(out_channel*e)) for e in self.search_out_channel])
            self.cv3_act = nn.SiLU()
        else:
            self.cv3 = ConvBNAct_search(2 * c_, out_channel, candidate_op=[(1,1)], candidate_ch=self.search_out_channel, stride=1, gumbel_channel=gumbel_channel, independent_ch_arch_param=False, merge_kernel=merge_kernel, bn=nn.BatchNorm2d, act=nn.SiLU())  

        if len(self.search_out_channel) > 1:
            self.init_arch_parameters('ch_alphas', len(self.search_out_channel))

        self.m = nn.Sequential(*[YOLOBottleneck_search(c_, c_, candidate_op, candidate_ch, shortcut, group, expansion=self.e_bottleneck[i], gumbel_channel=gumbel_channel, separable=separable, merge_kernel=merge_kernel) for i in range(num_repeat)])

    def forward(self, x):
        if self.gumbel_channel:
            ch_alphas = gumbel_softmax(F.log_softmax(self.ch_alphas, dim=-1), hard=True) if hasattr(self, 'ch_alphas') else None 
            out = self.cv3[0](self.m(self.cv1(x, ch_alphas=ch_alphas)), ch_alphas=ch_alphas) + self.cv3[1](self.cv2(x, ch_alphas=ch_alphas), ch_alphas=ch_alphas)
            a_e, idx = ch_alphas.max(dim=-1)
            return self.cv3_act(self.cv3_bn[idx](out))
        else:
            ch_alphas = nn.functional.softmax(self.ch_alphas, dim=-1) if hasattr(self, 'ch_alphas') else None
            return self.cv3(torch.cat((self.m(self.cv1(x, ch_alphas=ch_alphas)), self.cv2(x, ch_alphas=ch_alphas)), dim=1), ch_alphas=ch_alphas)

    def discretize(self, cfg=None, op_alphas=None, ch_alphas=None, edge_alphas=None, num_reserved_op=1, num_reserved_edge=2):
        assert num_reserved_op==1

        args = {}
        ch_alphas = getattr(self, 'ch_alphas', None) if ch_alphas is None else ch_alphas
        if ch_alphas is not None:
            ch_alphas_idx = self.get_reserved_idx(1, ch_alphas)[0]
            if cfg is not None:
                args['out_channel'] = cfg['args']['out_channel'] * cfg['args']['search_out_channel'][ch_alphas_idx]
            else:
                args['out_channel'] = self.out_channel * self.search_out_channel[ch_alphas_idx]
        kernel, dilation, e_bottleneck = [], [], []
        for m in self.m:
            if op_alphas is None: op_alphas = m.cv2.op_alphas
            op_alphas_idx = self.get_reserved_idx(num_reserved_op, op_alphas)[0]
            k, d = self.candidate_op[op_alphas_idx]
            kernel.append(k)
            dilation.append(d)

            if m.cv1.ch_alphas is not None:
                ch_alphas_idx = self.get_reserved_idx(1, m.cv1.ch_alphas)[0]
                e_bottleneck.append(m.expansion * self.candidate_ch[ch_alphas_idx])
            else: e_bottleneck.append(m.expansion)

        args['kernel'] = kernel
        args['dilation'] = dilation
        args['e_bottleneck'] = e_bottleneck

        new_cfg = self.init_output_yaml(cfg, outOp_name='YOLOC3', input_idx=-1, **args)
        return new_cfg

class YOLODetect_search(YOLODetect, SearchModule):
    def __init__(self, in_channel, strides, num_classes=80, anchors=()):  # detection layer
        super(YOLODetect_search, self).__init__(in_channel, strides, num_classes=num_classes, anchors=anchors)

    def _initialize_modules(self, in_channel):
        self.m = nn.ModuleList(ConvBNAct_search(x, self.no * self.na, candidate_op=[(1,1)], candidate_ch=[1.], stride=1, bias=False, act=nn.SiLU, bn=nn.BatchNorm2d) for x in in_channel)  # output conv
#        self.m = nn.ModuleList(ConvBNAct_search(x, self.no * self.na, candidate_op=[(1,1)], candidate_ch=[1.], stride=1, bias=False, act=None, bn=None) for x in in_channel)  # output conv

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        self.bias = torch.nn.Parameter(torch.zeros(len(self.strides), self.na, self.no), requires_grad=True)
        for i, s in enumerate(self.strides):  # from
            self.bias.data[i, :, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
        self.bias.data[:, :, 5:] += math.log(0.6 / (self.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls

    def discretize(self, cfg=None, op_alphas=None, ch_alphas=None, edge_alphas=None, num_reserved_op=1, num_reserved_edge=2):
        return self.init_output_yaml(cfg, outOp_name='YOLODetect')


