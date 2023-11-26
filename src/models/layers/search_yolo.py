import math
from collections import abc
import torch
import torch.nn as nn
import torch.nn.functional as F

from .search_common import SearchModule, ConvBNAct_search, SepConvBNAct_search, check_nesting
from .yolov5 import YOLODetect
from .utils import gumbel_softmax, get_norm
from ..utils import count_parameters_in_MB

class YOLOBottleneck_search(SearchModule):
    # Standard bottleneck
    def __init__(self, in_channel, out_channel, candidate_op=[(3,1),(5,1),(3,2)], candidate_ch=[1.], shortcut=True, group=1, expansion=0.5, separable=False, merge_kernel=True, bn_per_ch=True):  
        super(YOLOBottleneck_search, self).__init__()
        candidate_ch = check_nesting(candidate_ch, 1)
        candidate_op = check_nesting(candidate_op, 2)
        self.expansion = expansion

        c_ = int(out_channel * expansion)  # hidden channels
        c_max = int(c_ * max(candidate_ch))
        self.cv1 = ConvBNAct_search(in_channel, c_max, candidate_op=[(1,1)], candidate_ch=candidate_ch, stride=1, act=nn.SiLU, bn=nn.BatchNorm2d, merge_kernel=merge_kernel, bn_per_ch=bn_per_ch)
        if separable: my_conv = SepConvBNAct_search
        else: my_conv = ConvBNAct_search
        self.cv2 = my_conv(c_max, out_channel, candidate_op, candidate_ch=[1.], stride=1, group=group, act=nn.SiLU, bn=nn.BatchNorm2d, merge_kernel=merge_kernel)
        self.add = shortcut and in_channel == out_channel

    def forward(self, x, op_alphas=None, ch_alphas=None):
        out = self.cv2(self.cv1(x), op_alphas, ch_alphas)
        cout = min(x.size(1), out.size(1))
        return x[:,:cout,:,:] + out[:,:cout,:,:] if self.add else out

class YOLOC3_search(SearchModule):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, in_channel, out_channel, num_repeat=1, candidate_op=[(3,1),(5,1),(3,2)], candidate_ch=[1.], shortcut=True, group=1, expansion=0.5, e_bottleneck=1., search_out_channel=None, separable=False, merge_kernel=True, bn_per_ch=True):  
        super(YOLOC3_search, self).__init__()
        candidate_ch = check_nesting(candidate_ch, 1)
        candidate_op = check_nesting(candidate_op, 2)
        if isinstance(search_out_channel, (float, int)):
            self.search_out_channel = [search_out_channel]
        elif search_out_channel==True:
            self.search_out_channel = candidate_ch
        elif search_out_channel in [False, None]:
            self.search_out_channel = [1.]
        elif isinstance(search_out_channel, abc.Iterable):
            self.search_out_channel = search_out_channel
        else:
            raise(ValueError("search_out_channel has to be bool or None or an iterable instance of float"))

        self.bn_per_ch = bn_per_ch and len(self.search_out_channel)>1

        self.out_channel = out_channel
        self.candidate_ch = candidate_ch
        self.candidate_op = candidate_op
        self.e_bottleneck = [e_bottleneck for _ in range(num_repeat)] if isinstance(e_bottleneck, float) else e_bottleneck

        self.real_out_channel = out_channel = int(out_channel * max(self.search_out_channel))
        c_ = int(out_channel * expansion)  # hidden channels
        self.cv1 = ConvBNAct_search(in_channel, c_, candidate_op=[(1,1)], candidate_ch=[1.0], stride=1, merge_kernel=merge_kernel, bn=nn.BatchNorm2d, act=nn.SiLU)
        self.cv2 = ConvBNAct_search(in_channel, c_, candidate_op=[(1,1)], candidate_ch=[1.0], stride=1, merge_kernel=merge_kernel, bn=nn.BatchNorm2d, act=nn.SiLU)

        self.cv3 = nn.ModuleList([ConvBNAct_search(c_, self.out_channel, candidate_op=[(1,1)], candidate_ch=self.search_out_channel, stride=1, act=False, bn=False) for _ in range(2)])  
        self.cv3_act = nn.SiLU()
        if self.bn_per_ch:
            self.cv3_bn = nn.ModuleList([get_norm(nn.BatchNorm2d, int(self.out_channel*e)) for e in self.search_out_channel])
        else: self.cv3_bn = get_norm(nn.BatchNorm2d, out_channel)

        self.m = nn.Sequential(*[YOLOBottleneck_search(c_, c_, candidate_op, candidate_ch, shortcut, group, expansion=self.e_bottleneck[i], separable=separable, merge_kernel=merge_kernel, bn_per_ch=bn_per_ch) for i in range(num_repeat)])

    def forward(self, x, ch_alphas=None):
        ch_alphas = self.norm_arch_parameters(self.search_out_channel, ch_alphas)
        out = self.cv3[0](self.m(self.cv1(x, ch_alphas=ch_alphas)), ch_alphas=ch_alphas) + self.cv3[1](self.cv2(x, ch_alphas=ch_alphas), ch_alphas=ch_alphas)
        bn = self.get_norm_layer(ch_alphas, self.cv3_bn, self.bn_per_ch)
        return self.cv3_act(bn(out))

class YOLODetect_search(YOLODetect, SearchModule):
    def __init__(self, in_channel, strides, num_classes=80, anchors=()):  # detection layer
        super(YOLODetect_search, self).__init__(in_channel, strides, num_classes=num_classes, anchors=anchors)

    def _initialize_modules(self, in_channel):
        self.m = nn.ModuleList(ConvBNAct_search(x, self.no * self.na, candidate_op=[(1,1)], candidate_ch=[1.], stride=1, bias=False, act=nn.SiLU, bn=nn.BatchNorm2d) for x in in_channel)  # output conv
#        self.m = nn.ModuleList(ConvBNAct_search(x, self.no * self.na, candidate_op=[(1,1)], candidate_ch=[1.], stride=1, bias=False, act=False, bn=None) for x in in_channel)  # output conv

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
#        self.bias = torch.nn.Parameter(torch.zeros(len(self.strides), self.na, self.no), requires_grad=True)
        self.bias = torch.zeros(len(self.strides), self.na, self.no)
        for i, s in enumerate(self.strides):  # from
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.m[i].weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias[i], -bound, bound)
            self.bias.data[i, :, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
        self.bias.data[:, :, 5:] += math.log(0.6 / (self.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
        self.bias = nn.Parameter(self.bias.view(len(self.strides), -1), requires_grad=True)

