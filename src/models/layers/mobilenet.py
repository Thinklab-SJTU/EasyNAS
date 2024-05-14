import torch
import torch.nn as nn

from .common import ConvBNAct, SepConvBNAct
from .base import SearchModule

class InvertedResidual(nn.Module):
    def __init__(
            self, in_channel: int, out_channel: int, kernel: int, stride: int, expand_ratio: int) -> None:
        super().__init__()
        self.stride = stride
        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 instead of {stride}")

        hidden_dim = int(round(in_channel * expand_ratio))
        self.use_res_connect = self.stride == 1 and in_channel == out_channel

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(
                ConvBNAct(in_channel, hidden_dim, kernel=1, bn=True, act=nn.ReLU6())
            )
        layers.extend(
            [
                # dw
                ConvBNAct(
                    hidden_dim,
                    hidden_dim,
                    kernel=kernel,
                    stride=stride,
                    group=hidden_dim,
                    bn=True,
                    act=nn.ReLU6()
                ),
                # pw-linear
                ConvBNAct(hidden_dim, out_channel,
                    kernel=1,
                    stride=1,
                    bias=False,
                    bn=True,
                    act=None)
            ]
        )
        self.conv = nn.Sequential(*layers)
        self.out_channel = out_channel
        self.in_channel = in_channel
        self._is_cn = stride > 1

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class InvertedResidual_search(SearchModule):
    def __init__(
            self, in_channel: int, out_channel: int, expand_ratio: int, candidate_op=[(1,1), (3,1), (5,1), (3,2)], candidate_ch=[1.], gumbel_op=False, gumbel_channel=True, stride=1, merge_kernel=True) -> None:
        super().__init__()
        self.stride = stride
        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 instead of {stride}")

        hidden_dim = int(round(in_channel * expand_ratio))
        self.use_res_connect = self.stride == 1 and in_channel == out_channel

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(
#                ConvBNAct(in_channel, hidden_dim, kernel=1, bn=True, act=nn.ReLU6())
                ConvBNAct_search(in_channel, hidden_dim, 
                    candidate_op=[(1,1)], 
                    canddiate_ch=candidate_ch,
                    gumbel_channel=gumbel_channel,
                    stride=1,
                    bn=True, act=nn.ReLU6())
            )
        layers.extend(
            [
                # dw
                ConvBNAct_search(hidden_dim, hidden_dim, 
                    candidate_op=candidate_op, 
                    canddiate_ch=[1.],
                    gumbel_op=gumbel_op,
                    gumbel_channel=gumbel_channel,
                    stride=stride,
                    group=hidden_dim,
                    merge_kernel=merge_kernel,
                    independent_ch_arch_param=True,
                    independent_op_arch_param=True,
                    bn=True, act=nn.ReLU6()),
                # pw-linear
                ConvBNAct(hidden_dim, out_channel,
                    kernel=1,
                    stride=1,
                    bias=False,
                    bn=True,
                    act=None)
            ]
        )
        self.conv = nn.Sequential(*layers)
        self.out_channel = out_channel
        self.in_channel = in_channel
        self._is_cn = stride > 1

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

#    @classmethod
#    def genotype(cls, cfg):
