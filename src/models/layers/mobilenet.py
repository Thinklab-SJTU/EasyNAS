import torch
import torch.nn as nn

from .common import ConvBNAct, SepConvBNAct

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
