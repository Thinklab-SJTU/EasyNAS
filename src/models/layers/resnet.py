import torch
import torch.nn as nn

from .common import ConvBNAct, SepConvBNAct

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channel, out_channel, stride=1): 
    # in_channel代表输入通道数，out_channel代表输出通道数。
        super(BasicBlock, self).__init__()
        # Conv1
        self.cv1 = ConvBNAct(in_channel, out_channel, kernel=3, dilation=1, stride=stride, act=nn.ReLU(), bn=True, bias=False)
        # Conv2
        self.cv2 = ConvBNAct(out_channel, out_channel, kernel=3, dilation=1, stride=1, act=None, bn=True, bias=False)
        self.relu = nn.ReLU()
        # refine shortcut channel or downsample 
        self.downsample = ConvBNAct(in_channel, out_channel, kernel=1, dilation=1, stride=stride, bias=False, act=None, bn=True) if (in_channel != out_channel or stride > 1) else None

    def forward(self, x):
        out = self.cv1(x)
        out = self.cv2(out)

        residual = self.downsample(x) if self.downsample is not None else x
		# F(x)+x
        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4      
    def __init__(self, in_channel, out_channel, stride=1):
        super(Bottleneck, self).__init__()
        c_ = out_channel // self.expansion
        # conv1   1x1
        self.cv1 = ConvBNAct(in_channel, c_, kernel=1, dilation=1, stride=1, act=nn.ReLU(), bn=True, bias=False)
        # conv2   3x3
        self.cv2 = ConvBNAct(c_, c_, kernel=3, dilation=1, stride=stride, act=nn.ReLU(), bn=True, bias=False)
        # conv3   1x1  
        self.cv3 = ConvBNAct(c_, out_channel, kernel=1, dilation=1, stride=1, act=None, bn=True, bias=False)

        self.relu = nn.ReLU()

        # refine shortcut channel or downsample 
        self.downsample = ConvBNAct(in_channel, out_channel, kernel=1, dilation=1, stride=stride, bias=False, act=None, bn=True) if (in_channel != out_channel or stride > 1) else None

    def forward(self, x):
        out = self.cv1(x)
        out = self.cv2(out)
        out = self.cv3(out)

        residual = self.downsample(x) if self.downsample is not None else x

        out += residual
        out = self.relu(out)

        return out

