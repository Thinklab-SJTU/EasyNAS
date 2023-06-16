import torch
from torch import nn as nn

from .common import ConvBNAct, SepConvBNAct, GlobalPoolBNAct, PoolBNAct
from .search_common import ConvBNAct_search, SepConvBNAct_search
from .resnet import BasicBlock, Bottleneck
from .mobilenet import InvertedResidual

NEED_INOUTC_OPs = ("ConvBNAct_search", "SepConvBNAct_search", "ConvBNAct", "SepConvBNAct")
MULTIALPHA_OPs = ("ConvBNAct_search", "SepConvBNAct_search")


