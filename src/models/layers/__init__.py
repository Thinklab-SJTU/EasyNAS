import torch
from torch import nn as nn

from .common import ConvBNAct, SepConvBNAct, GlobalPoolBNAct, PoolBNAct, FuseLayer, Concat, Focus, SPP, FactorizedReduce, LinearBNAct, Zero
from .search_common import ConvBNAct_search, SepConvBNAct_search, AtomSearchModule, SPP_search
from .resnet import BasicBlock, Bottleneck
from .mobilenet import InvertedResidual

from .yolov5 import YOLODetect, YOLOBottleneck, YOLOBottleneckCSP, YOLOC3
from .search_yolo import YOLOBottleneck_search, YOLOC3_search, YOLODetect_search

from .darts_cell import darts_conv, darts_identity, Cell_search

#NEED_INOUTC_OPs = ("ConvBNAct_search", "SepConvBNAct_search", "ConvBNAct", "SepConvBNAct")
#MULTIALPHA_OPs = ("ConvBNAct_search", "SepConvBNAct_search")


