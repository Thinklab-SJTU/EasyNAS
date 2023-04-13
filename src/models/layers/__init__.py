import torch
import torch.nn as nn

from .utils import get_submodule
from .common import ConvBNAct, SepConvBNAct
from .search_common import ConvBNAct_search, SepConvBNAct_search

NEED_INOUTC_OPs = ("ConvBNAct_search", "SepConvBNAct_search", "ConvBNAct", "SepConvBNAct")
MULTIALPHA_OPs = ("ConvBNAct_search", "SepConvBNAct_search")


