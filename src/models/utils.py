import math
import numpy as np
import torch.nn as nn

def count_parameters_in_MB(model):
  return np.sum(v.numel() for name, v in model.named_parameters() if "auxiliary" not in name)/1e6

def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor

def default_init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
#        m.eps = 1e-3
#        m.momentum = 0.03
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.constant_(m.weight, 1)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0)
#    elif isinstance(m, (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6)):
#        m.inplace = True

def yolov5_init_weights(m):
    if isinstance(m, nn.BatchNorm2d):
        m.eps = 1e-3
        m.momentum = 0.03
#    elif isinstance(m, (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6)):
#        m.inplace = True
