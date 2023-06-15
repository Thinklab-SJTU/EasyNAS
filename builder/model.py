import copy
from typing import Union, List
from collections import namedtuple

import numpy as np
import torch

from .utils import get_submodule

def get_model(name: str, module_name: str=None, package_path: str=None):
    if module_name:
        return get_submodule(name, module_name, package_path)
    return get_submodule()

def create_model(cfg: dict, num_classes, input_size=None, log_path=None, local_rank=-1):
    model = get_model(cfg.get('name'), cfg.get('module_name', None), cfg.get('package_path', None))
    return model(cfg=cfg, output_ch=num_classes, input_size=input_size, log_path=log_path, local_rank=local_rank)
		

