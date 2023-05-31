import copy
from typing import Union, List
from collections import namedtuple

import numpy as np
import torch

from .utils import get_submodule

def load_criterion(name: str, module_name: str=None, package_path: str=None):
    if module_name:
        return get_submodule(name, module_name, package_path)
    try:
        return getattr(torch.nn.criterion, name)
    except:
        raise(ValueError(f"No criterion named as {name} in torch.nn.criterion or the given module"))

def create_criterion(cfg: dict):
    criterion = load_criterion(cfg.get('name'), cfg.get('module_name', None), cfg.get('package_path', None))
    return criterion(**cfg.get('criterion_args', {}))
		
