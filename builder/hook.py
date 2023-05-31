import inspect
import copy
from typing import Union, List
from collections import namedtuple

import numpy as np
import torch

from .utils import get_submodule


def create_hook(cfg: dict):
    submodule = get_submodule(cfg.get('hook_name'), cfg.get('module_name', 'src.hook'), cfg.get('package_name', None))
    return submodule(**cfg.get('hook_args', {}))
		

