import os
import yaml
from copy import deepcopy

from .utils import CfgLoader, parse_cfg, get_submodule_by_name
from .dataloader import create_dataloader 
from .optimizer import create_optimizer

def create_criterion(cfg: dict):
    return get_submodule_by_name(cfg.get('submodule_name'), search_path='torch.nn.criterion')(**cfg.get('args', {}))

def create_scheduler(cfg: dict):
    return get_submodule_by_name(cfg.get('submodule_name'), search_path='torch.optim.lr_scheduler')(**cfg.get('args', {}))

def create_model(cfg: dict, input_size=None, root_path=None, local_rank=-1):
    if root_path and cfg['args'].get('log_path', None):
        cfg['args']['log_path'] = os.path.join(root_path, cfg['args']['log_path'])
    model = get_submodule_by_name(cfg.get('submodule_name'), search_path=['src.models'])
    return model(input_size=input_size, local_rank=local_rank, **cfg['args'])

def create_hook(cfg: dict):
    return get_submodule_by_name(cfg.get('submodule_name'), search_path='src.hook')(**cfg.get('args', {}))
