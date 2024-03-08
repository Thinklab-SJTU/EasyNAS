import os
import yaml
from copy import deepcopy
import inspect

from .utils import get_submodule_by_name
from .yaml_parser import parse_cfg, CfgLoader, CfgDumper
from .dataloader import create_dataloader 
from .optimizer import create_optimizer
from src.search_space.base import SearchSpace, _SearchSpace

def create_criterion(cfg: dict, local_rank=-1):
    cfg = deepcopy(cfg)
    cls = get_submodule_by_name(cfg.get('submodule_name'), search_path='torch.nn.criterion')
    if 'local_rank' in inspect.getfullargspec(cls.__init__).args:
        cfg.setdefault('args', {}).setdefault('local_rank', local_rank)
    return cls(**cfg.get('args', {}))

def create_scheduler(cfg: dict):
    return get_submodule_by_name(cfg.get('submodule_name'), search_path='torch.optim.lr_scheduler')(**cfg.get('args', {}))

def create_model(cfg: dict, input_size=None, root_path=None, local_rank=-1):
    cfg = deepcopy(cfg)
    if root_path and cfg['args'].get('log_path', None):
        cfg['args']['log_path'] = os.path.join(root_path, cfg['args']['log_path'])
    model = get_submodule_by_name(cfg.get('submodule_name'), search_path=['src.models'])
    return model(input_size=input_size, local_rank=local_rank, **cfg['args'])

def create_hook(cfg: dict, search_path='src.hook'):
    return get_submodule_by_name(cfg.get('submodule_name'), search_path=search_path)(**cfg.get('args', {}))

def create_search_space(cfg: dict):
    if isinstance(cfg, _SearchSpace):
        return cfg
    return get_submodule_by_name('SearchSpace', search_path='src.search_space.base')(space=cfg)

def create_searcher(cfg: dict):
    return get_submodule_by_name(cfg.get('submodule_name'), search_path='src.searcher')(**cfg.get('args', {}))

def create_contractor(cfg: dict):
    return get_submodule_by_name(cfg.get('submodule_name'), search_path='src.evaluater')(**cfg.get('args', {}))

def create_module(cfg: dict, search_path=None):
    return get_submodule_by_name(cfg.get('submodule_name'), search_path=search_path)(**cfg.get('args', {}))
