import os
import yaml
from copy import deepcopy

from .criterion import create_criterion
from .model import create_model
from .dataloader import create_dataloader
from .hook import create_hook
from .utils import CfgLoader, create_submodule_from_dict


def parse_cfg(yaml_file):
    with open(yaml_file, 'r') as f:
        tmp_cfg = yaml.load(f.read(), CfgLoader)

    if isinstance(tmp_cfg, dict):
        cfg = {}
        for k, v in tmp_cfg.items():
            cfg[k] = parse_cfg(v) if isinstance(v, str) and os.path.isfile(v) else v
    elif isinstance(tmp_cfg, list):
        cfg = []
        for v in tmp_cfg:
            cfg.append(parse_cfg(v) if isinstance(v, str) and os.path.isfile(v) else v)
    else:
        cfg = deepcopy(tmp_cfg)

    return cfg

