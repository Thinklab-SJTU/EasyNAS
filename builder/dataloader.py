import copy
from typing import Union, List
from collections import namedtuple

import numpy as np
import torch

from .utils import get_submodule
from app.distribute_utils import is_dist_avail_and_initialized, get_world_size, get_rank


def build_one_dataset(submodule_name: str, module_name: str='dataset.datasets', package_path: str=None, **args_dict) -> torch.utils.data.Dataset:

    Dataset = get_submodule(submodule_name, module_name, package_path)
    return Dataset(**args_dict)


def create_dataloader(cfg: dict) -> dict:
    dataset_cfg = cfg['dataset']
    dataloader_cfg = cfg['dataloader']
    # build dataset
    datasets = {}
    for set_name, set_cfg in dataset_cfg.items():
       	cfg = copy.deepcopy(set_cfg)
       	submodule_name = cfg.pop('submodule_name', 'Dataset')
       	module_name = cfg.pop('module_name', 'dataset.datasets')
       	package_path = cfg.pop('package_path', None)
        datasets[set_name] = build_one_dataset(submodule_name, module_name, package_path, **cfg.get('dataset_args', {}))

    # build dataloader
    dataloaders = {}
    splitInfo = namedtuple('splitInfo', ['indices', 'start'])
    splitInfos = {}
    for loader_name, loader_cfg in dataloader_cfg.items():
        cfg = copy.deepcopy(loader_cfg)
        set_name = cfg.pop('dataset_name', loader_name)
        dataset = datasets[set_name]
        # train_portion to split the original dataset
       	portion =  cfg.pop('portion', None)
        if portion:
       	    info = splitInfos.get(set_name, None)
       	    if info is None:
                num_train = len(dataset)
       	        indices = np.random.permutation(num_train)
       	        start = 0
       	        splitInfos[set_name] = splitInfo(indices=indices, start=start)
       	    else: 
                indices, start, num_train = info.indices, info.start, len(info.indices)
            end = start + int(np.floor(portion * num_train))
            splitInfos[set_name] = splitInfos[set_name]._replace(start=end)
            dataset = torch.utils.data.Subset(dataset, indices=indices[start:end])

        if is_dist_avail_and_initialized() and cfg.get('use_dist', True):
            world_size = get_world_size()
            rank = get_rank()
            sampler = torch.utils.data.DistributedSampler(
                dataset, num_replicas=world_size, rank=rank, shuffle=True
            )
        else:
            sampler = torch.utils.data.RandomSampler(dataset)
        dataloaders[loader_name] = torch.utils.data.DataLoader(dataset, sampler=sampler, **cfg.get('dataloader_args', {}))
        dataloaders[loader_name].cfg = loader_cfg
    return datasets, dataloaders
		
