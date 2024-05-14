import copy
from typing import Union, List
from collections import namedtuple

import numpy as np
import torch

from .utils import get_submodule, get_submodule_by_name
from app.distribute_utils import is_dist_avail_and_initialized, get_world_size, get_rank


def create_dataloader(cfg: dict) -> dict:
    cfg = copy.deepcopy(cfg)

    dataset_cfg = cfg['dataset']
    dataloader_cfg = cfg['dataloader']
    # build dataset
    datasets = {}
    for set_name, set_cfg in dataset_cfg.items():
       	cfg = copy.deepcopy(set_cfg)
       	submodule_name = cfg.pop('submodule_name')
        Dataset = get_submodule_by_name(submodule_name, search_path='src.datasets')
        datasets[set_name] = Dataset(**cfg.get('dataset_args', {}))


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
            try:
                info = splitInfos[set_name]
            except KeyError as e:
                start, num_train = 0, len(dataset)
#       	 indices = np.random.permutation(num_train)
       	        indices = list(range(num_train))
       	        splitInfos[set_name] = splitInfo(indices=indices, start=start)
            else:
                indices, start, num_train = info.indices, info.start, len(info.indices)
#       	    info = splitInfos.get(set_name, None)
#       	    if info is None:
#                num_train = len(dataset)
#       	        indices = np.random.permutation(num_train)
#       	        start = 0
#       	        splitInfos[set_name] = splitInfo(indices=indices, start=start)
#       	    else: 
#                indices, start, num_train = info.indices, info.start, len(info.indices)

            end = start + int(np.floor(portion * num_train))
            if end <= num_train:
                tmp_indices = indices[start:end]
            else:
                end = end % num_train
                tmp_indices = indices[start:] + indices[:end]
            splitInfos[set_name] = splitInfos[set_name]._replace(start=end)
            dataset = torch.utils.data.Subset(dataset, indices=tmp_indices)

       	shuffle =  cfg.pop('shuffle', True)
        if is_dist_avail_and_initialized() and cfg.get('use_dist', True):
            world_size = get_world_size()
            rank = get_rank()
            sampler = torch.utils.data.DistributedSampler(
                dataset, num_replicas=world_size, rank=rank, shuffle=shuffle
            )
        else:
            sampler = torch.utils.data.RandomSampler(dataset) if shuffle else None

       	submodule_name = cfg.pop('submodule_name', 'torch.utils.data.DataLoader')
        Dataloader = get_submodule_by_name(submodule_name, search_path='src.datasets')
        dataloaders[loader_name] = Dataloader(dataset, sampler=sampler, **cfg.get('dataloader_args', {}))
        dataloaders[loader_name].cfg = loader_cfg
    return datasets, dataloaders
		
