import os
import random
import argparse
import numpy as np
import torch

from builder import parse_cfg, create_dataloader, create_model, create_optimizer, create_criterion, create_submodule_from_dict, create_hook 
from pipeline.trainer import Trainer
from distribute_utils import init_distributed_mode, get_rank

parser = argparse.ArgumentParser("train")
parser.add_argument('--cfg', type=str, help='location of the config file')
parser.add_argument('--seed', default=-1, type=int,
                    help='random seed')

# distributed training parameters
parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
parser.add_argument('--dist_backend', default='nccl', help='backend of distributed training')

args = parser.parse_args()

def main():
    # fix the seed for reproducibility
    args.seed = args.seed if args.seed >= 0 else random.randint(0, 1e4)
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    init_distributed_mode(args)
    print(args)

    # read the config file
    cfg = parse_cfg(args.cfg)
    if args.local_rank in [0, -1]:
        for k, v in cfg.items():
            print(k, v)
        if cfg.get('root_path', None):
            os.makedirs(cfg.get('root_path'), exist_ok=True)

    # build submodules
    # build data
    print("Building dataloader")
    datasets, dataloaders = create_dataloader(cfg['data'])

    # parse model
    if args.local_rank in [0, -1]:
        print("Building model")
    if cfg.get('root_path') and cfg['model'].get('log_path', None):
        cfg['model']['log_path'] = os.path.join(cfg['root_path'], cfg['model']['log_path'])
    model = create_model(cfg['model'], num_classes=cfg['data']['num_classes'], input_size=cfg['data'].get('input_size', None), log_path=cfg['model'].get('log_path', None), local_rank=args.local_rank)

    # parse criterion
    if args.local_rank in [0, -1]:
        print("Building criterion")
    criterion = create_criterion(cfg['criterion'])

    # parse optimizer
    if args.local_rank in [0, -1]:
        print("Building optimizer")
    optimizer = create_optimizer(model, cfg['optimizer'])
    opt_hook = cfg.get('opt_hook', None)
    if opt_hook:
        opt_hook['hook_args']['optimizer'] = optimizer
        opt_hook = create_hook(opt_hook)

    # parse scheduler
    if args.local_rank in [0, -1]:
        print("Building lr scheduler")
    cfg['lr_scheduler']['args']['optimizer'] = optimizer
    scheduler = create_submodule_from_dict(cfg['lr_scheduler'])
    scheduler_hook = cfg.get('lr_scheduler_hook', None)
    if scheduler_hook:
        scheduler_hook['hook_args']['lr_scheduler'] = scheduler
        scheduler_hook = create_hook(scheduler_hook)

    # parse other hooks
    if args.local_rank in [0, -1]:
        print("Building hooks")
    hooks = []
    for k, v in cfg.items():
        if 'hook' in k and ((not v.get('hook_args', {}).get('only_master', False)) or args.local_rank in [-1, 0]):
            hooks.append(create_hook(v))


    # build trainer
    trainer = Trainer(dataloaders=dataloaders, 
                      model=model, 
                      criterion=criterion, 
                      optimizer=opt_hook,
                      lr_scheduler=scheduler_hook,
                      hooks=hooks,
                      local_rank=args.local_rank,
                      amp=cfg['amp']
                      )
    if args.local_rank in [0, -1]:
        print("Training...")
    trainer.run(cfg['epoch'])
    

if __name__ == '__main__':
    main()
