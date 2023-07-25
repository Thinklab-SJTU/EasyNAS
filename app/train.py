import os
import random
import argparse
import numpy as np
import torch
torch.backends.cudnn.deterministic = True

from builder import parse_cfg, create_dataloader, create_model, create_optimizer, create_criterion, create_hook, create_scheduler
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
            print("\n", k, v)
        if cfg.get('root_path', None):
            os.makedirs(cfg.get('root_path'), exist_ok=True)

    # build submodules
    # build data
    print("Building dataloader")
    datasets, dataloaders = create_dataloader(cfg['data'])

    # parse model
    print("Building model")
#    assert cfg['data']['num_classes'] == cfg['model']['args']['output_ch']
    model = create_model(cfg['model'], input_size=cfg['data'].get('input_size', None), local_rank=args.local_rank)

    # parse criterion
    print("Building criterion")
    criterion = create_criterion(cfg['criterion'], local_rank=args.local_rank)

    # parse optimizer
    print("Building optimizer")
    optimizer = create_optimizer(model, cfg['optimizer'], criterion)

    # parse scheduler
    print("Building lr scheduler")
    cfg['lr_scheduler']['args']['optimizer'] = optimizer
    scheduler = create_scheduler(cfg['lr_scheduler'])

    # parse other hooks
    print("Building hooks")
    hooks = []
    for k, v in cfg['hooks'].items():
        if (not v.get('args', {}).get('only_master', False)) or args.local_rank in [-1, 0]:
            hooks.append(create_hook(v))


    # build trainer
    trainer = Trainer(dataloaders=dataloaders, 
                      model=model, 
                      criterion=criterion, 
                      optimizer=optimizer,
                      lr_scheduler=scheduler,
                      hooks=hooks,
                      local_rank=args.local_rank,
                      amp=cfg['amp']
                      )
    print("Training...")
    trainer.run(cfg['epoch'])
    

if __name__ == '__main__':
    main()
