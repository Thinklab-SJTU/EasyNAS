import os
import random
import argparse
import numpy as np
import torch
#torch.backends.cudnn.deterministic = True

from builder import parse_cfg, get_submodule_by_name
from distribute_utils import ddp_ctx, get_rank

parser = argparse.ArgumentParser("Search")
parser.add_argument('--cfg', type=str, help='location of the config file')
parser.add_argument('--seed', default=-1, type=int,
                    help='random seed')

args = parser.parse_args()

def main(args):
    # fix the seed for reproducibility
    args.seed = args.seed if args.seed >= 0 else random.randint(0, 1e4)
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    print(args)
    
    # read the config file
    cfg = parse_cfg(args.cfg)
    for k, v in cfg.items():
        if k == 'engine':
            print("\n", k, v)
    if cfg.get('root_path', None):
        os.makedirs(cfg.get('root_path'), exist_ok=True)

    # build engine
    engine_cfg = cfg['engine']
    engine = get_submodule_by_name(engine_cfg['submodule_name'], search_path='engines')(
                      **engine_cfg['args'],
                      )
    print("Engine is running...")
    engine.run()
    

if __name__ == '__main__':
    main(args)

