import os
import time
import yaml
import random
import argparse
from collections import OrderedDict, namedtuple
import numpy as np
import torch
#torch.backends.cudnn.deterministic = True

from builder import parse_cfg, get_submodule_by_name, CfgDumper

parser = argparse.ArgumentParser("Run")
parser.add_argument('--cfg', type=str, help='location of the config file')
parser.add_argument('--seed', default=-1, type=int,
                    help='random seed')
args = parser.parse_args()

def run_engine(cfg):
    engine_cfg = cfg['engine']
    engine = get_submodule_by_name(engine_cfg['submodule_name'], search_path='engines')(
                      **engine_cfg['args'],
                      )
    print("Engine is running...")
    engine.run(**engine_cfg.get('run_args', {}))
    return engine.info.results, engine

def save_yaml(yaml_file, data):
    with open(yaml_file, encoding='utf-8', mode='w') as f:
        try:
            yaml.dump(data=data, stream=f, allow_unicode=True, Dumper=CfgDumper, default_flow_style=False)
        except Exception as e:
            raise(e)

def ablation_bound(cfg, save_path):
    Setting = namedtuple('Setting', ['fn', 'sifParams', 'lr'])
    
    settings = [
            Setting('ARGTRIGLS', {'N':200}, 5e-3),
#            Setting('CHNROSNB', {'N':50}, 1e-2),
#            Setting('ROSENBR', None, 1e-1),
#            Setting('COATING', None, 1e-2),
#            Setting('MANCINO', {'N':100}, 1e-4),
#            Setting('BOXPOWER', {'N':1000}, 1e-3),
#            Setting('SROSENBR', {'N/2':250}, 5e-3),
#            Setting('BROYDNBDLS', {'N':50}, 1e-2),
            ]
    
    # ablation for bound
    os.makedirs(save_path, exist_ok=True)
#    bounds = [0, 1, 2, 5, 10]
#    bounds = list(range(0, 10))
    bounds = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
    data = {}
    for bound in bounds:
        data[bound] = {}
        for setting in settings:
            cfg['obj']['args']['fn_names'] = [setting.fn]
            cfg['obj']['args']['sifParams'] = [setting.sifParams]
            cfg['optimizer']['args']['lr'] = setting.lr
            cfg['optimizer']['args']['reuse_distance_bound'] = bound * setting.lr
    
            start = time.time()
            results, engine = run_engine(cfg)
            end = time.time()
            data[bound][setting.fn] = {
                    'obj': results.ignore_obj_list,
                    'best': results.ignore_best,
                    'fn': setting.fn,
                    'sifParams': setting.sifParams,
                    'lr': setting.lr,
                    'num_reuse': engine.optimizer.num_reuse,
                    'num_sample': cfg['optimizer']['args']['num_sample_per_step'],
                    'num_iter': cfg['engine']['run_args']['max_iter'],
                    'time': end-start,
                    }
        print(f"bound = {bound}")
        bests = []
        for key, v in data[bound].items():
            bests.append((key, v['best']))
        print(bests)
    yaml_file = os.path.join(save_path, 'ablation_bound_withtime_test1.yaml')
    save_yaml(yaml_file, data)

def ablation_num_sample(cfg, save_path, file_name):
    # ablation for num_sample_per_step
    Setting = namedtuple('Setting', ['fn', 'sifParams', 'lr', 'N'])
    
#    settings = [Setting('ARGTRIGLS', {'N':200}, 5e-3, 200),
#            Setting('CHNROSNB', {'N':50}, 1e-2, 50),
#            Setting('COATING', None, 1e-2, 134),
#            Setting('MANCINO', {'N':100}, 1e-4, 100),
#            Setting('BOXPOWER', {'N':1000}, 1e-3, 1000),
#            Setting('SROSENBR', {'N/2':250}, 5e-3, 500),
#            Setting('BROYDNBDLS', {'N':50}, 1e-2, 50),
#            ]

#    # zo-sgd
#    settings = [Setting('ARGTRIGLS', {'N':200}, 5e-4, 200),
#            Setting('CHNROSNB', {'N':50}, 1e-2, 50),
#            Setting('COATING', None, 5e-3, 134),
#            Setting('MANCINO', {'N':100}, 1e-4, 100),
#            Setting('BOXPOWER', {'N':1000}, 5e-3, 1000),
#            Setting('SROSENBR', {'N/2':250}, 2e-2, 500),
#            Setting('BROYDNBDLS', {'N':50}, 2e-2, 50),
#            ]
#    # zo-signsgd
#    settings = [Setting('ARGTRIGLS', {'N':200}, 1e-4, 200),
#            Setting('CHNROSNB', {'N':50}, 2e-3, 50),
#            Setting('COATING', None, 1e-3, 134),
#            Setting('MANCINO', {'N':100}, 1e-4, 100),
#            Setting('BOXPOWER', {'N':1000}, 5e-4, 1000),
#            Setting('SROSENBR', {'N/2':250}, 5e-4, 500),
#            Setting('BROYDNBDLS', {'N':50}, 2e-3, 50),
#            ]
    # zo-adam
    settings = [Setting('ARGTRIGLS', {'N':200}, 2e-4, 200),
            Setting('CHNROSNB', {'N':50}, 5e-2, 50),
            Setting('COATING', None, 5e-2, 134),
            Setting('MANCINO', {'N':100}, 1e-4, 100),
            Setting('BOXPOWER', {'N':1000}, 1e-2, 1000),
            Setting('SROSENBR', {'N/2':250}, 5e-3, 500),
            Setting('BROYDNBDLS', {'N':50}, 1e-2, 50),
            ]
#    # lizo
#    settings = [Setting('ARGTRIGLS', {'N':200}, 1e-4, 200),
#            Setting('CHNROSNB', {'N':50}, 5e-3, 50),
#            Setting('COATING', None, 2e-2, 134),
#            Setting('MANCINO', {'N':100}, 1e-4, 100),
#            Setting('BOXPOWER', {'N':1000}, 1e-3, 1000),
#            Setting('SROSENBR', {'N/2':250}, 5e-3, 500),
#            Setting('BROYDNBDLS', {'N':50}, 1e-2, 50),
#            ]
    
    os.makedirs(save_path, exist_ok=True)
    num_samples = [6, 8, 10, 20, 50, 100, 200]
    data = {}
    for setting in settings:
        data[setting.fn] = {}
        cfg['obj']['args']['fn_names'] = [setting.fn]
        cfg['obj']['args']['sifParams'] = [setting.sifParams]
        cfg['optimizer']['args']['lr'] = setting.lr
        if 'reuse_distance_bound' in cfg['optimizer']['args']:
            cfg['optimizer']['args']['reuse_distance_bound'] = 2 * setting.lr
        for num_sample in num_samples:
            if setting.N < num_sample: break
            cfg['optimizer']['args']['num_sample_per_step'] = num_sample
    
            results, _ = run_engine(cfg)
            data[setting.fn][num_sample] = {
                    'obj': results.ignore_obj_list,
                    'best': results.ignore_best,
                    'fn': setting.fn,
                    'sifParams': setting.sifParams,
                    'lr': setting.lr,
                    }
        print(f"setting = {setting}")
        bests = []
        for key, v in data[setting.fn].items():
            bests.append((key, v['best']))
        print(bests)
    yaml_file = os.path.join(save_path, file_name)
    save_yaml(yaml_file, data)

def ablation_bound_numsample(cfg, save_path, file_name):
    # ablation for num_sample_per_step
    Setting = namedtuple('Setting', ['fn', 'sifParams', 'lr', 'N'])
    
    settings = [Setting('ARGTRIGLS', {'N':200}, 5e-3, 200),
            Setting('CHNROSNB', {'N':50}, 1e-2, 50),
            Setting('COATING', None, 1e-2, 134),
            Setting('MANCINO', {'N':100}, 1e-4, 100),
            Setting('BOXPOWER', {'N':1000}, 1e-3, 1000),
            Setting('SROSENBR', {'N/2':250}, 5e-3, 500),
            Setting('BROYDNBDLS', {'N':50}, 1e-2, 50),
            ]
    
    os.makedirs(save_path, exist_ok=True)
    num_samples = [6, 8, 10, 20, 50, 100, 200]
    bounds = [0, 1, 2, 5]
    data = {}
    for setting in settings:
        data[setting.fn] = {}
        cfg['obj']['args']['fn_names'] = [setting.fn]
        cfg['obj']['args']['sifParams'] = [setting.sifParams]
        cfg['optimizer']['args']['lr'] = setting.lr
        for num_sample in num_samples:
            data[setting.fn][num_sample] = {}
            if setting.N < num_sample: break
            for bound in bounds:
                print(f'setting={setting}, num_sample={num_sample}, bound={bound}')
                cfg['optimizer']['args']['reuse_distance_bound'] = bound * setting.lr
                cfg['optimizer']['args']['num_sample_per_step'] = num_sample
        
                results, engine = run_engine(cfg)
                data[setting.fn][num_sample][bound] = {
                        'obj': results.ignore_obj_list,
                        'best': results.ignore_best,
                        'fn': setting.fn,
                        'sifParams': setting.sifParams,
                        'lr': setting.lr,
                        'num_reuse': engine.optimizer.num_reuse,
                        'num_sample': num_sample,
                        'num_iter': cfg['engine']['run_args']['max_iter']
                        }
        print(f"setting = {setting}")
    yaml_file = os.path.join(save_path, file_name)
    save_yaml(yaml_file, data)

def ablation_lr(cfg, save_path, file_name):
    # ablation for num_sample_per_step
    Setting = namedtuple('Setting', ['fn', 'sifParams'])
    
    settings = [
            Setting('ARGTRIGLS', {'N':200}),
            Setting('CHNROSNB', {'N':50}),
            Setting('COATING', None),
            Setting('MANCINO', {'N':100}),
            Setting('BOXPOWER', {'N':1000}),
            Setting('SROSENBR', {'N/2':250}),
            Setting('BROYDNBDLS', {'N':50}),
            ]
    
    os.makedirs(save_path, exist_ok=True)
    lrs = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 5e-1]
    data = {}
    for setting in settings:
        data[setting.fn] = {}
        cfg['obj']['args']['fn_names'] = [setting.fn]
        cfg['obj']['args']['sifParams'] = [setting.sifParams]
        for lr in lrs:
            print(f"setting={setting}, lr={lr}")
            cfg['optimizer']['args']['lr'] = lr
            if 'reuse_distance_bound' in cfg['optimizer']['args']:
                cfg['optimizer']['args']['reuse_distance_bound'] = 2 * lr
    
            results, _ = run_engine(cfg)
            data[setting.fn][lr] = {
                    'obj': results.ignore_obj_list,
                    'best': results.ignore_best,
                    'fn': setting.fn,
                    'sifParams': setting.sifParams,
                    }
        print(f"setting = {setting}")
        bests = []
        for key, v in data[setting.fn].items():
            bests.append((key, v['best']))
        print(bests)
    yaml_file = os.path.join(save_path, file_name)
    save_yaml(yaml_file, data)

args.seed = args.seed if args.seed >= 0 else random.randint(0, 1e4)
seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)

cfg = parse_cfg(args.cfg)

save_path = 'runs/pycutest/'

ablation_bound(cfg, save_path)
#for i in range(4,8):
#    ablation_num_sample(cfg, save_path, f'ablation_num_sample_test{i}.yaml')
#for i in range(0,3):
#    ablation_lr(cfg, save_path, f'ablation_lr_test{i}.yaml')
#ablation_bound_numsample(cfg, save_path, f'ablation_bound_numsample_test{i}.yaml')


