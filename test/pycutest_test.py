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
    bounds = [0] #0.01, 0.05, 0.1, 0.5, 1, 2]
    data = {}
    for bound in bounds:
        data[bound] = {}
        for setting in settings:
            cfg['obj']['args']['fn_names'] = [setting.fn]
            cfg['obj']['args']['sifParams'] = [setting.sifParams]
            cfg['optimizer']['args']['lr'] = setting.lr
            cfg['optimizer']['args']['reuse_distance_bound'] = bound * setting.lr
            cfg['engine']['run_args']['max_iter'] = 2000
    
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
                    'num_sample': int(cfg['optimizer']['args']['num_sample_per_step']),
                    'num_iter': int(cfg['engine']['run_args']['max_iter']),
                    'time': end-start,
                    }
        print(f"bound = {bound}")
        bests = []
        for key, v in data[bound].items():
            bests.append((key, v['best']))
        print(f"best: {bests}, reuse_rate: {sum(engine.optimizer.num_reuse)/(int(cfg['optimizer']['args']['num_sample_per_step']) * int(cfg['engine']['run_args']['max_iter']))}")
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
#            Setting('ARGTRIGLS', {'N':200}),
#            Setting('CHNROSNB', {'N':50}),
            Setting('COATING', None),
#            Setting('MANCINO', {'N':100}),
#            Setting('BOXPOWER', {'N':1000}),
#            Setting('BOXPOWER', {'N':10000}),
#            Setting('BOXPOWER', {'N':20000}),
#            Setting('SROSENBR', {'N/2':250}),
#            Setting('SROSENBR', {'N/2':500}),
#            Setting('SROSENBR', {'N/2':2500}),
#            Setting('SROSENBR', {'N/2':5000}),
#            Setting('BROYDNBDLS', {'N':50}),
#            Setting('BROYDNBDLS', {'N':100}),
#            Setting('BROYDNBDLS', {'N':500}),
#            Setting('BROYDNBDLS', {'N':1000}),
#            Setting('BROYDNBDLS', {'N':5000}),
#            Setting('BROYDNBDLS', {'N':10000}),
            ]
    
    os.makedirs(save_path, exist_ok=True)
#    lrs = [1e-6, 2e-6, 3e-6, 4e-6, 5e-6, 6e-6, 7e-6, 8e-6, 9e-6, 1e-5]
    lrs = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1 , 2e-1, 5e-1]
    data = {}
    for setting in settings:
#        fn_name = setting.fn
        if setting.sifParams:
            fn_name = f'{setting.fn}-{list(setting.sifParams.values())[0]}'
        else: fn_name = setting.fn
        data[fn_name] = {}
        cfg['obj']['args']['fn_names'] = [setting.fn]
        cfg['obj']['args']['sifParams'] = [setting.sifParams]
        for lr in lrs:
            print(f"setting={setting}, lr={lr}")
            cfg['optimizer']['args']['lr'] = lr
            if 'reuse_distance_bound' in cfg['optimizer']['args']:
                cfg['optimizer']['args']['reuse_distance_bound'] = 2 * lr
#            if 'line_search_fn' in cfg['optimizer']['args']:
#                cfg['optimizer']['args']['line_search_fn'] = None
    
            results, engine = run_engine(cfg)
            data[fn_name][lr] = {
                    'obj': results.ignore_obj_list,
                    'best': results.ignore_best,
                    'fn': setting.fn,
                    'sifParams': setting.sifParams,
                    'num_reuse': getattr(engine.optimizer, 'num_reuse', []),
                    'num_line_search_query': getattr(engine.optimizer, 'num_line_search_query', []),
                    'num_sample': cfg['optimizer']['args']['num_sample_per_step'],
                    }
            print('Line Search Query Num', sum(getattr(engine.optimizer, 'num_line_search_query', [])), getattr(engine.optimizer, 'num_line_search_query', []))
            print('Num Reuse', sum(getattr(engine.optimizer, 'num_reuse', [])), getattr(engine.optimizer, 'num_reuse', []))
        print(f"setting = {setting}")
        bests = []
        for key, v in data[fn_name].items():
            bests.append((key, v['best']))
        print('Best: ', bests)
    yaml_file = os.path.join(save_path, file_name)
    save_yaml(yaml_file, data)
    return data

args.seed = args.seed if args.seed >= 0 else random.randint(0, 1e4)
seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)

cfg = parse_cfg(args.cfg)

save_path = 'runs/pycutest_Param/'

ablation_bound(cfg, save_path)
#for i in range(4,8):
#    ablation_num_sample(cfg, save_path, f'ablation_num_sample_test{i}.yaml')
#for i in range(0,3):
#    ablation_lr(cfg, save_path, f'ablation_lr_test{i}.yaml')
#ablation_bound_numsample(cfg, save_path, f'ablation_bound_numsample_test{i}.yaml')

#bests = {}
#num_query_from_line_search = {}
#num_reuse = {}
#result = {}
#for i in range(3,4):
#    data = ablation_lr(cfg, save_path, f'ablation_lr_test{i}.yaml')
#    for fn_name, fn_data in data.items():
#        _best = min([v['best'] for v in fn_data.values()])
#        bests.setdefault(fn_name, []).append(_best)
#        result[fn_name] = [v['best'] for v in fn_data.values()]
#        num_query_from_line_search[fn_name] = [sum(v['num_line_search_query']) for v in fn_data.values()]
#        num_reuse[fn_name] = [sum(v['num_reuse']) for v in fn_data.values()]
#for fn_name, _bests in bests.items():
#    _bests = np.array(_bests)
#    print(fn_name)
##    print(fn_name, _bests, _bests.mean(), _bests.std())
#    print('Result', result[fn_name])
#    print('Num line search query', num_query_from_line_search[fn_name])
#    print('Num reuse', num_reuse[fn_name])

