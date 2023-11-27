import torch
import numpy as np

from builder import get_submodule_by_name

def get_random(task):
    return np.random.randn(1)[0]

def get_performance(task):
    task_cfg = task.cfg
    print(task_cfg)
    engine_cfg = task_cfg['engine']
    engine = get_submodule_by_name(engine_cfg['submodule_name'], search_path='engines')(
                      **engine_cfg['args'],
                      )
    print("Engine is running...")
    if task_cfg.get('epoch', 0):
        engine.run(task_cfg['epoch'])
    else:
        engine.validate()
    reward = engine.info.results.val.best
    del engine
    torch.cuda.empty_cache()
    return reward

def get_mip_reward(task):
    task_cfg = task.config
    print(task_cfg)
    engine_cfg = task_cfg['engine']
    engine = get_submodule_by_name(engine_cfg['submodule_name'], search_path='engines')(
                      **engine_cfg['args'],
                      )
    print("Engine is running...")
    engine.run()
    reward = -abs(engine.info.results['sum_primal_bound'])
    del engine
    torch.cuda.empty_cache()
    return reward
