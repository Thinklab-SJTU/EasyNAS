from builder import get_submodule_by_name
import numpy as np

def get_test(task):
    return np.random.randn(1)[0]

def get_performance(task):
    print('='*20)
    task_cfg = task.cfg
    print(task_cfg)
    engine_cfg = task_cfg['engine']
    engine = get_submodule_by_name(engine_cfg['submodule_name'], search_path='engines')(
                      **engine_cfg['args'],
                      )
    print("Engine is running...")
    if engine_cfg.get('epoch', 0):
        engine.run(engin_cfg['epoch'])
    else:
        engine.validate()
    print('='*20)
    return engine.info.results.val.best
