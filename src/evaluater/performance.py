import os
import torch
import numpy as np

from builder import parse_cfg, get_submodule_by_name, create_model
from src.hook import CkptHOOK
from src.search_space.base import SampleNode

def get_random(task):
    return np.random.randn(1)[0]

def get_performance(task):
    task_cfg = task.config
    print(task_cfg)
    engine_cfg = task_cfg['engine']
    engine = get_submodule_by_name(engine_cfg['submodule_name'], search_path='engines')(
                      **engine_cfg['args'],
                      )
    print("Engine is running...")
    engine.run(**engine_cfg.get('run_args', {}))
    reward = engine.extract_performance() #engine.info.results.val.best
    del engine
    torch.cuda.empty_cache()
    return reward

def get_edgeDevice_latency(task):
    if isinstance(task, SampleNode):
        task_cfg = task.config
    elif isinstance(task, dict):
        task_cfg = task
    else:
        raise(TypeError(f"No implementation for task typed as {type(task)}"))
    # get model
    engine_cfg = task_cfg['engine']
    engine = get_submodule_by_name(engine_cfg['submodule_name'], search_path='engines')(
                      **engine_cfg['args'],
                      )
    model = engine.model_without_ddp
    model.to('cpu')
    model.device = 'cpu'
    model.eval()
    # get checkpoint
    onnx_path = None
    for hook in engine._hooks:
        if isinstance(hook, CkptHOOK):
            onnx_path = os.path.join(hook.save_root, 'model.onnx')
            checkpoint = hook.get_pretrain_model(engine.device, pretrain=hook.save_root)
#            if checkpoint is not None:
#                model.load_state_dict(checkpoint['state_dict'], strict=False)

    # convert to onnx
    from src.evaluater.export import export_onnx
    onnx = export_onnx(model, onnx_path)

    # scp the command to the device that connected to the edge device.

#def get_mip_reward(task):
#    task_cfg = task.config
#    print(task_cfg)
#    engine_cfg = task_cfg['engine']
#    engine = get_submodule_by_name(engine_cfg['submodule_name'], search_path='engines')(
#                      **engine_cfg['args'],
#                      )
#    print("Engine is running...")
#    engine.run(**engine_cfg.get('run_args', {}))
#    reward = -abs(engine.info.results['sum_primal_bound'])
#    del engine
#    torch.cuda.empty_cache()
#    return reward

if __name__ == '__main__':
    cfg = 'cfg/EdgeDevice/test.yaml'
    cfg = parse_cfg(cfg)
    get_edgeDevice_latency(cfg)

