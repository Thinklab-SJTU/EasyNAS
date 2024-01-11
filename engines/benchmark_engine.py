import math
from easydict import EasyDict as edict
from typing import Union, List

from .base import BaseEngine
from builder import create_module, create_hook, create_optimizer, create_scheduler
from src.hook import hooks_run, hooks_iter

class BenchmarkEngine(BaseEngine):
    def __init__(self, obj, optimizer, lr_scheduler=None, hooks=tuple()):
        self.obj, self.optimizer, self.lr_scheduler = self.build_from_cfg(obj, optimizer, lr_scheduler, hooks)
        self.info = edict({
            'results': edict({}), 
            })

    def build_from_cfg(self, obj, optimizer_cfg, lr_scheduler_cfg, hooks_cfg):
        # build obj
        print("Building the object of the optimization")
        if isinstance(obj, dict):
            obj = create_module(obj, search_path='src.benchmark.object')
        elif isinstance(obj, str):
            obj = create_module({'submodule_name': 'Benchmark_function', 'args': obj}, search_path='src.benchmark.object')

        # build optimizer
        if optimizer_cfg:
            print("Building optimizer")
            optimizer = create_optimizer(obj, optimizer_cfg)
        else: optimizer = None

        # build scheduler
        if lr_scheduler_cfg:
            print("Building lr scheduler")
            lr_scheduler_cfg['args']['optimizer'] = optimizer
            lr_scheduler = create_scheduler(lr_scheduler_cfg)
        else: lr_scheduler = None

        # build other hooks
        print("Building hooks")
        self._hooks = []
        gen = hooks_cfg.values() if isinstance(hooks_cfg, dict) else iter(hooks_cfg)
        for v in gen:
            print(v)
            self.register_hook(create_hook(v, search_path=['src.hook', 'src.benchmark']))
        return obj, optimizer, lr_scheduler

    def run(self, max_iter):
        with hooks_run(self._hooks, self):
            self.info.results.ignore_obj_list = []
            for step in range(max_iter):
                self.info.current_iter = step
                with hooks_iter(self._hooks, self):
                    self.info.results.obj = self.obj()
                    self.info.results.ignore_obj_list.append(self.info.results.obj.item())
                    params_require_grad = []
                    for pg in self.optimizer.param_groups:
                        params_require_grad.extend(pg['params'])
                    if not getattr(self.optimizer, 'ZO', False):
                        self.info.results.obj.backward(inputs=params_require_grad)
                    # get best. it should be put to a hook in the future
                    if self.info.results.get('best', math.inf) > self.info.results.obj:
                        self.info.results.best = self.info.results.obj
#        print(self.info.results.ignore_obj_list)

    def extract_performance(self):
        return self.info.results.get('best')
