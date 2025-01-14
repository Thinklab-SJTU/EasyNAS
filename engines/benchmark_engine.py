import math
from easydict import EasyDict as edict
from typing import Union, List

from .base import BaseEngine
from builder import create_module, create_hook, create_optimizer, create_scheduler
from src.hook import hooks_run, hooks_iter

class BenchmarkEngine(BaseEngine):
    def __init__(self, obj, optimizer, lr_scheduler=None, hooks=tuple()):
        # build obj
        print("Building the object of the optimization")
        if isinstance(obj, dict):
            obj = create_module(obj, search_path='src.benchmark.object')
        elif isinstance(obj, str):
            obj = create_module({'submodule_name': 'Benchmark_function', 'args': obj}, search_path='src.benchmark.object')
        if callable(obj):
            obj = (obj, )
        assert isinstance(obj, (list, tuple))
        self.objs = obj

        self.optimizer_cfg, self.lr_scheduler_cfg, self.hooks_cfg = optimizer, lr_scheduler, hooks

        self.info = edict({
            'results': edict({}), 
            })

    def build_from_cfg(self, obj, optimizer_cfg, lr_scheduler_cfg, hooks_cfg):
        # build optimizer
        if optimizer_cfg:
            print("Building optimizer")
            optimizer = create_optimizer(optimizer_cfg, obj)
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
        return optimizer, lr_scheduler

    def optimize_one_object(self, obj_fn, max_iter):
        self.optimizer, self.lr_scheduler = self.build_from_cfg(obj_fn, self.optimizer_cfg, self.lr_scheduler_cfg, self.hooks_cfg)
        with hooks_run(self._hooks, self):
            for step in range(max_iter):
                self.info.current_iter = step
                with hooks_iter(self._hooks, self):
                    self.info.results.obj = obj_fn()
                    self.info.results.ignore_obj_list[-1].append(self.info.results.obj.item())
                    params_require_grad = []
                    for pg in self.optimizer.param_groups:
                        params_require_grad.extend(pg['params'])
                    if getattr(self.optimizer, 'ZO', False):
                        self.closure = obj_fn
                    else:
                        self.info.results.obj.backward(inputs=params_require_grad)
                    # get best. it should be put to a hook in the future
                    if self.info.results.get('best', math.inf) > self.info.results.obj:
                        self.info.results.best = float(self.info.results.obj)
#        print(self.info.results.ignore_obj_list[-1])

    def run(self, max_iter):
        self.info.results.ignore_obj_list = []
        self.info.results.ignore_best = []
        for obj in self.objs:
            print("Information of objection:")
            obj.display_info()
            self.info.results.ignore_obj_list.append([])
            self.optimize_one_object(obj, max_iter)
            self.info.results.ignore_best.append(self.info.results.best)
            self.info.results.best = math.inf
        print(self.info.results.ignore_best)
#            for step in range(max_iter):
#                self.info.current_iter = step
#                with hooks_iter(self._hooks, self):
#                    self.info.results.obj = self.obj()
#                    self.info.results.ignore_obj_list.append(self.info.results.obj.item())
#                    params_require_grad = []
#                    for pg in self.optimizer.param_groups:
#                        params_require_grad.extend(pg['params'])
#                    if getattr(self.optimizer, 'ZO', False):
#                        self.closure = self.obj
#                    else:
#                        self.info.results.obj.backward(inputs=params_require_grad)
#                    # get best. it should be put to a hook in the future
#                    if self.info.results.get('best', math.inf) > self.info.results.obj:
#                        self.info.results.best = self.info.results.obj

    def extract_performance(self):
        return self.info.results.get('best')

