from easydict import EasyDict
from typing import Union, List

from .base import BaseEngine
from builder import create_module
from src.hook import hooks_run, hooks_epoch

class MIPEngine(BaseEngine):
    def __init__(self, instance, solver, hooks=tuple()):
        self.instance_loader, self.solver = self.build_from_cfg(instance, opt, hooks)
        self.info = EasyDict({
            'results': OrderedDict({}), 
            })

    def build_from_cfg(self, instance_cfg, solver_cfg, hooks_cfg):
        # build data
        print("Building instance loader")
        instance_loader = create_module(instance_cfg, search_path='src.mip')

        # build model
        print("Building mip solver")
        solver = create_module(solver_cfg, search_path='src.mip')

        # build other hooks
        print("Building hooks")
        self._hooks = []
        gen = hooks_cfg.values() if isinstance(hooks_cfg, dict) else iter(hooks_cfg)
        for v in gen:
            print(v)
            self.register_hook(create_hook(v, search_path='src.mip'))
        return instance_loader, solver

    def run(self):
        with hooks_run(self._hooks, self):
            for i, instance in enumerate(self.instance_loader)
                with hooks_epoch(self._hooks, self):
                    sol, primal, time = self.solver.solve(instance)
                    self.info.results[instance] = {'sol': sol, 'primal': primal, 'time': time}
