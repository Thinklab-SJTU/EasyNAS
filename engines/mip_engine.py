from easydict import EasyDict as edict
from typing import Union, List

from .base import BaseEngine
from builder import create_module, create_hook
from src.hook import hooks_run, hooks_iter

class MIPEngine(BaseEngine):
    def __init__(self, instance, solver, hooks=tuple()):
#        self.instance_loader, self.solver = self.build_from_cfg(instance, solver, hooks)
        self.build_from_cfg(instance, solver, hooks)
        self.info = edict({
            'results': edict({}), 
            })

    def build_all(self, instance=None, solver=None, hooks=None):
        # build data
        if instance is not None:
            print("Building instance loader")
            self.instance_loader = create_module(instance, search_path='src.mip')

        # build model
        if solver is not None:
            print("Building mip solver")
            self.solver = create_module(solver, search_path='src.mip')

        # build other hooks
        if hooks is not None:
            print("Building hooks")
            self._hooks = []
            gen = hooks_cfg.values() if isinstance(hooks_cfg, dict) else iter(hooks_cfg)
            for v in gen:
                print(v)
                self.register_hook(create_hook(v, search_path='src.mip'))
        return instance_loader, solver

    def run(self):
        with hooks_run(self._hooks, self):
            for i, instance in enumerate(self.instance_loader.load_datasets()):
                self.info.current_iter = i
                with hooks_iter(self._hooks, self):
                    self.info.current_model = self.solver.solve(instance)

    def update(self, sample):
        self.build_all(**sample)
        self.info = edict({
            'results': edict({}), 
            })

    def extract_performance(self):
        return self.info.results.get('best')
        
