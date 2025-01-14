import os
from typing import Union
from functools import partial
import numpy as np
import torch

from .hook import HOOK, execute_period
from ..models.layers.common import DropPath

class ModuleApplyHOOK(HOOK):
    def __init__(self, apply_func, mode='epoch', apply_period=1, priority=0):
        self.apply_func = apply_func
        self.priority = priority
        self.apply_period = apply_period
        if mode == 'epoch': setattr(self, 'before_train_epoch', execute_period('apply_period')(self.apply))
        elif mode == 'batch': setattr(self, 'before_train_iter', execute_period('apply_period')(self.apply))

    def apply(self, runner):
        runner.model.apply(self.apply_func)

class DropPathProbHOOK(ModuleApplyHOOK):
    def __init__(self, mode='epoch', apply_period=1, priority=0):
        super(DropPathProbHOOK, self).__init__(self.set_drop_prob, mode, apply_period, priority)

    def set_drop_prob(self, m, eta):
        if isinstance(m, DropPath):
            m.drop_prob = self.init_drops[m] * eta
#            print(f"Set drop_path as {m.drop_prob}")

    def before_run(self, runner):
        self.init_drops = {m: m.drop_prob for m in runner.model.modules() if isinstance(m, DropPath)}

    def apply(self, runner):
        eta = runner.info.current_epoch / runner.info.epochs
        print(f"Multiply drop_path by {eta}")
        runner.model.apply(partial(self.set_drop_prob, eta=eta))


