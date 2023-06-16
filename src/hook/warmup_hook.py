import os
from typing import Union
import numpy as np
import torch

from .hook import HOOK, execute_period

class WarmupHOOK(HOOK):
    def __init__(self, max_iter, warmup_init_lr_rate, warmup_init_momentum_rate=None, priority=0, accumulate_gradient=1):
        self.count = 0
        self.max_iter = max_iter
        self.priority = priority
        self.accumulate_gradient = accumulate_gradient
        self.warmup_init_momentum_rate = warmup_init_momentum_rate
        self.warmup_init_lr_rate = warmup_init_lr_rate

    def get_lr_rate(self, group_id):
        xi = [0, self.max_iter]  # x interp
        # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
        warmup_init_lr_rate = self.warmup_init_lr_rate[group_id] if isinstance(self.warmup_init_lr_rate, (tuple, list)) else self.warmup_init_lr_rate
        if warmup_init_lr_rate is None: return 1.
        return np.interp(self.count, xi, [warmup_init_lr_rate, 1.]) / (np.interp(self.count-1, xi, [warmup_init_lr_rate, 1.]) if self.count > 0 else 1)

    def get_momentum_rate(self):
        xi = [0, self.max_iter]  # x interp
        return np.interp(self.count, xi, [self.warmup_init_momentum_rate, 1.]) / (np.interp(self.count-1, xi, [self.warmup_init_momentum_rate, 1.]) if self.count > 0 else 1)

    @execute_period('accumulate_gradient')
    def after_train_iter(self, runner):
        if self.count == self.max_iter:
            return
        for j, x in enumerate(runner.optimizer_hook.optimizer.param_groups):
            # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
            lr_rate = self.get_lr_rate(j)
            x['lr'] *= lr_rate
#            x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
            if self.warmup_init_momentum_rate:
                momentum_rate = self.get_momentum_rate()
                if 'momentum' in x:
                    x['momentum'] *= momentum_rate
        self.count += 1

