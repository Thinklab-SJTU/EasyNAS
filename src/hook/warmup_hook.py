import os
from typing import Union
import numpy as np
import torch

from .hook import HOOK, execute_period

class WarmupHOOK(HOOK):
    def __init__(self, max_iter, warmup_init_lr, warmup_init_momentum=None, max_epoch=0, priority=0, accumulate_gradient=1):
        self.count = 0
        self.max_iter = max_iter
        self.max_epoch = max_epoch
        self.priority = priority
        self.accumulate_gradient = accumulate_gradient
        self.warmup_init_momentum = warmup_init_momentum
        self.warmup_init_lr = warmup_init_lr

    def get_lr(self, group_id, final_lr):
        xi = [0, self.max_iter]  # x interp
        warmup_init_lr = self.warmup_init_lr[group_id] if isinstance(self.warmup_init_lr, (tuple, list)) else self.warmup_init_lr
        if warmup_init_lr is None: return final_lr
        return np.interp(self.count, xi, [warmup_init_lr, final_lr])

    def get_momentum(self, group_id, final_momentum):
        xi = [0, self.max_iter]  # x interp
        warmup_init_momentum = self.warmup_init_momentum[group_id] if isinstance(self.warmup_init_momentum, (tuple, list)) else self.warmup_init_momentum
        if warmup_init_momentum is None: return final_momentum
        return np.interp(self.count, xi, [warmup_init_momentum, final_momentum])

    def update_lr_momentum(self, runner):
        for j, x in enumerate(runner.optimizer.param_groups):
            x['lr'] = self.get_lr(j, self.final_lr[j])
            if self.warmup_init_momentum:
                if 'momentum' in x:
                    x['momentum'] = self.get_momentum(j, self.final_momentum[j])

    def before_run(self, runner):
        self.final_lr = [x['lr'] for x in runner.optimizer.param_groups]
        self.final_momentum = [x.get('momentum', None) for x in runner.optimizer.param_groups]
        self.max_iter = max(self.max_iter, self.max_epoch*len(runner.train_loader))
#        self.update_lr_momentum(runner)
#        self.count += 1

    @execute_period('accumulate_gradient')
    def before_train_iter(self, runner):
        if self.count == self.max_iter:
            return
        self.update_lr_momentum(runner)
        self.count += 1

