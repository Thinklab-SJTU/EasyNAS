import os
from typing import Union
import torch

from ..hook import HOOK, execute_period, only_master
from app.distribute_utils import synchronize_between_processes

class DDPHOOK(HOOK):
    def __init__(self, priority=0, only_master=False):
        self.priority = priority
        self.only_master = only_master

    def before_epoch(self, runner):
        if runner.is_ddp():
            for k, loader in runner.dataloaders.items():
                if loader.cfg.get('use_dist', True):
                    loader.sampler.set_epoch(runner.info.current_epoch)
#            if runner.train_loader.cfg.get('use_dist', True): runner.train_loader.sampler.set_epoch(runner.info.current_epoch)
#            if runner.val_loader.cfg.get('use_dist', True): runner.val_loader.sampler.set_epoch(runner.info.current_epoch)

    def after_train_epoch(self, runner):
        #TODO: if EvalHOOK set only_master as True, the master process will be hung since synchronize_between_processes will barrier but no other process will conduct the barrier
        if runner.is_ddp():
            for k, v in runner.info.results.train.items():
                runner.info.results.train[k] = synchronize_between_processes(v, device='cuda', mode='avg')

    def after_val_epoch(self, runner):
        if runner.is_ddp() and runner.val_loader.cfg.get('use_dist', True):
            for k, v in runner.info.results.val.items():
                runner.info.results.val[k] = synchronize_between_processes(v, device=runner.device, mode='avg')


