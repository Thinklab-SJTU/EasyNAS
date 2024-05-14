import os
import inspect
from functools import partial
import torch
import json
import yaml

from builder import get_submodule_by_name, create_criterion, CfgDumper
from ..hook import HOOK, execute_period, only_master, hooks_train_iter
from .. import OptHOOK
from .supernet_hook import to_device


class SPOSHOOK(HOOK):
    def __init__(self, update_freq=1, priority=0, save_root=None,             
           replace_settings={}
           ):
        self.priority = priority
        self.update_freq = update_freq  
        self.save_root = save_root
        if self.save_root: 
            os.makedirs(self.save_root, exist_ok=True)
        self.replace_settings = replace_settings

    def before_run(self, runner):
        runner.search_space.apply_sampler_weights(lambda x: to_device(x, runner.device), recurse=True)

    @execute_period("update_freq")
    def before_train_iter(self, runner):
        # update weights of sampler in search space
        pass


