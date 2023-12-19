import math
from copy import deepcopy
import torch
from itertools import chain

from .hook import HOOK, execute_period, only_master
from app.distribute_utils import is_parallel
from ..models.layers.base import SearchModule

class ModelEMA():
    def __init__(self, decay=0.9999):
        self.updates = 0  
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # decay exponential ramp (to help early epochs)
    def init(self, model):
        self.ema = deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        self.updates += 1
        d = self.decay(self.updates)

        msd = model.module.state_dict() if is_parallel(model) else model.state_dict()  # model state_dict
        for k, v in self.ema.state_dict().items():
            if isinstance(v, torch.Tensor) and v.dtype.is_floating_point:
                v *= d
                v += (1. - d) * msd[k].detach()

    def load_state_dict(self, ckpt):
        self.ema.load_state_dict(ckpt['ema'])
        self.updates = ckpt['update']

    def state_dict(self):
        return {'ema': self.ema.state_dict(),
                'update': self.updates,
               }

class ModelEMAHOOK(HOOK):
    def __init__(self, decay=0.999, accumulate_gradient=1, priority=0, only_master=True):
        self.decay = decay
        self.ema = ModelEMA(self.decay)
        self.priority = priority
        self.accumulate_gradient = accumulate_gradient
        self.only_master = only_master

    def load_state_dict(self, ckpt_ema): 
        self.ema.load_state_dict(ckpt_ema)

    def state_dict(self, runner):
        return self.ema.state_dict()

    @only_master
    def before_run(self, runner):
        self.ema.init(runner.model_without_ddp)

    @only_master
    @execute_period('accumulate_gradient')
    def after_train_iter(self, runner):
        with torch.no_grad():
            self.ema.update(runner.model_without_ddp)

    def before_val_epoch(self, runner):
        with torch.no_grad():
            self.model_bk = runner.model
            runner.model = self.ema.ema
            runner.model.eval()

    def after_val_epoch(self, runner):
        with torch.no_grad():
            runner.model = self.model_bk



class EMA():
    def __init__(self, decay=0.9999):
        self.updates = 0  
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # decay exponential ramp (to help early epochs)
        self.shadow = {}
        self.backup = {}

    def get_named_parameters(self, model):
        if isinstance(model, SearchModule):
            msd = chain(model.named_parameters(), model.named_buffers(), model.named_arch_parameters())
#            msd = chain(model.named_parameters(), model.named_buffers())
        else:
            msd = chain(model.named_parameters(), model.named_buffers())
        return  msd
 
    def update(self, model):
        if self.updates == 0:
            for name, param in self.get_named_parameters(model):
                if param.dtype.is_floating_point:
                    self.shadow[name] = param.data.clone().detach()
        else:
            d = self.decay(self.updates)
            for name, param in self.get_named_parameters(model):
                if param.dtype.is_floating_point:
                    self.shadow[name].mul_(d).add_((1.0 - d) * param.detach())
        self.updates += 1
 
    def apply_shadow(self, model):
        for name, param in self.get_named_parameters(model):
            if param.dtype.is_floating_point:
                self.backup[name] = param.data #.clone().detach()
                param.data = self.shadow[name]
 
    def restore(self, model):
        for name, param in self.get_named_parameters(model):
            if param.dtype.is_floating_point:
                param.data = self.backup[name]
        self.backup = {}

    def load_state_dict(self, ckpt):
        for k in ckpt.keys():
            assert hasattr(self, k)
        for k, v in ckpt.items():
            setattr(self, k, v)

    def state_dict(self):
        return {'shadow': self.shadow,
                'update': self.updates,
               }


class EMAHOOK(HOOK):
    def __init__(self, decay=0.999, accumulate_gradient=1, priority=0, only_master=True):
        self.decay = decay
        self.ema = EMA(self.decay)
        self.priority = priority
        self.accumulate_gradient = accumulate_gradient
        self.only_master = only_master

    def load_state_dict(self, ckpt_ema): 
        try:
            self.ema.load_state_dict(ckpt_ema)
        except Exception as e:
            print("Cannot load EMA checkpoint!")
            print(e)

    def state_dict(self, runner):
        return self.ema.state_dict()

    @only_master
    @execute_period('accumulate_gradient')
    def after_train_iter(self, runner):
        with torch.no_grad():
            self.ema.update(runner.model_without_ddp)

    def before_val_epoch(self, runner):
        if self.ema.updates == 0: return 
        with torch.no_grad():
            self.ema.apply_shadow(runner.model_without_ddp)

    def after_val_epoch(self, runner):
        if self.ema.updates == 0: return 
        with torch.no_grad():
            self.ema.restore(runner.model_without_ddp)
