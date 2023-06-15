from ..hook import HOOK, execute_period, only_master
from app.distribute_utils import is_parallel

class EMA():
    def __init__(self, decay=0.9999):
        self.updates = 0  
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # decay exponential ramp (to help early epochs)
        self.shadow = {}
        self.backup = {}
 
    def update(self, model):
        msd = model.module.state_dict() if is_parallel(model) else model.state_dict()  # model state_dict
        if self.updates == 0:
            for name, param in msd.items():
                self.shadow[name] = param.clone()
        else:
            d = self.decay(self.updates)
            for name, param in msd.items():
#                if param.requires_grad:
                self.shadow[name].mul_(d).add_((1.0 - d) * param.data.detach())
        self.updates += 1
 
    def apply_shadow(self, model):
        msd = model.module.state_dict() if is_parallel(model) else model.state_dict()  # model state_dict
        for name, param in msd.items():
#            if param.requires_grad:
            assert name in self.shadow
            self.backup[name] = param.data
            param.data = self.shadow[name]
 
    def restore(self, model):
        msd = model.module.state_dict() if is_parallel(model) else model.state_dict()  # model state_dict
        for name, param in msd.items():
#            if param.requires_grad:
            assert name in self.backup
            param.data = self.backup[name]
        self.backup = {}

    def load_state_dict(self, ckpt):
        for k, v in ckpt.items():
            assert k in self.shadow
            self.shadow[k] = v


class EMAHOOK(HOOK):
    def __init__(self, decay=0.999, accumulate_gradient=1, priority=0, only_master=True):
        self.decay = decay
        self.ema = EMA(self.decay)
        self.priority = priority
        self.accumulate_gradient = accumulate_gradient
        self.only_master = only_master

    def load_state_dict(self, ckpt_ema, ema_updates): 
        self.ema.load_state_dict(ckpt_ema)
        self.ema.updates = ema_updates

#    @only_master
#    def before_run(self, runner):
#        self.ema = EMA(self.decay)

    @only_master
    @execute_period('accumulate_gradient')
    def after_train_iter(self, runner):
        self.ema.update(runner.model)

    def before_val_epoch(self, runner):
        self.ema.apply_shadow(runner.model)

    def after_val_epoch(self, runner):
        self.ema.restore(runner.model)
