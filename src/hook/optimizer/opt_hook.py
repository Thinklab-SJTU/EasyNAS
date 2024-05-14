import torch
from ..hook import HOOK, execute_period

class OptHOOK(HOOK):
    def __init__(self, optimizer=None, accumulate_gradient=1, grad_clip=None, priority=0):
        self.priority = priority
        self.accumulate_gradient = accumulate_gradient
        self.optimizer = optimizer
        self.grad_clip = grad_clip

    def initialize(self, ckpt_opt): 
        self.optimizer.load_state_dict(ckpt_opt)

    def before_run(self, runner):
        if self.optimizer is None:
            self.optimizer = runner.optimizer
        self.optimizer.zero_grad()

#    @execute_period('accumulate_gradient')
#    def before_train_iter(self, runner):
#        self.optimizer.zero_grad()

    @execute_period('accumulate_gradient')
    def after_train_iter(self, runner):
        scaler = getattr(runner, 'scaler', None)
        if self.grad_clip:
            if scaler: scaler.unscale_(self.optimizer)
            params_require_grad = []
            for pg in self.optimizer.param_groups:
                params_require_grad.extend(pg['params'])
            torch.nn.utils.clip_grad_norm_(params_require_grad, self.grad_clip)
#            torch.nn.utils.clip_grad_norm_(runner.model.parameters(), self.grad_clip)
        if scaler:
            scaler.step(self.optimizer)
            scaler.update()
        else:
            self.optimizer.step()
        self.optimizer.zero_grad()

    @execute_period('accumulate_gradient')
    def after_iter(self, runner):
        self.after_train_iter(runner)


class ZOOptHOOK(OptHOOK):
    @execute_period('accumulate_gradient')
    def after_train_iter(self, runner):
        scaler = getattr(runner, 'scaler', None)
        if self.grad_clip:
            if scaler: scaler.unscale_(self.optimizer)
            params_require_grad = []
            for pg in self.optimizer.param_groups:
                params_require_grad.extend(pg['params'])
            torch.nn.utils.clip_grad_norm_(params_require_grad, self.grad_clip)
#            torch.nn.utils.clip_grad_norm_(runner.model.parameters(), self.grad_clip)
        if scaler:
            scaler.step(self.optimizer)
            scaler.update()
        else:
            self.optimizer.step(closure=runner.closure)
        self.optimizer.zero_grad()
