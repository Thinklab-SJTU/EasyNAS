import torch
from ..hook import HOOK, execute_period

class OptHOOK(HOOK):
    def __init__(self, optimizer, accumulate_gradient=1, grad_clip=None, priority=0):
        self.priority = priority
        self.optimizer = optimizer
        self.accumulate_gradient = accumulate_gradient
        self.grad_clip = grad_clip

    def initialize(self, ckpt_opt): 
        self.optimizer.load_state_dict(ckpt_opt)

    def before_run(self, runner):
        self.amp = runner.amp
        self.optimizer.zero_grad()

#    @execute_period('accumulate_gradient')
#    def before_train_iter(self, runner):
#        self.optimizer.zero_grad()

    @execute_period('accumulate_gradient')
    def after_train_iter(self, runner):
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(runner.model.parameters(), self.grad_clip)
        scaler = getattr(runner, 'scaler', None)
        loss = runner.info.train_bs_loss
        if scaler:
            scaler.step(self.optimizer)
            scaler.update()
        else:
            self.optimizer.step()
        self.optimizer.zero_grad()
