from ..hook import HOOK, execute_period
from .build import create_optimizer

def OptHOOK(HOOK):
    def __init__(self, optimizer, accmulate_gradient=1):
        self.optimizer = optimizer
        self.accumulate_gradient = accumulate_gradient

    @execute_period(self.accumulate_gradient)
    def before_train_iter(self, runner):
        self.optimizer.zero_grad()

    @execute_period(self.accumulate_gradient)
    def after_train_iter(self, runner):
        self.optimizer.step()
