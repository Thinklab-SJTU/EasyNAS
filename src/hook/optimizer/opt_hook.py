from ..hook import HOOK, execute_period

class OptHOOK(HOOK):
    def __init__(self, optimizer, accumulate_gradient=1):
        self.optimizer = optimizer
        self.accumulate_gradient = accumulate_gradient

    def initialize(self, ckpt_opt): 
        self.optimizer.load_state_dict(ckpt_opt)

    @execute_period('accumulate_gradient')
    def before_train_iter(self, runner):
        self.optimizer.zero_grad()

    @execute_period('accumulate_gradient')
    def after_train_iter(self, runner):
        self.optimizer.step()
