from ..hook import HOOK, execute_period

class LrScheduleHOOK(HOOK):
    def __init__(self, mode='epoch', priority=0):
        self.priority = priority
        if mode == 'epoch': 
            setattr(self, 'after_train_epoch', self.update_lr)
#            setattr(self, 'after_epoch', self.update_lr)
        elif mode == 'iter': 
            setattr(self, 'after_train_iter', self.update_lr)
#            setattr(self, 'after_iter', self.update_lr)
        else:
            raise(ValueError(f"No implementation for mode as {mode}"))
    def before_run(self, runner):
        self.lr_scheduler = runner.lr_scheduler

    def state_dict(self, runner):
        return self.lr_scheduler.state_dict()

    def load_state_dict(self, state_dict):
        self.lr_scheduler.load_state_dict(state_dict)

    def initialize(self, last_epoch):
        self.lr_scheduler.last_epoch = last_epoch

    def update_lr(self, runner):
        self.lr_scheduler.step()

