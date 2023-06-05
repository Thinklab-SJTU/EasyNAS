from ..hook import HOOK, execute_period

class LrScheduleHOOK(HOOK):
    def __init__(self, lr_scheduler, mode='epoch', priority=0):
        self.priority = priority
        self.lr_scheduler = lr_scheduler
        if mode == 'epoch': setattr(self, 'before_train_epoch', self.update_lr)
        elif mode == 'batch': setattr(self, 'before_train_iter', self.update_lr)
        else:
            raise(ValueError(f"No implementation for mode as {mode}"))

    def initialize(self, last_epoch):
        self.lr_scheduler.last_epoch = last_epoch

    def update_lr(self, runner):
        self.lr_scheduler.step()

