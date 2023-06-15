from ..hook import HOOK, execute_period

class OptHOOK(HOOK):
    def __init__(self, optimizer, accumulate_gradient=1, amp=False, priority=0):
        self.priority = priority
        self.optimizer = optimizer
        self.accumulate_gradient = accumulate_gradient

    def initialize(self, ckpt_opt): 
        self.optimizer.load_state_dict(ckpt_opt)

    def before_run(self, runner):
        self.amp = runner.amp
        self.scaler = amp.GradScaler(enabled=True) if self.amp else None

    @execute_period('accumulate_gradient')
    def before_train_iter(self, runner):
        self.optimizer.zero_grad()

    @execute_period('accumulate_gradient')
    def after_train_iter(self, runner):
        with amp.autocast(enabled=self.amp):
            logits = runner.model(runner.info.train_bs_input)
            loss = criterion(logits, runner.info.train_bs_target)
            runner.info.train_bs_logits = logits
            runner.info.train_bs_loss = loss
        scaler = getattr(self, 'scaler', None):
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update
        else:
            loss.backward()
            self.optimizer.step()
