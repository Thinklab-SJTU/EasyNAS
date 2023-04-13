from ..hook import HOOK, execute_period, OptHOOK

def DARTSHOOK(HOOK):
    def __init__(self, optimizer, dataloader, criterion, update_freq=1, accumulate_gradient=1):
        self.optimizer_hook = OptHOOK(optimizer, accumulate_gradient)
        self.dataloader = dataloader
        self.dataiter = iter(self.dataloader)
        self.criterion = criterion
        self.update_freq = update_freq  

#    def _initialize_arch_param(arch_params):
#        for p in arch_params:
#            torch.nn.init.normal_(p, mean=0.0, std=1e-6)
#
#    def before_run(self, runner):
#        arch_parameters = runner.model.get_arch_param()
#        self._initialize_arch_param(arch_param)

    def step(self, runner):
        arch_param = runner.model.get_arch_param()
        input_valid, target_valid = self.dataiter.next()
        logits = runner.model(input_valid)
        loss = self.criterion(logits, target_valid)
        loss.backward()

        grads =  torch.autograd.grad(loss, arch_param, grad_outputs=torch.ones_like(loss), allow_unused=True)
        for v, g in zip(arch_param, grads):
          if torch.isnan(g).any() or torch.isinf(g).any():
            raise(ValueError("gradient of architecture has NaN..."))
          if v.grad is None:
            if not (g is None):
              v.grad = Variable(g.data)
          else:
            if not (g is None):
              v.grad.data.add_(g.data)

    def before_train_epoch(self, runner)
        self.dataiter = iter(self.dataloader)

    @execute_period(self.update_freq)
    def before_train_iter(self, runner):
        self.optimizer_hook.before_train_iter(self)
        self.step(runner)
        self.optimizer_hook.after_train_iter(self)

    def after_train_epoch(self, runner):
        arch_param = runner.model.get_arch_param()
        self.runner.model.discretize()

