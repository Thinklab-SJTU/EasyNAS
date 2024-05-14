import torch
from torch.optim import Optimizer


class ZO_SGD(Optimizer):
    ZO = True

    def __init__(self, params, lr=1e-3, mu =1e-3, weight_decay=0, num_sample_per_step=8, momentum=0, sign=False):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: (} - should be >= 0.0".format(lr))

        defaults = dict(lr=lr, 
                        weight_decay=weight_decay,
                        momentum=momentum,
                        mu = mu)
        super().__init__(params, defaults)
        # Compute the size of the parameters vector
        self._params = self.param_groups[0]['params']
        self.numel_params = sum([p.numel() for p in self._params])
        self.sign = sign
        self.num_sample_per_step = num_sample_per_step
        self.mu = mu

    
    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.add_(update[offset:offset + numel].view_as(p), alpha=step_size)
            offset += numel
        assert offset == self.numel_params

    def _set_param(self, params_data):
        for p, pdata in zip(self._params, params_data):
            p.copy_(pdata)

    def _clone_param(self):
        return [p.clone(memory_format=torch.contiguous_format) for p in self._params]
    
    def _directional_evaluate(self, closure, x, t, d):
        self._add_grad(t, d)
        loss = float(closure())
        self._set_param(x)
        return loss
    
    def grad_estimate(self, closure, x, t):
        sum = 0
        loss = float(closure())
        for i in range(self.num_sample_per_step):
            d_i = torch.randn_like(x[0])
            d_i = d_i / d_i.norm()
            # print(d_i.norm())
            loss_i = self._directional_evaluate(closure, x, t, d_i)
            sum += (loss_i - loss) * d_i
        # print(x[0].numel())
        return sum / (self.num_sample_per_step * t)
            
    @torch.no_grad()
    def step(self, closure):
        assert len(self.param_groups) == 1

        device = self._params[0].device
        current_obj = float(closure())
        group = self.param_groups[0]
        sample_norm = 1e-5
        lr = group['lr']
        # print(lr)

        # sample to estimate the gradient
        x_init = self._clone_param()
        grad_estimate = self.grad_estimate(closure, x_init, sample_norm)
        grad_norm = grad_estimate.norm()
        grad_estimate.div_(grad_norm)
        lr *= grad_norm
        # print(grad_estimate.norm())
        # print(lr)
        if (self.sign):
            grad_estimate.sign_()
        # print(grad_estimate)
        # print(grad_estimate.norm())

        # update the parameters
        # print("lr: ", lr)
        self._add_grad(lr, grad_estimate.neg())
        return current_obj
