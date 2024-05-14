import torch
from torch.optim import Optimizer


class ZO_Adam(Optimizer):
    ZO = True

    def __init__(self, params, lr=1e-2, weight_decay=0, num_sample_per_step=8, beta1=0.9, beta2=0.3):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: (} - should be >= 0.0".format(lr))

        defaults = dict(lr=lr, 
                        weight_decay=weight_decay)
        super().__init__(params, defaults)
        # Compute the size of the parameters vector
        self._params = self.param_groups[0]['params']
        self.numel_params = sum([p.numel() for p in self._params])
        self.beta1 = beta1
        self.beta2 = beta2
        device = self._params[0].device
        self.num_sample_per_step = num_sample_per_step
        eps = 1e-8
    
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
#            d_i = torch.randn_like(x[0])
            d_i = torch.randn(self.numel_params, device=x[0].device)
            d_i = d_i / d_i.norm()
            # print(d_i.norm())
            loss_i = self._directional_evaluate(closure, x, t, d_i)
            sum += (loss_i - loss) * d_i
        # print(x[0].numel())
        return self.numel_params * sum / (self.num_sample_per_step * t)
            
    @torch.no_grad()
    def step(self, closure):
        assert len(self.param_groups) == 1

        device = self._params[0].device
        current_obj = float(closure())
        group = self.param_groups[0]
        sample_norm = 1e-5
        lr = group['lr']
        eps = 1e-8
        # print(lr)

        if not hasattr(self, 'v'):
            self.v = eps * torch.ones(self.numel_params, device=device)
            self.m = torch.zeros(self.numel_params, device=device)
            self.v_hat = self.v.clone()

        # sample to estimate the gradient
        x_init = self._clone_param()
        grad_estimate = self.grad_estimate(closure, x_init, sample_norm)
        # print(grad_estimate)
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad_estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad_estimate.pow(2)
        # # print("v: ", self.v)
        # # print("v_hat: ", self.v_hat)

        self.v_hat = torch.max(self.v_hat, self.v)
        # # print("m: ", self.m)
        # # print("v_hat: ", self.v_hat)
        grad = self.m / (self.v.sqrt() + eps)
        # print("grad: ", grad)
        # grad_norm = grad_estimate.norm()
        # print(grad)
        # grad_estimate.div_(grad_norm)
        # lr *= grad_norm
        # print(lr)
        self._add_grad(lr, grad.neg())

        return current_obj  
