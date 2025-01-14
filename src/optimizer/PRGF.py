import torch
from torch.optim import Optimizer

class PRGF(Optimizer):
    ZO = True

    def __init__(self, params, lr=1e-3, weight_decay=0, num_sample_per_step=8):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: (} - should be >= 0.0".format(lr))

        defaults = dict(lr=lr, 
                        state={},
                        weight_decay=weight_decay)
        super().__init__(params, defaults)
        # Compute the size of the parameters vector
        self._params = self.param_groups[0]['params']
        self.numel_params = sum([p.numel() for p in self._params])
        device = self._params[0].device
        self.num_sample_per_step = num_sample_per_step


    def _reset_state(self, state):
        device = self._params[0].device
        state['last_grad'] = None

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

    def get_samles(self, x):
        samples = []
        for i in range(self.num_sample_per_step-1):
            d_i = torch.rand_like(x[0])
            d_i = d_i / d_i.norm()
            samples.append(d_i)
        return samples

    @torch.no_grad()
    def step(self, closure):
        assert len(self.param_groups) == 1

        device = self._params[0].device
        current_obj = float(closure())
        group = self.param_groups[0]
        sample_norm = 5e-4
        lr = group['lr']
        state = group['state']
        last_grad = state.get('last_grad', None)
        # print(lr)

        # sample to estimate the gradient
        x_init = self._clone_param()
        samples = self.get_samles(x_init)
        # samples 与 last_grad concat
        if last_grad is not None:
            samples.append(last_grad)
        # 对samplse做正交化
        samples = torch.stack(samples)
        samples, _ = torch.qr(samples.T)
        samples = samples.T
        # print(samples @ samples.T)
        # print(samples.size())
        #取出每个正交向量
        loss = []
        for i in range(samples.shape[0]):
            d_i = samples[i]
            d_i = d_i / d_i.norm()
            loss_i = self._directional_evaluate(closure, x_init, sample_norm, d_i)
            loss.append(loss_i)
        loss = torch.tensor(loss).to(device) - current_obj
        # print(loss.size())
        grad_estimate = samples.T @ loss / sample_norm
        self._add_grad(lr, grad_estimate.neg())
        state['last_grad'] = grad_estimate
        return current_obj  
