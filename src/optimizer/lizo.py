import torch
from torch.optim import Optimizer

class LIZO(Optimizer):
    ZO = True
    """
    .. warning::
        This optimizer doesn't support per-parameter options and parameter
        groups (there can be only one).
    """
    def __init__(self, params, lr=1e-3, weight_decay=0, num_sample_per_step=8, reuse_distance_bound=0.01, orthogonal_sample=True, fast_alg=True)
        defaults = dict(lr=lr,
                        weight_decay=weight_decay
                        )
        super(LIZO, self).__init__(params, defaults)
        self._params = self.param_groups[0]['params']
        self.numel_params = sum([p.numel() for p in self._params])

        self.num_sample_per_step = min(num_sample_per_step, self.numel_params)
        if reuse_distance_bound is None:
            self.reuse_distance_bound = lr
        else: self.reuse_distance_bound = reuse_distance_bound
        self.orthogonal_sample = orthogonal_sample
        self.fast_alg = fast_alg

        self.last_delta_samples = None
        self.last_obj = None
        self.dist_matrix = torch.zeros(self.num_sample_per_step, self.num_sample_per_step)

    def _flat_param(self, params):
        views = []
        for p in params:
            views.append(p.grad.data.view(-1))
        return torch.cat(views, 0)

    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.add_(update[offset:offset + numel].view_as(p), alpha=step_size)
            offset += numel
        assert offset == self._numel()

    def _clone_param(self):
        return [p.clone(memory_format=torch.contiguous_format) for p in self._params]

    def _set_param(self, params_data):
        for p, pdata in zip(self._params, params_data):
            p.copy_(pdata)

    def _directional_evaluate(self, closure, x, t, d):
        self._add_grad(t, d)
        loss = float(closure())
        self._set_param(x)
        return loss

    #TODO: Euclidean distance
    def get_distance(self, delta_samples, distance_mode='euclidean'):
        return current @ delta_samples.t()

    #TODO: sample points
    def get_samples(self, last_delta_samples, num_to_samples, orthogonal=True):
        if orthogonal:
            pass
        else:
            # directly sample
            pass
        return new_delta_samples

    @torch.no_grad()
    def step(self, closure):
        assert len(self.param_groups) == 1

        loss = None
        group = self.param_groups[0]
        state = group['state']
        lr = group['lr']

        flat_params = self._flat_param(self._params)
        # get reused samples from last samples
        if self.last_delta_samples is not None:
            distances = self.get_distance(self.last_delta_samples-self.last_grad.view(1,-1))
            sample_idx = torch.where(distances < self.reuse_distance_bound)[0]
            # if all samples can be reused then remove the farthest, since last_sample should be used
            if len(sample_idx) == self.num_sample_per_step:
                sample_idx = sample_idx[torch.where(distances!=distances.max())[0]]
        else: sample_idx = []

        # random sample orthogonal points
        num_random = self.num_sample_per_step -1 - len(sample_idx)
        if self.last_delta_samples is None or num_random > 0:
            new_delta_samples = self.get_samples(self.last_delta_samples, num_random, orthogonal=self.orthogonal_sample)
            # get object of the new sampled points
            x_init = self._clone_param()
            for idx in range(len(sample_idx)+1, len(self.last_obj)):
                self.last_obj[idx] = self._directional_evaluate(closure, x_init, lr, new_delta_samples[idx-1-len(sample_idx)])

        # compute dist_matrix: \delta_w * \delta_w^\top
        # the first row of last_delta_samples should be last_grad, so we should deal with the first row and first column of dist_matrix
        self.dist_matrix[0,0] = self.last_lr * self.last_lr
        if len(sample_idx) > 0:
            self.last_delta_samples[1:len(sample_idx)+1] = self.last_delta_samples[sample_idx]
            self.last_delta_samples[0] = -self.last_grad
            self.last_obj[1:len(sample_idx)+1] = self.last_obj[sample_idx]

            tmp = self.last_delta_samples[1:1+len(sample_idx)] @ self.last_grad
            self.dist_matrix[0, 1:1+len(sample_idx)] = -tmp + self.last_lr * self.last_lr
            self.dist_matrix[1:1+len(sample_idx), 0] = self.dist_matrix[0, 1:1+len(sample_idx)].t()
        self.last_delta_samples[-num_random:] = new_delta_samples

        if len(sample_idx) > 0 and self.fast_alg:
            #TODO: fast algorithm to compute dist_matrix
            self.dist_matrix[1:1+len(sample_idx),1:1+len(sample_idx)] = self.dist_matrix[sample_idx][:,sample_idx] - tmp.view(-1, 1) - tmp.t().view(1, -1) + self.last_lr*self.last_lr
            if num_random > 0:
                if self.orthogonal_sample:
                    self.dist_matrix[-num_random:] = 0
                    self.dist_matrix[:, -num_random:] = 0
                    #TODO: only need the diagonal items
                    vector = (new_delta_samples * new_delta_samples).sum(dim=-1)
                    self.dist_matrix.diagonal()[-num_random:] = vector
    #                self.dist_matrix[-num_random:, -num_random:] = new_delta_samples @ new_delta_samples.t()
                else:
                    self.dist_matrix[-num_random:] = new_delta_samples[-num_random:] @ new_delta_samples.t()
                    self.dist_matrix[:, -num_random:] = self.dist_matrix[-num_random:].t()

        else:
            self.dist_matrix = self.last_delta_samples @ self.last_delta_samples.t()

        self.last_grad = self.dist_matrix.inverse() @ self.last_obj.t()
        self.last_grad = self.last_delta_samples.t() @ self.last_grad
        self.last_lr = lr

        self._add_grad(lr, self.last_grad.neg())

        return loss

    


