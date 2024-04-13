from abc import ABC,abstractmethod
import torch
import numpy as np

class _Sampler(ABC):
    def __init__(self, seed=None):
        self._weights = {}
        if seed is None: 
            self.seed = np.random.randint(0,10000, 1)[0]
        else: self.seed = seed

    def register_weight(self, name, weight):
        assert name not in self._weights and not hasattr(self, name), f"{self.__class__.__name__} has an attribution named as {name}"
        self._weights[name] = weight
        setattr(self, name, weight)

    def named_weights(self):
        for n, w in self._weights.items():
            yield n, w

    def weights(self):
        return self._weights

    @abstractmethod
    def sample(self, num, replace):
        pass

    def topk(self, space, k=None):
        raise(NotImplementedError(f"No implementation for discretize method for class {self.__class__.__name__}"))

    def __repr__(self):
        string = f"{self.__class__.__name__}(seed={self.seed})"
        return string

class _NumpySampler(_Sampler):
    def __init__(self, seed=None):
        super(_NumpySampler, self).__init__(seed)
        self.rdm = np.random.RandomState(self.seed)

#############################################

# Parameterless, Discrete
class UniformDiscreteSampler(_NumpySampler):
    def sample(self, space, num, replace):
        return self.rdm.choice(space, size=num, replace=replace)
class BinormialSampler(_NumpySampler):
    def sample(self, n, p, num):
        return self.rdm.binomial(n, p, size=num)
class PoissonSampler(_NumpySampler):
    def sample(self, lam, num):
        return self.rdm.poisson(lam, size=num)

# Parametric, Discrete
class WeightedSampler(_NumpySampler):
    def __init__(self, space_size, norm_fn='softmax', seed=None):
        super(WeightedSampler, self).__init__(seed)
        self.register_weight('weight', torch.zeros(space_size, requires_grad=True))
#        self.register_weight('weight', (1e-3*torch.randn(space_size, requires_grad=True)).clone().detach().requires_grad_())
        self.norm_fn = NORM_FN[norm_fn]
    def init_weight(self):
        nn.init.xavier_normal_(self.weight)
    def sample(self, space, num, replace):
        normed_weight = self.norm_fn(self.weight)
        return self.rdm.choice(space, size=num, p=normed_weight.detach().cpu().numpy(), replace=replace)
    def topk(self, space, k=None):
        return_list = False if k is None else True
        k = 1 if k is None else k
#        topk_idx = np.argpartition(-self.weight, k, axis=-1)
        _, topk_idx = self.weight.topk(k, dim=-1, largest=True, sorted=True)
        return [space[tmp] for tmp in topk_idx] if return_list else space[topk_idx[0]]
#    def __repr__(self):
#        with torch.no_grad():
#            normed_weight = self.norm_fn(self.weight)
#        string = f"{self.__class__.__name__}(seed={self.seed}, \nweights={self.weight.data}, \nnormed_weights={normed_weight})"
#        return string

class UniformDiscreteWeightedSampler(UniformDiscreteSampler):
    def __init__(self, space_size, norm_fn='softmax', seed=None):
        super(UniformDiscreteWeightedSampler, self).__init__(seed)
        self.register_weight('weight', torch.ones(space_size) / space_size)
        self.norm_fn = lambda x: x
    def sample(self, space, num, replace):
        return self.rdm.choice(space, size=num, replace=replace)


#####################
# norm_fn
#####################
NORM_FN = {
        'normalize': lambda x, dim=-1: x / x.sum(dim=-1, keepdim=True),
        'standarize': lambda x, dim=-1: (x-x.mean(dim=dim, keepdim=True))/x.std(dim=dim, deepdim=True),
#        'normalize': lambda x, dim=-1: x / x.sum(axis=-1, keepdims=True),
#        'standarize': lambda x, dim=-1: (x-x.mean(axis=dim, keepdims=True))/x.std(axis=dim, deepdims=True),
        }

def register_norm_fn(norm_fn):
    NORM_FN[norm_fn.__name__] = norm_fn
    return norm_fn

@register_norm_fn
def softmax(x, dim=-1, temperature=1):
    return torch.softmax(x / temperature, dim=dim)
#    exp_x = np.exp(x)
#    return exp_x/exp_x.sum(axis=dim, keepdims=True)

@register_norm_fn
def pseudo_gumbel_softmax(logits, temperature=1, hard=True):
    y = torch.nn.functional.softmax(logits / temperature, dim=-1)
    if not hard:
        return y
    idx = torch.tensor(np.random.choice(list(range(y.shape[-1])), size=1, p=y.cpu().detach().numpy()), device=y.device)

    y_hard = torch.zeros_like(y)
    y_hard.scatter_(-1, idx, 1.)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = y_hard - y.detach() + y
    return y_hard

@register_norm_fn
def gumbel_softmax(logits, temperature=1, hard=True):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    while True:
      gumbel = -torch.log(-torch.log(torch.empty(logits.shape, device=logits.device).uniform_()))
#      gumbel = -torch.empty(shape, device=device).exponential_().log()
#      U = torch.rand(shape, device=device)
#      gumbel = -torch.log(-torch.log(U + eps) + eps)
      y = logits + gumbel 
      y = torch.nn.functional.softmax(y / temperature, dim=-1)
      if torch.isinf(y).any() or torch.isnan(y).any(): continue
      else: break

    if not hard:
        return y

    shape = y.size()
    _, ind = y.max(dim=-1, keepdim=True)
    y_hard = torch.zeros_like(y)
    y_hard.scatter_(-1, ind, 1.)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = y_hard - y.detach() + y
    return y_hard

#############################################

# Parameterless, Continuous
class UniformContinousSampler(_NumpySampler):
    def __init__(self, start=0, end=1):
        self.start, self.end = start, end
    def set_param(self, space):
        self.start, self.end = [float(tmp) for tmp in space.split(':')]
    def sample(self, num):
        return self.rdm.uniform(self.start, self.end, size=num)
class NormalSampler(_NumpySampler):
    def __init__(self, mean=0, std=1):
        self.mean, self.std = mean, std
    def set_param(self, space):
        self.mean, self.std = [float(tmp) for tmp in space.split(':')]
    def sample(self, num):
        return self.rdm.normal(loc=self.mean, scale=self.std, size=num)

# Parametric, Continuous
class ContinuousWeightedSampler(_Sampler):
    def __init__(self, weight_shape, init_fn='default', mapping='identity', seed=None):
        super(ContinuousWeightedSampler, self).__init__(seed)
        self.register_weight('weight', torch.zeros(weight_shape, requires_grad=True))
        if init_fn == 'default':
            init_fn = nn.init.xavier_normal_
        assert callable(init_fn)
        self.init_fn = init_fn
        self.init_weight()

        if isinstance(mapping, str):
            self.mapping = MAPPING_FN[mapping]
        else: self.mapping = mapping
        assert callable(self.mapping)

    def init_weight(self):
        self.init_fn(self.weight)

    def sample(self, num, replace):
        assert num == 1
        normed_weight = self.mapping(self.weight)
        return normed_weight 


#####################
# mapping_fn
#####################
MAPPING_FN = {
        'identity': lambda x: x, 
        }

def register_mapping_fn(mapping_fn):
    MAPPING_FN[mapping_fn.__name__] = mapping_fn
    return mapping_fn

@register_mapping_fn
def sigmoid(x):
    return torch.sigmoid(x)

@register_mapping_fn
def clamp(x, min=None, max=None):
    return torch.clamp(x, min, max)
