from abc import ABC,abstractmethod
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
        for n, w in self._weights:
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
#        self.register_weight('weight', torch.tensor(torch.ones(space_size) / space_size, requires_grad=True))
        self.register_weight('weight', torch.tensor(1e-3*torch.randn(space_size, requires_grad=True), requires_grad=True))
        self.norm_fn = NORM_FN[norm_fn]
    def sample(self, space, num, replace):
        normed_weight = self.norm_fn(self.weight)
        return self.rdm.choice(space, size=num, p=normed_weight.numpy(), replace=replace)
    def topk(self, space, k=None):
        return_list = False if k is None else True
        k = 1 if k is None else k
#        topk_idx = np.argpartition(-self.weight, k, axis=-1)
        _, topk_idx = self.weight.topk(k, dim=-1, largest=True, sorted=True)
        return [space[tmp] for tmp in topk_idx] if return_list else space[topk_idx[0]]
    def __repr__(self):
        with torch.no_grad():
            normed_weight = self.norm_fn(self.weight)
        string = f"{self.__class__.__name__}(seed={self.seed}, \nweights={self.weight.data}, \nnormed_weights={normed_weight})"
        return string

# Parameterless, Continuous
class UniformContinousSampler(_NumpySampler):
    def sample(self, start, end, num):
        return self.rdm.uniform(start, end, size=num)
class NormalSampler(_NumpySampler):
    def sample(self, mean, std, num):
        return self.rdm.normal(loc=mean, scale=std, size=num)

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
def gumbel_softmax(logits, temperature=1, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    while True:
      gumbel = -torch.log(-torch.log(torch.empty(shape, device=device).uniform_()))
#      gumbel = -torch.empty(shape, device=device).exponential_().log()
#      U = torch.rand(shape, device=device)
#      gumbel = -torch.log(-torch.log(U + eps) + eps)
      y = logits + gumbel 
      y = nn.functional.softmax(y / temperature, dim=-1)
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
