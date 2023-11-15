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

    @property
    def weights(self):
        return self._weights

    @abstractmethod
    def sample(self, num, replace):
        pass

    def discretize(self):
        raise(NotImplementedError(f"No implementation for discretize method for class {self.__class__.__name__}"))

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
        self.register_weight('weight', np.ones(space_size) / space_size)
#        self.register_weight('weight', self.rdm.randn(space_size) * 1e-3)
        self.norm_fn = NORM_FN[norm_fn]
    def sample(self, space, num, replace):
        normed_weight = self.norm_fn(self.weight)
        return self.rdm.choice(space, size=num, p=normed_weight, replace=replace)
    def discretize(self, space, num):
        topk_idx = np.argpartition(-self.weight, num, axis=-1)
        return [space[tmp] for tmp in topk_idx]

# Parameterless, Continuous
class UniformContinousSampler(_NumpySampler):
    def sample(self, start, end, num):
        return self.rdm.uniform(start, end, size=num)
class NormalSampler(_NumpySampler):
    def sample(self, mean, std, num):
        return self.rdm.normal(loc=mean, scale=std, size=num)

#NORM_FN = {
#        'normalize': lambda x, dim=-1: x / x.sum(axis=-1, keepdims=True),
#        'standarize': lambda x, dim=-1: (x-x.mean(axis=dim, keepdims=True))/x.std(axis=dim, deepdims=True),
#        }
#def register_norm_fn():
#    def wrapper(func):
#        NORM_FN[func.__name__] = func
#        return func
#    return wrapper
#
#@register_norm_fn
def softmax(x, dim=-1):
    exp_x = np.exp(x)
    return exp_x/exp_x.sum(axis=dim, keepdims=True)

NORM_FN = {
        'softmax': softmax,
        }

