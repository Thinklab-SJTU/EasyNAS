import math
import torch
import torch.nn as nn

BENCHMARK = {}

def register_benchmark(benchmark):
    BENCHMARK[benchmark.__name__] = benchmark
    return benchmark

@register_benchmark
def rosenbrock(x):
    """
    the smaller the better
    """
    d = len(x)
    out = 0
    for i in range(1, d):
        out += 100*(x[i]-x[i-1].pow(2)).pow(2) + (x[i-1]-1).pow(2)
    return out

@register_benchmark
def ackley(x, a=20, b=0.2, c=2*math.pi):
    """
    the smaller the better
    """
    d = len(x)
    sum1 = torch.pow(x, 2).sum()
    sum2 = torch.cos(c*x).sum()
    term1 = -a * torch.exp(-b * torch.sqrt(sum1/d))
    term2 = -torch.exp(sum2/d)
    out = term1 + term2 + a + torch.exp(torch.tensor(1))
    return out

class Benchmark_func(nn.Module):
    def __init__(self, function, num_var=None, init_point=None):
        super(Benchmark_func, self).__init__()
        if isinstance(function, str):
            self.function = BENCHMARK[function]
        elif callable(function):
            self.function = function
        if init_point is None:
            assert num_var is not None
            self.parameter = nn.Parameter(torch.zeros(num_var, requires_grad=True))
        else:
            self.parameter = nn.Parameter(torch.tensor(init_point, requires_grad=True))

    def forward(self):
        return self.function(self.parameter)

