import numpy as np
import math
import torch
import torch.nn as nn

BENCHMARK = {}

def register_benchmark(benchmark):
    BENCHMARK[benchmark.__name__] = benchmark
    return benchmark


@register_benchmark
def branin(x, a=1, b=5.1/(4*math.pi*math.pi), c=5/math.pi, r=6, s=10, t=1/(8*math.pi)):
    x1, x2 = x[0], x[1]
    term1 = a * (x2 - b*x1^2 + c*x1 - r).pow(2)
    term2 = s*(1-t)*torch.cos(x1) 
    out = term1 + term2 + s
    return out

@register_benchmark
def rosenbrock(x):
    """
    the smaller the better
    """
    return sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)
#    d = len(x)
#    out = 0
#    for i in range(1, d):
#        out += 100*(x[i]-x[i-1].pow(2)).pow(2) + (x[i-1]-1).pow(2)
#    return out

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

@register_benchmark
def sphere(x):
    """The sphere function"""
    res = sum(x**2)
    return res

@register_benchmark
def beale(x_0):
    """The Beale function"""
    x = x_0[0]
    y = x_0[1]
    res = (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (
        (2.625 - x + x*y**3)**2)
    return res

@register_benchmark
def booth(x_0):
    """The Booth function"""
    x = x_0[0]
    y = x_0[1]
    return ((x+2*y-7)**2 + (2*x+y-5)**2)

@register_benchmark
def bukin(x_0):
    """The Bukin function"""
    x = x_0[0]
    y = x_0[1]
    return (100 * torch.sqrt(abs(y - 0.01*x**2)) + 0.01 * (abs(x+10)))


class Benchmark_func(nn.Module):
    def __init__(self, function, num_var=None, init_point=None):
        super(Benchmark_func, self).__init__()
        if isinstance(function, str):
            self.function = BENCHMARK[function]
        elif callable(function):
            self.function = function
        if init_point is None:
            assert num_var is not None
            self.input = nn.Parameter(torch.zeros(num_var, requires_grad=True))
        else:
            self.input = nn.Parameter(torch.tensor(init_point, requires_grad=True))

    def update_weight(weight):
        if isinstance(weight, np.array):
            weight = torch.from_numpy(weight, device=self.input.device, dtype=self.input.dtype)
        self.input.data.copy_(weight)

    def forward(self):
        return self.function(self.input)

    def display_info(self):
        print(f"Function name: {self.function.__name__}")
        print(f"No. parameters: {self.input.shape}")
        print(f"Initial point: {self.input.data}")

