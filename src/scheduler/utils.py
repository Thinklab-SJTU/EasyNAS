import math
import bisect

def one_cycle(x, start=0.0, end=1.0, steps=100):
    return ((1 - math.cos(x * math.pi / steps)) / 2) * (end - start) + start

def linear(x, epoch, init_lr):
    return init_lr * (epoch - 5 - x) / (epoch - 5) if epoch - x > 5 else init_lr * (epoch - x) / ((epoch - 5)*5)

def step(x, gamma, step_size):
    return gamma**(x//step_size)

def multistep(x, gamma, milestones):
    return gamma**(bisect.bisect(milestones, x))
