
def one_cycle(start=0.0, end=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (end - start) + start

def linear(epoch, init_lr):
    return lambda x: init_lr * (epoch - 5 - x) / (epoch - 5) if epoch - x > 5 else init_lr * (epoch - x) / ((epoch - 5)*5)

