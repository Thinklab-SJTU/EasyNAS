import torch
from itertools import chain

from .utils import get_submodule, get_submodule_by_name

class module_iters(object):
    def __init__(self, *modules):
        self.modules = modules

    def parameters(self):
        return chain(*[m.parameters() for m in self.modules])
#        for m in self.modules:
#            gen = m.parameters()
#            yield from gen
    def named_modules(self):
        return chain(*[m.named_modules() for m in self.modules])
#        for m in self.modules:
#            gen = m.named_modules()
#            yield from gen
    def named_parameters(self):
        return chain(*[m.named_parameters() for m in self.modules])

def all_parameters(model, ingroup_param=set()):
    if len(ingroup_param) == 0:
        return model.parameters()
    return list(set(list(model.parameters())) - ingroup_param)
def conv_parameters(model, ingroup_param=set()):
    params = []
    for k, v in model.named_modules():
        if isinstance(v, [nn.Conv2d, nn.Linear]):
            params.append(v.weight) 
    ingroup_param.add(set(params))
    return params
def bn_parameters(model, ingroup_param=set()):
    params = []
    for k, v in model.named_modules():
        if isinstance(v, nn.BatchNorm2d):
            params.append(v.weight) 
    ingroup_param.add(set(params))
    return params
def bias_parameters(model, ingroup_param=set()):
    params = []
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            params.append(v.bias)  
    ingroup_param.add(set(params))
    return params

func_map = {
        'all_parameters': all_parameters,
        'conv_parameters': conv_parameters,
        'bn_parameters': bn_parameters,
        'bias_parameters': bias_parameters,
        }

def create_optimizer(model, cfg: dict, criterion=None):
    if criterion:
        model = module_iters(model, criterion)

    optimizer = get_submodule_by_name(cfg.get('submodule_name'), search_path=('torch.optim',))
    args = cfg.get('args', {})
    if args.get('params', None):
        ingroup_param = set()
        for i in range(len(args['params'])):
            pg = args['params']
            if pg['params'] == 'all_parameters' and i!=len(args['params'])-1:
                args['params'].append(args['params'].pop(i))
            else: pg['params'] = globals()[pg['params']](model, ingroup_param) #func_map[pg['params']](model, criterion, ingroup_param)
    else:
        args['params'] = model.parameters()

    return optimizer(**args)
		
