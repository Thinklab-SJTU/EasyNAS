import torch
import torch.nn as nn
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
#    return [v for k, v in model.named_parameters() if k not in ingroup_param]
    return list(set(list(model.parameters())) - ingroup_param)
def conv_parameters(model, ingroup_param=set()):
    keys, params = [], []
    for k, v in model.named_modules():
        if isinstance(v, (nn.Conv2d, nn.Linear)):
            params.append(v.weight) 
            keys.append(k)
    ingroup_param |= set(params)
    return params
def bn_parameters(model, ingroup_param=set()):
    keys, params = [], []
    for k, v in model.named_modules():
        if isinstance(v, nn.BatchNorm2d):
            params.append(v.weight) 
            keys.append(k)
    ingroup_param |= set(params)
    return params
def bias_parameters(model, ingroup_param=set()):
    keys, params = [], []
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            params.append(v.bias)  
            keys.append(k)
    ingroup_param |= set(params)
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
        i = 0
        while i < len(args['params']):
            pg = args['params'][i]
            if pg['params'] == 'all_parameters' and i!=len(args['params'])-1:
                if args['params'][-1]['params'] != 'all_parameters':
                    args['params'].append(args['params'].pop(i))
                else: args['params'].pop(i)
            else: 
                pg['params'] = globals()[pg['params']](model, ingroup_param) #func_map[pg['params']](model, criterion, ingroup_param)
                i += 1
    else:
        args['params'] = model.parameters()

    return optimizer(**args)
		
