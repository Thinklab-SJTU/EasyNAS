import torch

from .utils import get_submodule, get_submodule_by_name

class model_criterion(object):
    def __init__(self, model, criterion):
        self.model = model
        self.criterion = criterion

    def parameters(self):
        gen = self.model.parameters()
        yield from gen
        gen = self.criterion.parameters()
        yield from gen
    def named_modules(self):
        gen = self.model.named_modules()
        yield from gen
        gen = self.criterion.named_modules()
        yield from gen

def all_parameters(model, ingroup_param=set()):
    if len(ingroup_param) == 0:
        return model.parameters()
    return list(set(list(model.parameters())) - ingroup_param)
def conv_parameters(model, criterion, ingroup_param=set()):
    params = []
    for k, v in model.named_modules():
        if isinstance(v, [nn.Conv2d, nn.Linear]):
            params.append(v.weight) 
    ingroup_param.add(set(params))
    return params
def bn_parameters(model, criterion, ingroup_param=set()):
    params = []
    for k, v in model.named_modules():
        if isinstance(v, nn.BatchNorm2d):
            params.append(v.weight) 
    ingroup_param.add(set(params))
    return params
def bias_parameters(model, criterion, ingroup_param=set()):
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
        model = model_criterion(model, criterion)
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
		
