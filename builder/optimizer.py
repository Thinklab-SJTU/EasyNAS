import torch

from .utils import get_submodule

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

def load_optimizer(name: str, module_name: str=None, package_path: str=None):
    if module_name:
        return get_submodule(name, module_name, package_path)
    try:
        return getattr(torch.optim, name)
    except:
        raise(ValueError(f"No criterion named as {name} in torch.nn.criterion or the given module"))

def create_optimizer(model, cfg: dict):
    optimizer = load_optimizer(cfg.get('submodule_name'), cfg.get('module_name', None), cfg.get('package_path', None))
    args = cfg.get('args', {})
    if args.get('params', None):
        ingroup_param = set()
        for i in range(len(args['params'])):
            pg = args['params']
            if pg['params'] == 'all_parameters' and i!=len(args['params'])-1:
                args['params'].append(args['params'].pop(i))
            else: pg['params'] = globals()[pg['params']](model, ingroup_param) #func_map[pg['params']](model, ingroup_param)
    else:
        args['params'] = model.parameters()
    
    return optimizer(**args)
		

