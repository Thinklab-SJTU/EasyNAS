import sys
import os
sys.path.append(os.getcwd())
import inspect
from functools import partial
from easydict import EasyDict as edict

import yaml
import importlib


def _get_submodule(submodule_name: str, module_name: str='dataset.datasets', package_path: str=None):
#    print(importlib.util.find_spec("dataset.datasets"))
#    module = importlib.import_module('dataset.datasets')
#    print(getattr(module, 'CIFAR10'))

#    package = package_path.split('.')[0].replace('/', '.') if isinstance(package_path, str) else None
    package = package_path.replace('/', '.') if isinstance(package_path, str) else None
    if importlib.util.find_spec(module_name, package=package):
        module = importlib.import_module(module_name, package=package_path)
        try:
            submodule= getattr(module, submodule_name)
            return submodule
        except AttributeError:
            raise(ImportError(e))
        except Exception as e:
            raise(e)
    else:
        raise(ImportError(f"[{module_name}] is not found in the package [{package_path}]"))

#def create_submodule(submodule_name, module_name, package_path, **args):
#    submodule = _get_submodule(submodule_name, module_name, package_path)
#    return submodule(**args)

def get_submodule(submodule_name, module_name, package_path=None, loaded_submodule={}):
    submodule = loaded_submodule.get(submodule_name, None)
    if submodule: 
        return submodule
    submodule = _get_submodule(submodule_name, module_name, package_path)
    loaded_submodule[submodule_name] = submodule
    return submodule

def create_submodule_by_dict(cfg: dict, search_path=None):
    submodule_name = cfg.get('submodule_name')
    return get_submodule_by_name(submodule_name, search_path)(**cfg.get('args', {}))

def get_submodule_by_name(name, search_path=None, loaded_submodule=None):
    if loaded_submodule and loaded_submodule.get(name, None): 
        return loaded_submodule.get(name)

    module_name = str(name).split('.')
    try: 
        submodule = _get_submodule(module_name[-1], '.'.join(module_name[:-1]))
    except ImportError as e: 
        if search_path:
            search_path = (search_path,) if isinstance(search_path, str) else search_path
            for p in search_path:
                if not name.startswith(p):
                    try:
                        submodule = get_submodule_by_name('.'.join([p, name]))
                    except: pass
                    else:
                        if loaded_submodule:
                            loaded_submodule[name] = submodule
                        return submodule

            raise(e)
        else: 
            raise(e)
    except Exception as e:
        raise(e)
    else:
        if loaded_submodule:
            loaded_submodule[name] = submodule
        return submodule

#    if default_path is not None and not name.startswith(default_path):
#        module_name = '.'.join(default_path, module_name)
#    return _get_submodule(module_name[-1], '.'.join(module_name[:-1]))

def parse_cfg(yaml_file):
    with open(yaml_file, 'r') as f:
        tmp_cfg = yaml.load(f.read(), CfgLoader)

    if isinstance(tmp_cfg, dict):
        cfg = {}
        for k, v in tmp_cfg.items():
            cfg[k] = parse_cfg(v) if isinstance(v, str) and os.path.isfile(v) else v
    elif isinstance(tmp_cfg, list):
        cfg = []
        for v in tmp_cfg:
            cfg.append(parse_cfg(v) if isinstance(v, str) and os.path.isfile(v) else v)
    else:
        cfg = deepcopy(tmp_cfg)

    return cfg


class CfgLoader(yaml.SafeLoader):
    def construct_python_tuple(self, node):
        return tuple(self.construct_sequence(node))

    # !join [str1, str2, ...]
    def join(self, node):
        return ''.join([str(i) for i in self.construct_sequence(node)])

    # !get_module [module_name]
    # !get_module [module_name, dict(kwargs)]
    def get_module(self, node):
#        module_name = str(self.construct_scalar(node.value[0])).split('.')
#        args = self.construct_mapping(node.value[1])
        name_args = self.construct_sequence(node, deep=True)
        module = get_submodule_by_name(name_args[0])
        if len(name_args) > 1:
            return partial(module, **name_args[1])
        else:
            return partial(module)

    def construct_python_edict(self, node):
        return edict(self.construct_mapping(node))

    def _update_dict(self, data, update):
        if update.pop('recurse', True):
            return data.update(update)
        for k, v in update.items():
            if k in data and isinstance(v, dict):
                self._update_dict(data[k], v)
            else:
                data[k] = v

    # !include [crossFile]
    # !include [crossFile, dict(kwargs)]
    # !include [crossFile:key1:key2:...]
    # !include [crossFile:key1:key2:..., dict(kwargs)]
    def construct_crossRef(self, node):
        crossRef_replaceArgs = self.construct_sequence(node, deep=True)
        crossRef = crossRef_replaceArgs[0].split(':')
        with open(crossRef[0], 'r') as f:
            data = yaml.load(f.read(), CfgLoader)
        if len(crossRef_replaceArgs) > 1:
            data = {k: data[k] for k in crossRef[1:]}
            self._update_dict(self, data, crossRef_replaceArgs[1])
#            data.update(crossRef_replaceArgs[1])
        else:
            if len(crossRef) == 2:
                data = data[crossRef[1]]
            elif len(crossRef) > 2:
                data = {k: data[k] for k in crossRef[1:]}
        return data

CfgLoader.add_constructor(
    '!tuple', CfgLoader.construct_python_tuple)
CfgLoader.add_constructor('!join', CfgLoader.join)
CfgLoader.add_constructor('!get_module', CfgLoader.get_module)
CfgLoader.add_constructor('!get_func', CfgLoader.get_module)
CfgLoader.add_constructor('!edict', CfgLoader.construct_python_edict)
CfgLoader.add_constructor('!cross_ref', CfgLoader.construct_crossRef)

class CfgDumper(yaml.SafeDumper):
    def represent_python_edict(self, data):
        return self.represent_mapping('!edict', dict(data))
    def represent_python_tuple(self, data):
        return self.represent_sequence('!tuple', list(data))
    def represent_python_partial(self, data):
        cls_or_func = data.func
        module = inspect.getmodule(cls_or_func)
        return self.represent_sequence('!get_module', [module.__name__+'.'+cls_or_func.__name__, data.keywords])
CfgDumper.add_representer(edict, CfgDumper.represent_python_edict)
CfgDumper.add_representer(tuple, CfgDumper.represent_python_tuple)
CfgDumper.add_representer(partial, CfgDumper.represent_python_partial)

if __name__ == '__main__':

  import torch
  from src.datasets.cifar import get_transforms
  bn = partial(torch.nn.BatchNorm2d, affine=False)
  tr = partial(get_transforms, cutout=False)
  doc = yaml.dump({'bn': bn, 'tr': tr}, Dumper=CfgDumper)
  print(repr(doc))
  thing = yaml.load(doc, Loader=CfgLoader)
  print(thing)
  print(thing['bn'](128))
  print(thing['tr'](True, 0, 0))
