import sys
import os
sys.path.append(os.getcwd())
from functools import partial

import yaml
import importlib

#import src.models.layers.base.OP_CFG

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
            search_path = [search_path] if isinstance(search_path, str) else search_path
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

#    def construct_OP_CFG(self, node):
#        return OP_CFG(self.construct_sequence(node))

    def join(self, node):
        return ''.join([str(i) for i in self.construct_sequence(node)])

    def get_module(self, node):
#        module_name = str(self.construct_scalar(node.value[0])).split('.')
#        args = self.construct_mapping(node.value[1])
        name_args = self.construct_sequence(node, deep=True)
        module = get_submodule_by_name(name_args[0])
        if len(name_args) > 1:
            return partial(module, **name_args[1])
#            return module(**name_args[1])
        else:
            return module


CfgLoader.add_constructor(
    u'tag:yaml.org,2002:python/tuple',
    CfgLoader.construct_python_tuple)
#CfgLoader.add_constructor(
#    u'!!python/object/new:src.models.layers.base.OP_CFG',
#    CfgLoader.construct_OP_CFG)
CfgLoader.add_constructor('!join', CfgLoader.join)
CfgLoader.add_constructor('!get_module', CfgLoader.get_module)

if __name__ == '__main__':
  doc = yaml.dump(tuple("foo bar baaz".split()))
  print(repr(doc))
  thing = yaml.load(doc, Loader=PrettySafeLoader)
  print(thing)

