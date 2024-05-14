import sys
import os
sys.path.append(os.getcwd())
import inspect
import importlib
from functools import partial
from easydict import EasyDict as edict
import yaml
import numpy as np


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
        except AttributeError as e:
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
                    except ImportError:
                        pass
                    except Exception as ee:
                        raise(ee)
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

    if default_path is not None and not name.startswith(default_path):
        module_name = '.'.join(default_path, module_name)
    return _get_submodule(module_name[-1], '.'.join(module_name[:-1]))

