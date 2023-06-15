import sys
import os
sys.path.append(os.getcwd())

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
        except Exception as e:
            print(e)
    else:
        raise(ValueError(f"[{module_name}] is not found in the package [{package_path}]"))

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

def create_submodule_from_dict(cfg: dict):
    submodule_name = cfg.get('submodule_name')
    module_name = cfg.get('module_name', None)
    package_path = cfg.get('package_path', None)
    args = cfg.get('args', {})
    submodule = _get_submodule(submodule_name, module_name, package_path)
    return submodule(**args)



class CfgLoader(yaml.SafeLoader):
    def construct_python_tuple(self, node):
        return tuple(self.construct_sequence(node))

    def join(self, node):
        return ''.join([str(i) for i in self.construct_sequence(node)])

    def get_module(self, node):
        name_args = self.construct_sequence(node)
        module_name = str(name_args[0]).split('.')
        module = _get_submodule(module_name[-1], '.'.join(module_name[:-1]))
        if len(name_args) > 1:
            return partial(module, **self.construct_mapping(name_args[1], deep=False))
        else:
            return module

CfgLoader.add_constructor(
    u'tag:yaml.org,2002:python/tuple',
    CfgLoader.construct_python_tuple)
CfgLoader.add_constructor('!join', CfgLoader.join)
CfgLoader.add_constructor('!get_module', CfgLoader.get_module)

if __name__ == '__main__':
  doc = yaml.dump(tuple("foo bar baaz".split()))
  print(repr(doc))
  thing = yaml.load(doc, Loader=PrettySafeLoader)
  print(thing)

