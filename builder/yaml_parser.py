import sys
import os
sys.path.append(os.getcwd())
import inspect
from functools import partial, reduce
from easydict import EasyDict as edict
import yaml
import numpy as np

from src.search_space.base import SearchSpace, SampleNode
from .utils import get_submodule_by_name

def parse_cfg(yaml_file):
    with open(yaml_file, 'r') as f:
        tmp_cfg = yaml.load(f.read(), CfgLoader)

    if isinstance(tmp_cfg, dict):
        cfg = {}
        for k, v in tmp_cfg.items():
            cfg[k] = parse_cfg(v) if isinstance(v, str) and v.endswith('yaml') and os.path.isfile(v) else v
    elif isinstance(tmp_cfg, list):
        cfg = []
        for v in tmp_cfg:
            cfg.append(parse_cfg(v) if isinstance(v, str) and os.path.isfile(v) else v)
    else:
        cfg = deepcopy(tmp_cfg)

    return cfg


DIGITS={'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9}
def str2int(s):
    return int(float(s))
def str2float(s):
    return float(s)
    s=s.split('.')
    if s[0]==0:
        return 0+reduce(lambda x,y:x/10+y , map(lambda x:DIGITS[x],s[1][::-1]))/10
    else:
        return reduce(lambda x,y:x*10+y,map(lambda x:DIGITS[x],s[0]))+reduce(lambda x,y:x/10+y , map(lambda x:DIGITS[x],s[1][::-1]))/10

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
        if isinstance(node, yaml.ScalarNode):
            name = self.construct_scalar(node)
            name_args = [name, {}]
        elif isinstance(node, yaml.SequenceNode):
            name_args = self.construct_sequence(node, deep=True)
        module = get_submodule_by_name(name_args[0])
        if len(name_args) > 1:
            return partial(module, **name_args[1])
        else:
            return partial(module)

    def construct_python_edict(self, node):
        return edict(self.construct_mapping(node, deep=True))

    def _update_dict(self, data, update):
        if not update.pop('recurse', True):
            return data.update(update)
        for k, v in update.items():
            if k in data and isinstance(v, dict):
                self._update_dict(data[k], v)
            else:
                data[k] = v

    # !include crossFile
    # !include [crossFile, dict(kwargs)]
    # !include crossFile:key1:key2:...
    # !include [crossFile:key1:key2:..., dict(kwargs)]
    def construct_crossRef(self, node):
        if isinstance(node, yaml.ScalarNode):
            crossRef, replaceArgs = self.construct_scalar(node), {}
        elif isinstance(node, yaml.SequenceNode):
            crossRef_replaceArgs = self.construct_sequence(node, deep=True)
            crossRef, replaceArgs = crossRef_replaceArgs[0], {} if len(crossRef_replaceArgs)==1 else crossRef_replaceArgs[1]

        crossRef = crossRef.split(':')
        with open(crossRef[0], 'r') as f:
            data = yaml.load(f.read(), CfgLoader)
        if len(crossRef) == 2:
            data = data[crossRef[1]]
        elif len(crossRef) > 2:
            data = {k: data[k] for k in crossRef[1:]}

        if replaceArgs:
            self._update_dict(data, replaceArgs)
#            data.update(crossRef_replaceArgs[1])
        return data

    def construct_expression(self, node):
        def check_standard_expr(expr):
            for char in expr:
                assert (ord(char)>=48 and ord(char)<=58) or char in ['(', ')', '+', '-', '*', '/', ' ', '.']
        expr = ''.join([str(x) for x in self.construct_sequence(node, deep=True)])
        check_standard_expr(expr)
        return eval(expr)

    def construct_search_space(self, node):
        if isinstance(node, yaml.ScalarNode):
            ss_args = self.construct_scalar(node)
            ss_args_tmp = ss_args.split(':')
            if len(ss_args_tmp) == 2:
                ss_args = {'space': ss_args}
            else:
                assert len(ss_args_tmp) == 3
                str2type = str2int
                for tmp in ss_args_tmp:
#                    if '.' in tmp or ('e' in tmp and '-' in tmp.split('e')[-1]):
                     if str2int(tmp) != str2float(tmp):
                        str2type = str2float
                        break
                ss_args = {'space': np.arange(*[str2type(tmp) for tmp in ss_args_tmp]).tolist()}
        elif isinstance(node, yaml.SequenceNode):
            ss_args = {'space': self.construct_sequence(node, deep=True)}
        elif isinstance(node, yaml.MappingNode):
            ss_args = self.construct_mapping(node, deep=True)
            if 'space' not in ss_args:
                ss_args = {'space': ss_args}
        return SearchSpace(**ss_args)
#        def foo_constructor(loader, node):
#            instance = Foo.__new__(Foo)
#            yield instance
#            state = loader.construct_mapping(node, deep=True)
#            instance.__init__(**state)


CfgLoader.add_constructor(
    '!tuple', CfgLoader.construct_python_tuple)
CfgLoader.add_constructor('!join', CfgLoader.join)
CfgLoader.add_constructor('!get_module', CfgLoader.get_module)
CfgLoader.add_constructor('!get_func', CfgLoader.get_module)
CfgLoader.add_constructor('!edict', CfgLoader.construct_python_edict)
CfgLoader.add_constructor('!cross_ref', CfgLoader.construct_crossRef)
CfgLoader.add_constructor('!expr', CfgLoader.construct_expression)
CfgLoader.add_constructor('!search_space', CfgLoader.construct_search_space)

class CfgDumper(yaml.SafeDumper):
    def represent_python_edict(self, data):
        return self.represent_mapping('!edict', dict(data))
    def represent_python_tuple(self, data):
        return self.represent_sequence('!tuple', list(data))
    def represent_python_partial(self, data):
        cls_or_func = data.func
        module = inspect.getmodule(cls_or_func)
        return self.represent_sequence('!get_module', [module.__name__+'.'+cls_or_func.__name__, data.keywords])
    def represent_sampleNode(self, data):
        return self.represent_mapping('!sample_node', data.config)
CfgDumper.add_representer(edict, CfgDumper.represent_python_edict)
CfgDumper.add_representer(tuple, CfgDumper.represent_python_tuple)
CfgDumper.add_representer(partial, CfgDumper.represent_python_partial)
CfgDumper.add_representer(SampleNode, CfgDumper.represent_sampleNode)

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
