from abc import ABC,abstractmethod
from functools import reduce, partial
from itertools import product
import inspect
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from copy import deepcopy
import math

import torch
import numpy as np
import warnings

from builder.utils import get_submodule_by_name

def get_item(src, idx):
    if isinstance(src, dict): 
        return src.get(idx)
    elif isinstance(src, (list, tuple)):
        return src[int(idx)]
    else:
        raise(TypeError(f"Index {idx} from {src} is not supported"))

def set_item(src, idx, dst):
    if isinstance(src, dict): 
        src[idx] = dst
    elif isinstance(src, (list, tuple)):
        src[int(idx)] = dst
    else:
        raise(TypeError(f"Index {idx} from {src} is not supported"))

def my_product(iter_fns, out=None):
    depth = len(iter_fns)
    if out is None: out = [None for _ in range(depth)]
    for tmp in iter_fns[0]():
        out[-depth] = tmp
        if depth == 1:
            yield tuple(out)
        else:
            yield from my_product(iter_fns[1:], out)

class SampleNode(object):
    def __init__(self, space, sample):
        """
        If space is IIDSpace, sample should be {prefix: samples}. If space is DiscreteSpace, sample should be {idxes: samples}. If space is ContinuousSpace, sample should be {sample/range: samples}
        """
        self.space = space
        self.sample = sample

    def _build_sample_attr(self, name, fn, sample):
        if hasattr(self, name):
            return getattr(self, name)
        setattr(self, name, fn(sample))
        return getattr(self, name)

    @property
    def config(self, replace_setting=None):
        return self.space.build_config(self.sample)
#        return self._build_sample_attr('_config', self.space.build_config, self.sample)

    def replace_setting(self, new_setting):
        self.sample.update(new_setting)

    @property
    def embedding(self):
        return self._build_sample_attr('_embedding', self.space.build_embedding, self.sample)

    def get_sampleNode(self, sample):
        if isinstance(sample, SampleNode):
            return [sample]
        elif isinstance(sample, (list, tuple)):
            sub_samplenodes = []
            for tmp in sample:
                sub_samplenodes.extend(self.get_sampleNode(tmp))
            return sub_samplenodes
        elif isinstance(sample, dict):
            sub_samplenodes = []
            for tmp in sample.values():
                sub_samplenodes.extend(self.get_sampleNode(tmp))
            return sub_samplenodes
        else:
            return []
#            raise(TypeError(f"sample should be either SampleNode, list, tuple, or dict, but get {type(sampe)}"))

    def __eq__(self, other):
        return hash(self) == hash(other)
#        return self.config == other.config

    def __hash__(self):
        def to_hash(param):
            if isinstance(param, dict):
                return tuple(to_hash(v) for k, v in sorted(param.items(), key=lambda item: item[0]))
            elif isinstance(param, list):
                return tuple(param)
            else: return param
        return hash(to_hash(self.sample))
#        return hash(tuple(to_hash(v) for v in self.sample.values()))
#        return hash(tuple(hash(v) if isinstance(v, SampleNode) else v for v in self.sample.values()))
#        return hash((self.embedding, *self.sample.values()))

    def __repr__(self):
        string = f"SampleNode(space_type={self.space.__class__.__name__}, sample={self.config})"
        return string

###########################################

def sample_monitor(func):
    def inner(self, num_to_sample, replace=False, *args, **kwargs):
        if not replace and num_to_sample >= self.size:
            msg = "Required sample number is larger than the space. You should decrease 'num_to_sample' or set replace as True, otherwise, there contains at leaset one ContinuousSpace."
            warnings.warn(msg, RuntimeWarning)

        samples = func(self, num_to_sample, replace, *args, **kwargs)

        _num = num_to_sample - len(samples)
        if getattr(self, '_tmp_iter_in_sample', 0) >= self.MAX_ITER_NUM and _num > 0:
            msg = f"Reach the maximum iteration number to sample but still require {_num} samples. Please check and enlarge your search space or decrease the 'num_to_sample' or increase MAX_ITER_NUM (default: 10000) and retry."
            raise(RuntimeWarning(msg))
        return samples
    return inner

###########################################

class SearchSpace(object):
    def __new__(cls, **kwargs):
        space = kwargs['space']
        repeat = kwargs.pop('num_repeat', None)
        ind_repeat = kwargs.pop('independent', True)
        flatten = kwargs.pop('flatten', False)
        if isinstance(space, dict):
            space = IIDSpace(**kwargs)
#            return IIDSpace.__new__(IIDSpace, **kwargs)
        elif isinstance(space, (list, tuple)):
            if flatten:
                space = FlattenSampledDiscreteSpace(**kwargs)
            else:
                space = DiscreteSpace(**kwargs)
#            return DiscreteSpace.__new__(DiscreteSpace, **kwargs)
        elif isinstance(space, str):
            space = ContinuousSpace(**kwargs)
#            return ContinuousSpace.__new__(ContinuousSpace, **kwargs)
        else:
            raise(NotImplementedError("No Implementation as such a search space"))
        if repeat:
            return RepeatSpace(space, num_repeat=repeat, independent=ind_repeat)
        else: return space

###########################################

class _SearchSpace(ABC):
    MAX_ITER_NUM = 10000
    _cnt = 0
    _samplers = {}
    def __init__(self, space, sampler_cfg, embed_fn=None, label=None):
        self.space = space
        self.embed_fn = embed_fn
        self.sampler_cfg = deepcopy(sampler_cfg)
        if label is None:
            self.label = f'_SearchSpace#{_SearchSpace._cnt}'
        else: self.label = label
        _SearchSpace._cnt += 1
        self._child_spaces = self.extract_child_space(space)
        if sampler_cfg is not None and self.label not in _SearchSpace._samplers:
            if isinstance(sampler_cfg, str):
                sampler_cfg = {'submodule_name': sampler_cfg}
            sampler_cls = get_submodule_by_name(sampler_cfg.get('submodule_name'), search_path='src.search_space.sampler')
            if 'space_size' in inspect.getfullargspec(sampler_cls.__init__).args:
                sampler_cfg.setdefault('args', {}).update({'space_size': self.size})
            _SearchSpace._samplers[self.label] = sampler_cls(**sampler_cfg.get('args', {}))

    def extract_child_space(self, space, prefix="", child_spaces=None):
        if child_spaces is None: child_spaces = {}
        if isinstance(space, _SearchSpace):
            prefix = prefix.rstrip('.')
            child_spaces[prefix] = space
        elif isinstance(space, dict):
            for k, v in space.items():
                self.extract_child_space(v, prefix+f"{space.__class__.__name__}::{k}.", child_spaces)
        elif isinstance(space, (list, tuple)):
            for i, v in enumerate(space):
                self.extract_child_space(v, prefix+f"{space.__class__.__name__}::{i}.", child_spaces)
        return child_spaces

    @property
    def size(self):
        name = '_size'
        if hasattr(self, name):
            return getattr(self, name)
        setattr(self, name, self.get_size())
        return getattr(self, name)
    @property
    def sampler(self):
        return _SearchSpace._samplers.get(self.label, None)

    def deduplicate(self, samples):
        return list(set(samples))

    @abstractmethod
    def get_size(self, label_computed=None):
        pass
    @abstractmethod
    def build_embedding(self, *args, **kwargs):
        pass
    @abstractmethod
    def build_config(self, *args, **kwargs):
        pass
    @abstractmethod
    def _sample_once(self, label_samples=None):
        pass
    @abstractmethod
    def sample_from_node(self, node, label_samples):
        pass
    @abstractmethod
    def enum_space(self):
        pass

    @sample_monitor
    def sample(self, num_to_sample=1, replace=False):
        sample_nodes = []
        self._tmp_iter_in_sample, _num = 0, num_to_sample
        while _num and self._tmp_iter_in_sample < self.MAX_ITER_NUM:
            label_samples = {}
            sample_nodes.append(self._sample_once(label_samples))
            if not replace:
                sample_nodes = self.deduplicate(sample_nodes)
            _num = num_to_sample - len(sample_nodes)
            self._tmp_iter_in_sample += 1
        return sample_nodes

    def build_nodes(self, src_samples):
        sample_nodes = []
        for idx, sample in enumerate(src_samples):
            sample_nodes.append(self.build_node(sample))
        return sample_nodes

    def sample_from_nodes(self, src_sample_nodes, label_samples=None):
        sample_nodes = [] 
        if label_samples is None: label_samples = {}
        for idx, node in enumerate(src_sample_nodes):
            assert self.__class__.__name__ == node.space.__class__.__name__
            sample_nodes.append(self.sample_from_node(node, label_samples))
        return sample_nodes

    def named_sampler(self, prefix='', recurse=True, memo=None):
        if memo is None: memo = set()
        if self.sampler is not None and self.sampler not in memo:
            memo.add(self.sampler)
            if prefix: name = '.'.join([prefix, name])
            yield name, self.sampler
        if recurse:
            for name, child_space in self._child_spaces.items():
                yield from child_space.named_sampler(prefix=name, recurse=recurse, memo=memo)

    def named_sampler_weights(self, prefix='', recurse=True, memo=None):
        if memo is None: memo = set()
        if self.sampler is not None and self.sampler not in memo:
            memo.add(self.sampler)
            for name, weight in self.sampler.named_weights():
                if prefix: name = '.'.join([prefix, name])
                yield name, weight
        if recurse:
            for name, child_space in self._child_spaces.items():
                yield from child_space.named_sampler_weights(prefix=name, recurse=recurse, memo=memo)

    def sampler_weights(self, recurse=True):
        for name, weight in self.named_sampler_weights(recurse=recurse):
            yield weight

    def apply(self, fn):
        for _, space in self._child_spaces.items():
            space.apply(fn)
        fn(self)
        return self

    def apply_sampler_weights(self, fn, recurse=True, memo=None):
        if memo is None:
            memo = set()
#        new_weights = {}
        if self.sampler is not None and self.sampler not in memo:
            memo.add(self.sampler)
            for key, param in self.sampler._weights.items():
                out_param = fn(param)
    #            new_weights[name] = out_param
                self.sampler._weights[key] = out_param
                if hasattr(self.sampler, key): 
                    delattr(self.sampler, key)
                    setattr(self.sampler, key, out_param)

        if recurse:
            for _, space in self._child_spaces.items():
                space.apply_sampler_weights(fn, recurse, memo)

    def __repr__(self):
        string = f"{self.__class__.__name__}(space_len={len(self.space)}, label={self.label}, sampler={self.sampler})"
        return string
    def __getitem__(self, key):
        return self.space[key]
    def __setitem__(self, key, value):
        self.space[key] = value

    @abstractmethod
    def discretize(self, **replace_settings):
        pass

    def show_info(self, prefix='', showed_label=None):
        if showed_label is None: showed_label = set()
        if self.label not in showed_label and self.sampler is not None:
            name = '.'.join([tmp.split('::')[-1] for tmp in prefix.split('.')])
            string = f"name={name}, label={self.label}, space={self.__class__.__name__}, sampler={self.sampler.__class__.__name__}: "
            for name, weight in self.sampler.named_weights():
                string += f"\n\t{name}={weight.data};\nnormed={self.sampler.norm_fn(weight).data}"
            print(string)
            showed_label.add(self.label)
        for child_prefix, child_space in self._child_spaces.items():
            child_space.show_info(prefix='.'.join([prefix, child_prefix]), showed_label=showed_label)

#        for prefix, child_space in self._child_spaces.items():
#            name = '.'.join([tmp.split('::')[-1] for tmp in prefix.split('.')])
#            string = f"name={name}, label={child_space.label}, space={child_space.__class__.__name__}, sampler={child_space.sampler.__class__.__name__}"
#            for n, v in child_space.named_sampler_weights():
#                string += f"\n{n}={v.data}"
#            print(string)

    def _get_item_by_name(self, src, name: str):
        keys = [tmp.split('::')[-1] for tmp in name.split('.')]
        tmp = src
        for k in keys:
            if k == '': continue
            elif isinstance(tmp, (list, tuple)):
                tmp = tmp[int(k)]
            elif isinstance(tmp, dict): 
                tmp = tmp.get(k)
            else:
                raise(TypeError(f"Index {name} from {src} is not supported"))
        return tmp

    def _set_item_by_name(self, src, name: str, value):
        keys = [tmp.split('::')[-1] for tmp in name.split('.')]
        if len(keys) == 0:
            return value
        tmp = self._get_item_by_name(src, '.'.join(keys[:-1]))
        idx = keys[-1]
        if isinstance(tmp, dict): 
            tmp[idx] = value
        elif isinstance(tmp, (list, tuple)):
            tmp[int(idx)] = value
        else:
            raise(TypeError(f"Setting {value} to the item {name} of {src} is not supported"))
        return src

    def new_space(self, **kwargs):
        for key in inspect.getfullargspec(self.__class__.__init__).args:
            if key not in kwargs and hasattr(self, key):
                kwargs[key] = getattr(self, key)
        if getattr(kwargs, 'return_list', False):
            kwargs['num_reserve'] = None
        for prefix, child_space in self._child_spaces.items():
            new_child_space = child_space.new_space()
            self._set_item_by_name(kwargs['space'], prefix, new_child_space)
        return self.__class__(**kwargs)




###########################################
class IIDSpace(_SearchSpace):
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)
    def __init__(self, space, sampler_cfg=None, embed_fn=None, label=None):
        super(IIDSpace, self).__init__(space, sampler_cfg, embed_fn, label)

        # Similar to easydict
        for k, v in self.space.items():
            assert not hasattr(self, k), f"The name ({k}) in the search space cannot be used."
            setattr(self, k, v)

    #TODO: We should build the tree of search space according to the label rather than the name.
    def get_size(self, label_computed=None):
        try:
            if label_computed is None: label_computed = set()
            if self.label in label_computed: 
                return 1
            label_computed.add(self.label)
            return reduce(lambda x,y: x*y, [1 if x.label in label_computed else x.get_size(label_computed) for x in self._child_spaces.values()])
        except:
            return len(list(self.enum_space()))

    def build_config(self, sample):
        def get_item(src, idx):
            if isinstance(src, dict): 
                return src.get(idx)
            elif isinstance(src, (list, tuple)):
                return src[int(idx)]
            else:
                raise(TypeError(f"Index {idx} from {src} is not supported"))
        def set_item(src, idx, dst):
            if isinstance(dst, SampleNode):
                set_item(src, idx, dst.config)
            elif isinstance(src, dict): 
                src[idx] = dst
            elif isinstance(src, (list, tuple)):
                src[int(idx)] = dst
            else:
                raise(TypeError(f"Index {idx} from {src} is not supported"))

        config = deepcopy(self.space)
        for prefix, sub_sample in sample.items():
            keys = [tmp.split('::')[-1] for tmp in prefix.split('.')]
            tmp = config
            for k in keys[:-1]:
                tmp = get_item(tmp, k)
            set_item(tmp, keys[-1], sub_sample)
        return config

    def build_embedding(self, sample):
        if self.embed_fn:
            return self.embed_fn(sample)
        else:
            return None #len(sample)
                
    def _sample_once(self, label_samples=None):
        if label_samples is None: label_samples = {}
        if self.label in label_samples:
            return self.sample_from_node(label_samples[self.label], label_samples)
        sample = {}
        for prefix, space in self._child_spaces.items():
            sample[prefix] = space._sample_once(label_samples)
        sample = SampleNode(self, sample)
        if not self.label.startswith('_SearchSpace#'): label_samples[self.label] = sample
        return sample

    def sample_from_node(self, src_sample_node, label_samples):
        sample = {}
        for prefix, s in src_sample_node.sample.items():
            assert prefix in self._child_spaces, f"No found {prefix} in space."
            assert isinstance(s, SampleNode)
            if self._child_spaces[prefix].label.startswith('_SearchSpace#') or self._child_spaces[prefix].label==s.space.label:
                sample[prefix] = self._child_spaces[prefix].sample_from_node(s, label_samples)
            else:
                sample[prefix] = self._child_spaces[prefix]._sample_once(label_samples)
        return SampleNode(self, sample)

    def build_node(self, src_sample):
        sample = {}
        queue = [("", self.space, src_sample)]
        while len(queue) > 0:
            prefix, space, src = queue.pop()
            if isinstance(space, _SearchSpace):
                prefix = prefix.rstrip('.')
                sample[prefix] = space.build_node(src)
            elif isinstance(space, dict):
                for sub_k, sub_v in src.items():
                    queue.append((prefix+f"{space.__class__.__name__}::{sub_k}.", space[sub_k], sub_v))
            elif isinstance(space, (list, tuple)):
                for i, v in enumerate(src):
                    queue.append((prefix+f"{space.__class__.__name__}::{i}.", space[i], v))
        return SampleNode(self, sample)


    def enum_from_node(self, src_sample_node, label_sample):
        for child_sample in my_product([partial(space.enum_from_node, src_sample_node.sample[prefix], label_sample) for prefix, space in self._child_spaces.items()]):
            sample = {prefix: sub_sample for prefix, sub_sample in zip(self._child_spaces.keys(), child_sample)}
            yield SampleNode(self, sample)

    def enum_space(self, recurse=True, label_sample=None):
        if not recurse:
            yield self
        else:
            if label_sample is None: label_sample = {}
            if self.label in label_sample:
                yield from self.enum_from_node(label_sample[self.label], label_sample)
            else:
                for child_sample in my_product([partial(space.enum_space, recurse, label_sample) for space in self._child_spaces.values()]):
                    sample = {prefix: sub_sample for prefix, sub_sample in zip(self._child_spaces.keys(), child_sample)}
                    sample = SampleNode(self, sample)
                    label_sample[self.label] = sample
                    yield sample
                    del label_sample[self.label]
    
    def discretize(self, **replace_settings):
        for prefix, child_space in self._child_spaces.items():
            replace_settings[prefix] = child_space.discretize()
        return self.build_config(replace_settings)

class RepeatSpace(IIDSpace):
    def __init__(self, space, num_repeat, independent=True, label=None):
        assert isinstance(space, _SearchSpace)
        self.num_repeat = num_repeat
        self.independent = independent
        space = {0: space}
        if independent:
            for i in range(1, num_repeat):
                repeat_label = None if label is None else label+'repeat_%d'%i
                space[i] = space[0].new_space(label=repeat_label)
#        space = {i: space.new_space(label=None) if i>0 and independent else space for i in range(num_repeat)}
        super(IIDSpace, self).__init__(space, sampler_cfg=None, embed_fn=None, label=label)

    def __len__(self):
        return self.num_repeat

    def build_config(self, sample):
        _config = super().build_config(sample)
        if self.independent:
            config = [None for _ in range(self.num_repeat)]
            for k, v in _config.items():
                config[int(k)] = v
        else:
            config = [_config['0'] for _ in range(self.num_repeat)]
        return config

    def discretize(self, **replace_settings):
        return [self.space[i].discretize(**replace_settings) for i in range(self.num_repeat)]

###########################################
class DiscreteSpace(_SearchSpace):
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)
    def __init__(self, space, sampler_cfg='UniformDiscreteSampler', num_reserve=None, reserve_replace=False, embed_fn=None, label=None):
        """
        if num_reserve is None, we set it as 1 and return the value.
        if num_reserve is 1, we return a list with len equals to 1
        """
        space = self.to_tuple(space)
        super(DiscreteSpace, self).__init__(space, sampler_cfg, embed_fn, label)
        self.return_list = num_reserve is not None
        self.num_reserve = 1 if num_reserve is None else num_reserve
        self.reserve_replace = reserve_replace

    def to_tuple(self, space):
        for i in range(len(space)):
            if isinstance(space[i], list):
                space[i] = tuple(space[i])
        return space 

    def get_size(self, label_computed=None):
        try:
            if label_computed is None: label_computed = set()
            if self.label in label_computed: 
                return 1
            label_computed.add(self.label)
            cand_sizes = [cand.get_size(label_computed) if isinstance(cand, _SearchSpace) else 1 for cand in self.space]
            return reduce(lambda x,y: x+y, cand_sizes)
        except:
            return len(list(self.enum_space()))

    def _sample_once(self, label_samples=None):
        if label_samples is None: label_samples = {}
        if self.label in label_samples:
            return self.sample_from_node(label_samples[self.label], label_samples)
        sample = {}
        for idx in self.sampler.sample(range(len(self.space)), num=self.num_reserve, replace=self.reserve_replace):
            cand = deepcopy(self.space[idx])
            for ch_prefix, child_space in self._child_spaces.items():
                keys = [tmp.split('::')[-1] for tmp in ch_prefix.split('.')]
                if int(keys[0]) == idx:
                    ch_cand = child_space._sample_once(label_samples=label_samples)
                    if len(keys) == 1:
                        cand = ch_cand
                        break
                    else:
                        self._set_item_by_name(cand, '.'.join(keys[1:]), ch_cand)
            sample[idx] = cand 
        sample = SampleNode(self, sample)
        if not self.label.startswith('_SearchSpace#'): label_samples[self.label] = sample
        return sample

    def sample_from_node(self, src_sample_node, label_samples):
        sample = {}
        for idx, s in src_sample_node.sample.items():
            cand = deepcopy(self.space[idx])
            for ch_prefix, child_space in self._child_spaces.items():
                keys = [tmp.split('::')[-1] for tmp in ch_prefix.split('.')]
                if int(keys[0]) == idx:
                    ch_s = self._get_item_by_name(s, '.'.join(keys[1:]))
                    if isinstance(ch_s, SampleNode):
                        ch_cand = child_space.sample_from_node(ch_s, label_samples)
                    else:
                        ch_cand = child_space._sample_once(label_samples=label_samples)
                    if len(keys) == 1:
                        cand = ch_cand
                        break
                    else:
                        self._set_item_by_name(cand, '.'.join(keys[1:]), ch_cand)
            sample[idx] = cand
        return SampleNode(self, sample)

    def build_node(self, src_sample):
        sample = {}
        for i, src in enumerate(src_sample):
            try:
                idx = self.space.index(src)
                sample[idx] = src
            except IndexError as e:
                print("Default sample has not supported nesting DiscreteSpace yet.")
                raise(e)
        return SampleNode(self, sample)

    def build_config(self, sample):
        config = []
        for cand_idx, sample in sample.items():
            conf = deepcopy(sample)
            for ch_prefix in self._child_spaces.keys():
                keys = [tmp.split('::')[-1] for tmp in ch_prefix.split('.')]
                if int(keys[0]) == cand_idx:
                    ch_s = self._get_item_by_name(sample, '.'.join(keys[1:]))
                    if len(keys) == 1:
                        conf = ch_s.config
                        break
                    else:
                        self._set_item_by_name(conf, '.'.join(keys[1:]), ch_s.config)
            config.append(conf)
#            if isinstance(sample, SampleNode):
#                config.append(sample.config)
#            else: config.append(sample)
        return config if len(config)>1 or self.return_list else config[0]

    def build_embedding(self, sample):
        if self.embed_fn:
            return self.embed_fn(sample)
        else:
            embed = np.zeros(len(self.space))
            for cand_idx in sample.keys():
                embed[int(cand_idx)] = 1./self.num_reserve
            return embed

    def enum_from_node(self, src_sample_node, label_sample):
        assert len(src_sample_node.sample) == 1
        idx, s = list(src_sample_node.sample.items())[0]
        config = deepcopy(self.space[idx])
        enum_child_spaces = []
        for ch_prefix, child_space in self._child_spaces.items():
            keys = [tmp.split('::')[-1] for tmp in ch_prefix.split('.')]
            if int(keys[0]) == idx:
                enum_child_spaces.append((keys, child_space))
        if len(enum_child_spaces) > 0:
            for child_sample in my_product([partial(key_space[1].enum_space, recurse=True, label_sample=label_sample) for key_space in enum_child_spaces]):
                cand = deepcopy(config)
                for k_space, s in zip(enum_child_spaces, child_sample):
                    k = k_space[0]
                    if len(k) == 1:
                        cand = s
                    else:
                        self._set_item_by_name(cand, '.'.join(k[1:]), s)
                yield SampleNode(self, {idx: cand})

        else:
            yield SampleNode(self, {idx: config})

        
    def enum_space(self, recurse=True, label_sample=None):
#        print("***", self.label, label_sample)
        if not recurse:
            for config in self.space:
                yield config
        else:
            if label_sample is None: label_sample = {}
            if self.label in label_sample:
                yield from self.enum_from_node(label_sample[self.label], label_sample)
            else:
                for idx, config in enumerate(self.space):
                    enum_child_spaces = []
                    for ch_prefix, child_space in self._child_spaces.items():
                        keys = [tmp.split('::')[-1] for tmp in ch_prefix.split('.')]
                        if int(keys[0]) == idx:
                            enum_child_spaces.append((keys, child_space))
                    if len(enum_child_spaces) > 0:
                        for child_sample in my_product([partial(key_space[1].enum_space, recurse, label_sample) for key_space in enum_child_spaces]):
                            cand = deepcopy(config)
                            for k_space, s in zip(enum_child_spaces, child_sample):
                                k = k_space[0]
                                if len(k) == 1:
                                    cand = s
                                else:
                                    self._set_item_by_name(cand, '.'.join(k[1:]), s)
                            sample = SampleNode(self, {idx: cand})
                            label_sample[self.label] = sample
                            yield sample
                            del label_sample[self.label]
                    else:
                        sample = SampleNode(self, {idx: config})
                        label_sample[self.label] = sample
                        yield sample
                        del label_sample[self.label]

    def __iter__(self):
        return self.enum_space(recurse=False) #iter(self.space)
    def __len__(self):
        return len(self.space)
    def index(self, v):
        return self.space.index(v)

    def discretize(self, **replace_settings):
        out = [o.discretize(**replace_settings) if isinstance(o, _SearchSpace) else o for o in self.sampler.topk(self.space, k=self.num_reserve)]
        return out if self.return_list else out[0]

###########################################
#TODO: Maybe it is better to utilize FlattenView of a space rather than build a new space
class FlattenSampledDiscreteSpace(DiscreteSpace):
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)
    def __init__(self, space, sampler_cfg='UniformDiscreteSampler', num_reserve=None, reserve_replace=False, embed_fn=None, label=None):
        super(FlattenSampledDiscreteSpace, self).__init__(space, sampler_cfg, num_reserve=num_reserve, reserve_replace=reserve_replace, embed_fn=embed_fn, label=label)
        self.return_list, return_list_bk = False, self.return_list
        self.flattened_space = [super(FlattenSampledDiscreteSpace, self).build_config(sample.sample) for sample in self.enum_space()]
        self.return_list = return_list_bk
        self.ori_space = self.space
#        super(FlattenSampledDiscreteSpace, self).__init__(flattened_space, sampler_cfg, num_reserve, reserve_replace, embed_fn, label)

    @contextmanager
    def change_to_flattened_space(self):
        self.space = self.flattened_space
        self._child_spaces, self._child_spaces_bk = {}, self._child_spaces
        yield
        self.space = self.ori_space
        self._child_spaces = self._child_spaces_bk

    def _sample_once(self, label_samples=None):
        with self.change_to_flattened_space():
            return super(FlattenSampledDiscreteSpace, self)._sample_once(label_samples)

    def sample_from_node(self, node, label_samples):
        with self.change_to_flattened_space():
            return super(FlattenSampledDiscreteSpace, self).sample_from_node(node, label_samples)

    def build_node(self, src_sample):
        with self.change_to_flattened_space():
            return super(FlattenSampledDiscreteSpace, self).build_node(src_sample)

    def build_config(self, sample):
        with self.change_to_flattened_space():
            return super(FlattenSampledDiscreteSpace, self).build_config(sample)

    def build_embedding(self, sample):
        with self.change_to_flattened_space():
            return super(FlattenSampledDiscreteSpace, self).build_embedding(sample)

    def discretize(self, **replace_settings):
        with self.change_to_flattened_space():
            return super(FlattenSampledDiscreteSpace, self).discretize(**replace_settings)
    def index(self, v):
        return self.flattened_space.index(v)

    

###########################################
class ContinuousSpace(_SearchSpace):
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)

    def __init__(self, space, sampler_cfg='UniformContinousSampler', num_reserve=None, embed_fn=None, label=None):
        """
        space is a string with format "start:end"
        if num_reserve is None, we set it as 1 and return the value.
        if num_reserve is 1, we return a list with len equals to 1
        """
        super(ContinuousSpace, self).__init__(space, sampler_cfg, embed_fn, label)
        self.start, self.end = [float(tmp) for tmp in space.split(':')]
#        self.sampler.set_param(space)
        self.return_list = num_reserve is not None
        self.num_reserve = 1 if num_reserve is None else num_reserve

    def get_size(self, label_computed=None):
        if label_computed is None: label_computed = set()
        if self.label in label_computed: 
            return 1
        label_computed.add(self.label)
        if math.isinf(self.end) or math.isinf(self.start):
            return 1
        else:
            return self.end - self.start

    def _sample_once(self, label_samples=None):
        if label_samples is None: label_samples = {}
        if self.label in label_samples:
            return self.sample_from_node(label_samples[self.label], label_samples)
        sample = tuple(self.sampler.sample(self.num_reserve))
        sample = SampleNode(self, {s: s*self.size+self.start for s in sample})
#        sample = SampleNode(self, {(s-self.start)/self.size: s for s in sample})
        if not self.label.startswith('_SearchSpace#'): label_samples[self.label] = sample
        return sample

    def sample_from_node(self, src_sample_node, label_samples):
        sample = {}
        for ratio in src_sample_node.sample.keys():
            sample[ratio] = ratio * self.size + self.start
        return SampleNode(self, sample)

    def build_node(self, src_sample):
        sample = {}
        for v in src_sample:
            ratio = (v-self.start) / self.size
            sample[ratio] = v 
        return SampleNode(self, sample)

    def __len__(self):
        return self.end - self.start

    def build_config(self, sample):
        cfg = []
        for idx, sub_sample in sample.items():
            if isinstance(sub_sample, SampleNode):
                cfg.append(sub_sample.config)
            else: cfg.append(sub_sample)
        return cfg if len(cfg)>1 or self.return_list else cfg[0]

    def build_embedding(self, sample):
        if self.embed_fn:
            return self.embed_fn(sample)
        else:
            vals = np.array([(v-self.start)/self.size for v in sample.values()])
            return vals

    def enum_space(self, recurse=True, label_sample=None):
        raise(TypeError("ContinuousSpace does not support to enumrate the space."))

    def discretize(self, **replace_settings):
        out = self.sampler.topk(self.space, k=self.num_reserve)
        return out if self.return_list else out[0]

class ContinuousTensorSpace(ContinuousSpace):
    def __init__(self, space, sampler_cfg='UniformContinousSampler', size=1, embed_fn=None, label=None):
        """
        space is a string with format "start:end"
        """
        super(ContinuousSpace, self).__init__(space, sampler_cfg, embed_fn, label)
        self.start, self.end = [float(tmp) for tmp in space.split(':')]
#        self.sampler.set_param(space)
        if isinstance(size, int):
            self.size = (size, )
        else: self.size = tuple(size)

    def get_size(self, label_computed=None):
        return super(ContinuousTensorSpace, self).get_size(label_computed)

    def _sample_once(self, label_samples=None):
        if label_samples is None: label_samples = {}
        if self.label in label_samples:
            return self.sample_from_node(label_samples[self.label], label_samples)
        sample = tuple(self.sampler.sample(self.size))
        sample = SampleNode(self, {0: sample})
        if not self.label.startswith('_SearchSpace#'): label_samples[self.label] = sample
        return sample

    def sample_from_node(self, src_sample_node, label_samples):
        sample = {}
        for k, v in src_sample_node.sample.items():
            sample[k] = v 
        return SampleNode(self, sample)

    def __len__(self):
        return self.end - self.start

    def build_config(self, sample):
        return sample[0]

    def build_embedding(self, sample):
        if self.embed_fn:
            return self.embed_fn(sample)
        else:
            vals = (sample[0]-self.start)/(self.end-self.start)
            return vals
    def discretize(self, **replace_settings):
        raise(TypeError("ContinuousTensorSpace does not support to discretize the space."))
