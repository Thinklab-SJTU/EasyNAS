from functools import reduce
from collections.abc import Iterable
from copy import deepcopy
import numpy as np
import warnings

from .base import SampleNode, sample_monitor, IIDSpace, DiscreteSpace, ContinuousSpace


class WeightedIIDSpace(IIDSpace):
    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)
    def __init__(self, cfg, embed_fn=None, label=None):
        self.cfg = cfg
        self.label = label
        self.spaces = self.extract_search_space(self.cfg)
        self._size = reduce(lambda x,y: x*y, [x.size for x in self.spaces.values()])
        self.embed_fn = embed_fn

        # Similar to easydict
#        super(SearchSpace, self).__init__(cfg)
        for k, v in cfg.items():
            assert not hasattr(self, k), f"The name ({k}) in the search space cannot be used."
            setattr(self, k, v)

    def __getitem__(self, key):
        return self.cfg.get(key)

    def extract_search_space(self, cfg, prefix="", spaces=None):
        if spaces is None: spaces = {}
        if isinstance(cfg, (DiscreteSpace, ContinuousSpace, IIDSpace)):
            prefix = prefix.rstrip('.')
            spaces[prefix] = cfg
        elif isinstance(cfg, dict):
            for k, v in cfg.items():
                self.extract_search_space(v, prefix+f"{cfg.__class__.__name__}::{k}.", spaces)
        elif isinstance(cfg, (list, tuple)):
            for i, v in enumerate(cfg):
                self.extract_search_space(v, prefix+f"{cfg.__class__.__name__}::{i}.", spaces)
        return spaces

    def build_cfg(self, subsamples):
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

        cfg = deepcopy(self.cfg)
        for prefix, sub_sample in subsamples.items():
            keys = [tmp.split('::')[-1] for tmp in prefix.split('.')]
            tmp = cfg
            for k in keys[:-1]:
                tmp = get_item(tmp, k)
            set_item(tmp, keys[-1], sub_sample.cfg)
        return cfg 

    def build_embedding(self, subsamples):
        if self.embed_fn:
            return self.embed_fn(subsamples)
        else:
            return len(subsamples)
                
    @sample_monitor
    def sample(self, num_to_sample=1, replace=True, label_samples=None):
        if label_samples is None: label_samples = {}
        if self.label in label_samples:
            return self.sample_from_nodes(label_samples[self.label], label_samples)

        sample_nodes = []
        self._tmp_iter_in_sample, _num = 0, num_to_sample
        while _num and self._tmp_iter_in_sample < self.MAX_ITER_NUM:
            sub_samples = [{} for _ in range(_num)]
            label_samples = {}
            for prefix, space in self.spaces.items():
                _sub_samples = space.sample(_num, replace=True, label_samples=label_samples)
                for i in range(_num):
                    sub_samples[i][prefix] = _sub_samples[i]
            sample_nodes.extend([SampleNode(self, sub_sample) for sub_sample in sub_samples])
            if not replace:
                sample_nodes = self.deduplicate(sample_nodes)
            _num = num_to_sample - len(sample_nodes)
            self._tmp_iter_in_sample += 1

        if self.label is not None: label_samples[self.label] = sample_nodes
        return sample_nodes

    def sample_from_node(self, src_sample_node, label_samples):
       sample = {}
       for prefix, s in src_sample_node.sample.items():
           assert prefix in self.spaces, f"No found {prefix} in spaces."
           assert isinstance(s, SampleNode)
           sample[prefix] = self.spaces[prefix].sample_from_node(s, label_samples)
       return SampleNode(self, sample)
    


class DiscreteSpace(SearchSpace):
    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)
    def __init__(self, candidates, num_reserve=None, reserve_replace=False, distribution=None, random_seed=None, embed_fn=None, label=None):
        """
        if num_reserve is None, we set it as 1 and return the value.
        if num_reserve is 1, we return a list with len equals to 1
        """
        self.candidates = self.to_tuple(candidates)
        self.return_list = num_reserve is not None
        self.num_reserve = 1 if num_reserve is None else num_reserve
        self.reserve_replace = reserve_replace
        self.rdm = np.random.RandomState(random_seed)
        self.cand_sizes = [cand.size if isinstance(cand, SearchSpace) else 1 for cand in self.candidates]
        self._size = reduce(lambda x,y: x+y, self.cand_sizes)

        self.distribution = [s/self._size for s in self.cand_sizes] if distribution is None else distribution
        self.embed_fn = embed_fn
        self.label = label

    def to_tuple(self, candidates):
        for i in range(len(candidates)):
            if isinstance(candidates[i], list):
                candidates[i] = tuple(candidates[i])
        return candidates


    @sample_monitor
    def sample(self, num_to_sample=1, replace=True, label_samples=None):
        if label_samples is None: label_samples = {}
        if self.label in label_samples:
            return self.sample_from_nodes(label_samples[self.label], label_samples)

        sample_nodes = []
        self._tmp_iter_in_sample, _num = 0, num_to_sample
        while _num and self._tmp_iter_in_sample < self.MAX_ITER_NUM:
            for i in range(_num):
                sample = {}
                for idx in self.rdm.choice(range(len(self.candidates)), size=self.num_reserve, p=self.distribution, replace=self.reserve_replace):
                    cand = self.candidates[idx] 
                    if isinstance(cand, SearchSpace):
                        cand = cand.sample(num_to_sample=1, replace=True, label_samples=label_samples)[0]
                    sample[idx] = cand 
                sample_nodes.append(SampleNode(self, sample))
            if not replace:
                sample_nodes = self.deduplicate(sample_nodes)
            _num = num_to_sample - len(sample_nodes)
            self._tmp_iter_in_sample += 1

        if self.label is not None: label_samples[self.label] = sample_nodes
        return sample_nodes

    def sample_from_node(self, src_sample_node, label_samples):
        sample = {}
        for idx, s in src_sample_node.sample.items():
            cand = self.candidates[idx]
            if isinstance(cand, SearchSpace):
                if isinstance(s, SampleNode):
                    cand = cand.sample_from_node(s, label_samples)
                else: 
                    cand = cand.sample(num_to_sample=1, replace=True, label_samples=label_samples)[0]
            sample[idx] = cand
        return SampleNode(self, sample)
    

    def build_cfg(self, sample):
        cfg = []
        for cand_idx, sample in sample.items():
            if isinstance(sample, SampleNode):
                cfg.append(sample.cfg)
            else: cfg.append(sample)
        return cfg if len(cfg)>1 or self.return_list else cfg[0]

    def build_embedding(self, sample):
        if self.embed_fn:
            return self.embed_fn(sample)
        else:
            embed = [0 for _ in range(len(self.candidates))]
            for cand_idx in sample.keys():
                embed[int(cand_idx)] = 1./self.num_reserve
            return tuple(embed)

    def __iter__(self):
        return iter(self.candidates)
    def __getitem__(self, idx):
        return self.candidates[idx]
    def __len__(self):
        return len(self.candidates)
    def __repr__(self):
        string = f"DiscreteSpace(candidate={self.candidates})"
        return string
    def index(self, v):
        return self.candidates.index(v)
    
class ContinuousSpace(SearchSpace):
    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, start, end, num_reserve=None, reserve_replace=True, distribution='uniform', random_seed=None, embed_fn=None, label=None):
        """
        if num_reserve is None, we set it as 1 and return the value.
        if num_reserve is 1, we return a list with len equals to 1
        """
        self.start = start
        self.end = end
        self._size = end - start
        self.return_list = num_reserve is not None
        self.num_reserve = 1 if num_reserve is None else num_reserve
        self.reserve_replace = reserve_replace
        self.distribution = distribution
        self.rdm = np.random.RandomState(random_seed)
        self.embed_fn = embed_fn 
        self.label = label

    @sample_monitor
    def sample(self, num_to_sample=None, replace=True, label_samples=None):
        if label_samples is None: label_samples = {}
        if self.label in label_samples:
            return self.sample_from_nodes(label_samples[self.label], label_samples)

        def _sample_once(num, replace):
            if replace:
                samples = tuple(getattr(self.rdm, self.distribution)(self.start, self.end, num))
                return samples # if return_list else samples[0]
            else:
                samples = set()
                _num = num
                while _num > 0:
                    samples.union(set(getattr(self.rdm, self.distribution)(self.start, self.end, _num)))
                    if not replace:
                        samples = self.deduplicate(samples)
                    _num = num - len(samples)
                return tuple(samples) # if return_list else samples[0]

        if replace:
            samples = []
            for _ in range(num_to_sample):
                samples.append(_sample_once(self.num_reserve, self.reserve_replace))
        else:
            samples = set()
            while len(samples) < num_to_sample:
                samples.add(_sample_once(self.num_reserve, self.reserve_replace))

#        sample_nodes = [SampleNode(self, {sample/self.size: sample}) for sample in samples]
        sample_nodes = []
        for sample in samples:
            sample_nodes.append(SampleNode(self, {s/self.size: s for s in sample}))
        if self.label is not None: label_samples[self.label] = sample_nodes
        return sample_nodes

    def sample_from_node(self, src_sample_node, label_samples):
        sample = {}
        for ratio in src_sample_node.sample.keys():
            sample[ratio] = ratio * self.size + self.start
        return SampleNode(self, sample)

    def __len__(self):
        return self.end - self.start

    def build_cfg(self, subsamples):
        cfg = []
        for idx, sample in subsamples.items():
            if isinstance(sample, SampleNode):
                cfg.append(sample.cfg)
            else: cfg.append(sample)
        return cfg if len(cfg)>1 or self.return_list else cfg[0]

    def build_embedding(self, sample):
        if self.embed_fn:
            return self.embed_fn(sample)
        else:
            vals = [v for v in sample.values()]
            return vals[0] if len(vals)==0 else tuple(vals)

