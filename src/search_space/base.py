from functools import reduce
from collections.abc import Iterable
from copy import deepcopy
import numpy as np
import warnings

class SampleNode(object):
    def __init__(self, space, sample):
        """
        If space is IIDSpace, sample should be {prefix: samples}. If space is DiscreteSpace, sample should be {idxes: samples}. If space is ContinuousSpace, sample should be {i: samples}
        """
        self.space = space
        self.sample = sample
        self.embedding = space.build_embedding(sample) 
        self.cfg = space.build_cfg(sample)

    def __eq__(self, other):
        return hash(self) == hash(other)
#        return self.cfg == other.cfg

    def __hash__(self):
        return hash((self.embedding, *self.sample.values()))

    def __repr__(self):
        string = f"SampleNode(space_type={self.space.__class__.__name__}, sample={self.cfg})"
        return string

def sample_monitor(func):
    def inner(self, num_to_sample=1, replace=True):
        if not replace and num_to_sample >= self.size:
            msg = "Required sample number is larger than the space. You should decrease 'num_to_sample' or set replace as True, otherwise, there contains at leaset one ContinuousSpace."
            warnings.warn(msg, RuntimeWarning)

        samples = func(self, num_to_sample, replace)

        _num = num_to_sample - len(samples)
        if getattr(self, '_tmp_iter_in_sample', 0) >= self.MAX_ITER_NUM and _num > 0:
            msg = f"Reach the maximum iteration number to sample but still require {_num} samples. Please check and enlarge your search space or decrease the 'num_to_sample' or increase MAX_ITER_NUM (default: 10000) and retry."
            raise(RuntimeWarning(msg))
        return samples
    return inner

class SearchSpace(object):
    MAX_ITER_NUM = 10000
    def __new__(cls, **kwargs):
        if 'cfg' in kwargs:
            return IIDSpace(**kwargs)
        elif 'candidates' in kwargs:
            return DiscreteSpace(**kwargs)
        else:
            assert 'start' in kwargs and 'end' in kwargs
            return ContinuousSpace(**kwargs)

    @property
    def size(self):
        return getattr(self, '_size', 0)

    def deduplicate(self, samples):
        return list(set(samples))

    def build_embedding(self, *args, **kwargs):
        raise(NotImplementedError("No implementation"))
    def build_cfg(self, *args, **kwargs):
        raise(NotImplementedError("No implementation"))

class IIDSpace(SearchSpace):
    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)
    def __init__(self, cfg, embed_fn=None):
        self.cfg = cfg
        self.spaces = self.extract_search_space(self.cfg)
        self._size = reduce(lambda x,y: x*y, [x.size for x in self.spaces.values()])
        self.embed_fn = embed_fn


    def extract_search_space(self, cfg, prefix="", spaces=None):
        if spaces is None: spaces = {}
        if isinstance(cfg, (DiscreteSpace, ContinuousSpace, IIDSpace)):
            spaces[prefix.rstrip('.')] = cfg
        elif isinstance(cfg, dict):
            for k, v in cfg.items():
                self.extract_search_space(v, prefix+f"{cfg.__class__.__name__}::{k}.", spaces)
        elif isinstance(cfg, (list, tuple)):
            for i, v in enumerate(cfg):
                self.extract_search_space(v, prefix+f"{cfg.__class__.__name__}::{i}.", spaces)
        return spaces

    def build_cfg(self, subsamples):
        def get_item(src, idx, dst=None):
            if isinstance(src, dict): 
                if dst: src[idx] = dst
                return src.get(idx)
            elif isinstance(src, (list, tuple)):
                if dst: src[int(idx)] = dst
                return src[int(idx)]
            else:
                raise(TypeError(f"Index {idx} from {src} is not supported"))

        cfg = deepcopy(self.cfg)
        for prefix, sub_sample in subsamples.items():
            keys = [tmp.split('::')[-1] for tmp in prefix.split('.')]
            tmp = cfg
            for k in keys[:-1]:
                tmp = get_item(tmp, k)
            get_item(tmp, keys[-1], sub_sample.cfg)
        return cfg 

    def build_embedding(self, subsamples):
        if self.embed_fn:
            return self.embed_fn(subsamples)
        else:
            return len(subsamples)
                
    @sample_monitor
    def sample(self, num_to_sample=1, replace=True):
        sample_nodes = []
        self._tmp_iter_in_sample, _num = 0, num_to_sample
        while _num and self._tmp_iter_in_sample < self.MAX_ITER_NUM:
            sub_samples = [{} for _ in range(_num)]
            for prefix, space in self.spaces.items():
                _sub_samples = space.sample(_num, replace=True)
                for i in range(_num):
                    sub_samples[i][prefix] = _sub_samples[i]
            sample_nodes.extend([SampleNode(self, sub_sample) for sub_sample in sub_samples])
            if not replace:
                sample_nodes = self.deduplicate(sample_nodes)
            _num = num_to_sample - len(sample_nodes)
            self._tmp_iter_in_sample += 1

        return sample_nodes


class DiscreteSpace(SearchSpace):
    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)
    def __init__(self, candidates, num_reserve=1, reserve_replace=False, distribution=None, random_seed=None, embed_fn=None):
        self.candidates = self.to_tuple(candidates)
        self.num_reserve = num_reserve
        self.reserve_replace = reserve_replace
        self.rdm = np.random.RandomState(random_seed)
        self.cand_sizes = [cand.size if isinstance(cand, SearchSpace) else 1 for cand in self.candidates]
        self._size = reduce(lambda x,y: x+y, self.cand_sizes)

        self.distribution = [s/self._size for s in self.cand_sizes] if distribution is None else distribution
        self.embed_fn = embed_fn

    def to_tuple(self, candidates):
        for i in range(len(candidates)):
            if isinstance(candidates[i], list):
                candidates[i] = tuple(candidates[i])
        return candidates


    @sample_monitor
    def sample(self, num_to_sample=1, replace=True):
        sample_nodes = []
        self._tmp_iter_in_sample, _num = 0, num_to_sample
        while _num and self._tmp_iter_in_sample < self.MAX_ITER_NUM:
            for i in range(_num):
                sample = {}
                for idx in self.rdm.choice(range(len(self.candidates)), size=self.num_reserve, p=self.distribution, replace=self.reserve_replace):
                    cand = self.candidates[idx] 
                    if isinstance(cand, SearchSpace):
                        cand = cand.sample(num_to_sample=1, replace=True)[0]
                    sample[idx] = cand
                sample_nodes.append(SampleNode(self, sample))
            if not replace:
                sample_nodes = self.deduplicate(sample_nodes)
            _num = num_to_sample - len(sample_nodes)
            self._tmp_iter_in_sample += 1

        return sample_nodes

    def build_cfg(self, sample):
        cfg = []
        for cand_idx, sample in sample.items():
            if isinstance(sample, SampleNode):
                cfg.append(sample.cfg)
            else: cfg.append(sample)
        return cfg[0] if len(cfg)==1 else cfg

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
        
class ContinuousSpace(SearchSpace):
    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, start, end, num_reserve=1, reserve_replace=True, distribution='uniform', random_seed=None, embed_fn=None):
        self.start = start
        self.end = end
        self._size = end - start
        self.num_reserve = num_reserve
        self.reserve_replace = reserve_replace
        self.distribution = distribution
        self.rdm = np.random.RandomState(random_seed)
        self.embed_fn = embed_fn 

    @sample_monitor
    def sample(self, num_to_sample=None, replace=True):
        def _sample_once(num, replace):
            if replace:
                samples = tuple(getattr(self.rdm, self.distribution)(self.start, self.end, num))
                return samples if num > 1 else samples[0]
            else:
                samples = set()
                _num = num
                while _num > 0:
                    samples.union(set(getattr(self.rdm, self.distribution)(self.start, self.end, _num)))
                    if not replace:
                        samples = self.deduplicate(samples)
                    _num = num - len(samples)
                return tuple(samples) if num > 1 else samples[0]

        if replace:
            samples = []
            for _ in range(num_to_sample):
                samples.append(_sample_once(self.num_reserve, self.reserve_replace))
        else:
            samples = set()
            while len(samples) < num_to_sample:
                samples.add(_sample_once(self.num_reserve, self.reserve_replace))
        return [SampleNode(self, {0: sample}) for sample in samples]

    def __len__(self):
        return self.end - self.start

    def build_cfg(self, subsamples):
        cfg = []
        for idx, sample in subsamples.items():
            if isinstance(sample, SampleNode):
                cfg.append(sample.cfg)
            else: cfg.append(sample)
        return cfg[0] if len(cfg)==1 else cfg

    def build_embedding(self, sample):
        if self.embed_fn:
            return self.embed_fn(sample)
        else:
            vals = [v for v in sample.values()]
            return vals[0] if len(vals)==0 else tuple(vals)
