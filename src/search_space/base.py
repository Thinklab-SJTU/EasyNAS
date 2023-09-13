from functools import reduce
from collections.abc import Iterable
from copy import deepcopy
import numpy as np

class Space(object):
    def __new__(cls, **kwargs):
        if 'cfg' in kwargs:
            return return SearchSpace(**kwargs)
        else:
            assert 'candidates' in kwargs
            if isinstance(kwargs['candidates'], str):
                candidates = kwargs.pop('candidates').split(':')
                if len(candidates) == 2:
                    kwargs['start'], kwargs['end'] = candidates
                else:
                    assert len(candidates) == 3
                    kwargs['candidates'] = list(range(*candidates))
            if 'candidates' in kwargs:
                return DiscreteSpace(**kwargs)
            else:
                return ContinousSpace(**kwargs)

class SearchSpace(Space):
    def __init__(self, cfg):
        self.cfg = cfg
        self.spaces = self.extract_search_space(self.cfg)
        self.size = reduce(lambda x,y: x*y, [x.size for x in self.spaces])


    def extract_search_space(self, cfg, prefix="", spaces=None):
        spaces = {} if spaces is None
        if isinstance(cfg, (DiscreteSpace, ContinuousSpace)):
            spaces[prefix.lstrip('.')] = cfg
        elif isinstance(cfg, dict):
            for k, v in cfg.items():
                self.extract_search_space(v, prefix+f"{type(cfg)}::{k}.", spaces)
        elif isinstance(cfg, Iterable):
            for i, v in enumerate(cfg):
                self.extract_search_space(v, prefix+f"{type(cfg)}::{i}.", spaces)
        return spaces

    @property
    def size(self):
        return self.size

    def sample(self, num_to_sample=1, replace=False):
        if replace:
            samples = []
            for i in range(num_to_sample):
                samples.append((k, space.sample() for k, space in self.spaces.items()))
        else:
            samples = set()
            while len(samples) < num_to_sample:
                samples.add(tuple((k, space.sample()) for k, space in self.spaces.items()))
        for i in range(num_to_sample):
            cfg = deepcopy(self.cfg)
        return samples, embedding


class DiscreteSpace(Space):
    def __init__(self, candidates, num_reserve=1, reserve_replace=False, distribution=None, random_seed=None):
        self.candidates = candidates
        self.num_reserve = num_reserve
        self.reserve_replace = reserve_replace
        self.rdm = np.random.RandomState(random_seed)
        self.cand_sizes = [cand.size if isinstance(cand, Space) else 1 for cand in self.candidates]
        self.size = reduce(lambda x,y: x+y, self.cand_sizes)

        self.distribution = [s/self.size for s in self.cand_sizes] if distribution is None else distribution

    def sample(self, num_to_sample=1, replace=False):
        if replace:
            sample_idx = []
            for _ in range(num_to_sample):
                sample_idx.append(self.rdm.choice(range(len(self.candidates)), size=self.num_reserve, p=self.distribution, replace=self.reserve_replace))
        else:
            sample_idx = set()
            while len(sample_idx) < num_to_sample:
                sample_idx.add(self.rdm.choice(range(len(self.candidates)), size=self.num_reserve, p=self.distribution, replace=self.reserve_replace))
        samples = []
        for idxes in sample_idx:
            cands = [self.candidates[idx] for idx in idxes]
            for i in range(len(cands)):
                if isinstance(cands[i], Space):
                    cands[i] = cands[i].sample()

            if num_to_sample == 1:
                samples.append(cands[0])
            else:
                samples.append(tuple(cands))

        return tuple(samples)

    def __getitem__(self, idx):
        return self.candidates[idx]
    def __len__(self):
        return len(self.candidates)
    @property
    def size(self):
        return self.size
        
class ContinuousSpace(Space):
    def __init__(self, start, end, num_reserve=1, reserve_replace=True, distribution='uniform', random_seed=None):
        self.start = start
        self.end = end
        self.num_reserve = num_reserve
        self.reserve_replace = reserve_replace
        self.distribution = distribution
        self.rdm = np.random.RandomState(random_seed)

    def sample(self, num_to_sample=None, replace=False):
        def _sample_once(num, replace):
            if replace:
                samples = tuple(getattr(self.rdm, distribute_to_sample)(self.start, self.end, num) for _ in range(num))
                return samples if num > 1 else samples[0]
            else:
                samples = set()
                while len(samples) < num:
                    samples.add(getattr(self.rdm, distribute_to_sample)(self.start, self.end, self.num_reserve))
                return tuple(samples) if num > 1 else samples[0]

        if replace:
            samples = []
            for _ in range(num_to_sample):
                samples.append(_sample_once(self.num_reserve, self.reserve_replace))
        else:
            samples = set()
            while len(samples) < num_to_sample:
                samples.add(_sample_once(self.num_reserve, self.reserve_replace))
        return tuple(samples)

    def __len__(self):
        return self.end - self.start
    @property
    def size(self):
        return self.end - self.start
