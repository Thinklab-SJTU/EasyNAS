from copy import deepcopy
import numpy as np
import random

from .base import Searcher
from src.search_space.base import IIDSpace

class LocalSearch(Searcher):
    def __init__(self, 
            search_space, 
            num_initial, 
            num_epoch,
            num_sample_per_epoch=None,
            prob_mutation=0.1,
            num_reward_one_deal=1, 
            init_points=None):
        self.num_sample_per_epoch = num_sample_per_epoch or num_initial
        self.num_epoch = num_epoch
        self.prob_mutation = prob_mutation
        self.seen = set()
        super(LocalSearch, self).__init__(search_space, self.num_population, num_reward_one_deal, init_points=init_points)

    def stop_search(self):
        return self.num_epoch > 0

    def query_initial(self):
        queries = super(RandomSearch, self).query_initial()
        for q in queries:
            self.seen.add(hash(q))
        return queries

    def _mutation(self, survive_query, prob_mutation):
        label_samples = {}
        sample = deepcopy(self._choose(survive_query, 1))[0]
        stack = [sample]
        while len(stack) > 0:
            _sample = stack.pop()
            if _sample.space.label in label_samples:
                _sample.sample = _sample.space.sample_from_node(label_samples[_sample.space.label], label_samples).sample
                continue
            if isinstance(_sample.space, IIDSpace):
                stack.extend(list(_sample.sample.values()))
            else:
                if np.random.random_sample() < prob_mutation:
                    _sample.sample = _sample.space._sample_once(label_samples).sample
                else:
                    for idx, sub_sample in _sample.sample.items():
                        stack.extend(_sample.get_sampleNode(sub_sample))
            label_samples[_sample.space.label] = _sample
        return sample

    def reproduction(self, num_new_children, fn, hash_children=None, **fn_kwargs):
        new_children = []
        if hash_children is None: 
            hash_children = set()
        _iter, max_iter = 0, num_new_children*10
        while len(new_children) < num_new_children and _iter < max_iter:
            _iter += 1
            cand = fn(**fn_kwargs)
            hash_cand = hash(cand)
            if hash_cand not in hash_children:
                hash_children.add(hash_cand)
                new_children.append(cand)
        return new_children 

    def query_next(self):
        self.num_epoch -= 1
        self.current_survive += self.history_reward[-1]
        population = self.reproduction(self.num_sample_per_epoch, self._mutation, survive_query=survive_query, prob_mutation=self.prob_mutation, hash_children=self.seen)
        return population
