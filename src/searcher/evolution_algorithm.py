import os
import sys
import time
from copy import deepcopy
import numpy as np
import random
import functools
#print = functools.partial(print, flush=True)
#sys.setrecursionlimit(10000)

from .base import Searcher
from src.search_space.base import IIDSpace

class EvolutionAlgorithm(Searcher):
    def __init__(self, search_space,
            num_epoch,
            num_survive,
            num_crossover,
            num_mutation,
            prob_mutation,
            num_population=None,
            num_reward_one_deal=-1):
        self.num_epoch = num_epoch
        self.num_survive = num_survive
        self.num_crossover = num_crossover
        self.num_mutation = num_mutation
        if num_population is None: 
            self.num_population = num_crossover + num_mutation
        else: self.num_population = max(num_population, num_crossover+num_mutation)
        self.prob_mutation = prob_mutation

        self.current_epoch = 1
        self.current_survive = []
        self.seen = set()
        super(EvolutionAlgorithm, self).__init__(search_space, self.num_population, num_reward_one_deal)

    def query_initial(self):
        queries = self.search_space.sample(self.num_initial, replace=False)
        for q in queries:
            self.seen.add(hash(q))
        return queries

    def stop_search(self):
        return self.current_epoch >= self.num_epoch

    def natural_selection(self, cands, num_survive):
        return sorted(cands, key=lambda x: x.reward, reverse=True)[:num_survive]

    def _choose(self, cands, num, replace=False):
        idx = np.random.choice(len(cands), size=num, replace=replace)
        return [cands[i] for i in idx]

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

    def _crossover(self, survive_query):
        label_samples = {}
        father = deepcopy(self._choose(survive_query, 1))[0]
        mother = self._choose(survive_query, 1)[0]
        stack = [(father, mother)]
        while len(stack) > 0:
            _father, _mother = stack.pop()
            if _father.space.label in label_samples:
                _father.sample = _father.space.sample_from_node(label_samples[_father.space.label], label_samples).sample
                continue
            if _father.space.label != _mother.space.label: 
                continue
            if isinstance(_father.space, IIDSpace):
                stack.extend(list(zip(_father.sample.values(), _mother.sample.values())))
            else:
                if np.random.random_sample() < 0.5:
                    _father.sample = _mother.sample
                else:
                    for idx, sub_sample in _father.sample.items():
                        if idx in _mother.sample:
                            stack.extend(list(zip(_father.get_sampleNode(_father.sample[idx]), _mother.get_sampleNode(_mother.sample[idx]))))
            label_samples[_father.space.label] = _father
        return father

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
        self.current_survive += self.history_reward[-1]
        self.current_survive = self.natural_selection(self.current_survive, self.num_survive)
        survive_query = [qr.query for qr in self.current_survive]
        # mutation
        population = self.reproduction(self.num_mutation, self._mutation, survive_query=survive_query, prob_mutation=self.prob_mutation, hash_children=self.seen)
        print(f"Mutation... Population has {len(population)} identities")
        # crossover
        population.extend(self.reproduction(self.num_crossover, self._crossover, survive_query=survive_query, hash_children=self.seen))
        print(f"Crossover... Population has {len(population)} identities")
        # random search
        population.extend(self.reproduction(self.num_population-self.num_mutation-self.num_crossover, self.search_space._sample_once, hash_children=self.seen))
        print(f"Random Select... Population has {len(population)} identities")

        self.current_epoch += 1
        return population

    def state_dict(self):
        ckpt = super(EvolutionAlgorithm, self).state_dict()
        ckpt['current_epoch'] = self.current_epoch
        ckpt['current_survive'] = self.current_survive
        return ckpt

