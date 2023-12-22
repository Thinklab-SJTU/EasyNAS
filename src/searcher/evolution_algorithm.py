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

        self.current_epoch = 0
        self.current_survive = []
        super(EvolutionAlgorithm, self).__init__(search_space, self.num_population, num_reward_one_deal)

    def stop_search(self):
        return self.current_epoch >= self.num_epoch

    def natural_selection(self, cands, num_survive):
        return sorted(cands, key=lambda x: x.reward, reverse=True)[:num_survive]

    def _choose(self, cands, num, replace=False):
        idx = np.random.choice(len(cands), size=num, replace=replace)
        return [cands[i] for i in idx]

    def _mutation(self, survive_query, prob_mutation):
        sample = deepcopy(self._choose(survive_query, 1))[0]
        stack = [sample]
        while len(stack) > 0:
            _sample = stack.pop()
            if isinstance(_sample.space, IIDSpace):
                stack.extend(list(_sample.sample.values()))
            else:
                if np.random.random_sample() < prob_mutation:
                    _sample.sample = _sample.space.sample(1)[0].sample
                else:
                    for idx, sub_sample in _sample.sample.items():
                        stack.extend(_sample.get_sampleNode(sub_sample))
        return sample

    def _crossover(self, survive_query):
        father = deepcopy(self._choose(survive_query, 1))[0]
        mother = self._choose(survive_query, 1)[0]
        stack = [(father, mother)]
        while len(stack) > 0:
            _father, _mother = stack.pop()
            if isinstance(_father.space, IIDSpace) and _father.space.label == _mother.space.label:
                stack.extend(list(zip(_father.sample.values(), _mother.sample.values())))
            elif _father.space.label == _mother.space.label:
                if np.random.random_sample() < 0.5:
                    _father.sample = _mother.sample
                else:
                    for idx, sub_sample in _father.sample.items():
                        if idx in _mother.sample:
                            stack.extend(list(zip(_father.get_sampleNode(_father.sample[idx]), _mother.get_sampleNode(_mother.sample[idx]))))
        return father

    def reproduction(self, num_new_children, fn, children=None, **fn_kwargs):
        if children is None: 
            children, num_init = set(), 0
        else:
            num_init = len(children)
        _iter, max_iter = 0, num_new_children*10
        while len(children) < num_init + num_new_children and _iter < max_iter:
            _iter += 1
            cand = fn(**fn_kwargs)
            if cand not in children:
                children.add(cand)
        return list(children)

    def query_next(self):
        self.current_survive += self.history_reward[-1]
        self.current_survive = self.natural_selection(self.current_survive, self.num_survive)
        survive_query = [qr.query for qr in self.current_survive]
        # mutation
        population = self.reproduction(self.num_mutation, self._mutation, survive_query=survive_query, prob_mutation=self.prob_mutation)
        print(f"Mutation... Population has {len(population)} identities")
        # crossover
        population = self.reproduction(self.num_crossover, self._crossover, survive_query=survive_query, children=set(population))
        print(f"Crossover... Population has {len(population)} identities")
        # random search
        population = self.reproduction(self.num_population-self.num_mutation-self.num_crossover, self.search_space._sample_once, children=set(population))
        print(f"Random Select... Population has {len(population)} identities")

        self.current_epoch += 1
        return population

