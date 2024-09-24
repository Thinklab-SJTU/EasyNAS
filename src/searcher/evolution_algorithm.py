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
            init_points=None,
            num_reward_one_deal=-1):
        self.num_epoch = num_epoch
        self.num_survive = num_survive
        self.num_crossover = num_crossover
        self.num_mutation = num_mutation
        if num_population is None: 
            self.num_population = num_crossover + num_mutation
        else: self.num_population = max(num_population, num_crossover+num_mutation)
        self.num_total = num_epoch * self.num_population
        self.prob_mutation = prob_mutation

        self.current_epoch = 1
        self.current_survive = []
        self.seen = set()
        super(EvolutionAlgorithm, self).__init__(search_space, self.num_population, num_reward_one_deal, init_points=init_points)

    def query_initial(self):
#        queries = []
#        num_sample = self.num_initial
#        if self.init_points is not None:
#            sample_nodes = self.search_space.build_nodes(self.init_points)
#            queries.extend(self.search_space.sample_from_nodes(sample_nodes))
#            print("Get default initial queries:")
#            for q in queries:
#                print(q)
#            num_sample = self.num_initial - len(self.init_points)
#        queries.extend(self.search_space.sample(num_sample, replace=False))
        queries = super(EvolutionAlgorithm, self).query_initial()
        for q in queries:
            self.seen.add(hash(q))
        return queries

    def stop_search(self):
        return self.num_total <= 0
#        return self.current_epoch >= self.num_epoch

    def natural_selection(self, cands, num_survive):
        return sorted(cands, key=lambda x: x.reward, reverse=True)[:num_survive]

    def _norm_p(self, p):
        # normalize
        p = (p-p.mean()) / (p.std()+1e-6)
        # softmax
        exp_p = np.exp(p)
        return exp_p / np.sum(exp_p)

    def _choose(self, cands, num, replace=False):
        cand_query = [qr.query for qr in cands]
        cand_p = np.array([qr.reward[0] for qr in cands])
        cand_p = self._norm_p(cand_p)
        idx = np.random.choice(len(cand_query), size=num, replace=replace, p=cand_p)
        return [cand_query[i] for i in idx]

    def _mutation(self, survive, prob_mutation):
        label_samples = {}
        sample = deepcopy(self._choose(survive, 1))[0]
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

    def _crossover(self, survive):
        label_samples = {}
        father = deepcopy(self._choose(survive, 1))[0]
        mother = self._choose(survive, 1)[0]
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
                    _father.sample = deepcopy(_mother.sample)
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
        self.num_total -= len(self.history_reward[-1])
        self.current_survive += self.history_reward[-1]
        self.current_survive = self.natural_selection(self.current_survive, self.num_survive)
        if self.num_reward_one_deal == -1:
            num_mutation, num_crossover, num_random = self.num_mutation, self.num_crossover, self.num_population-self.num_mutation-self.num_crossover
        else:
            num_sample = len(self.history_reward[-1])
            prob_mutation, prob_crossover = self.num_mutation/self.num_population, self.num_crossover/self.num_population
            prob_random = 1 - prob_mutation - prob_crossover
            sample = np.array(np.random.choice([1,2,3], size=num_sample, p=[prob_mutation, prob_crossover, prob_random], replace=True))
            num_mutation, num_crossover, num_random = (sample==1).sum(), (sample==2).sum(), (sample==3).sum()
        # mutation
        population = self.reproduction(num_mutation, self._mutation, survive=self.current_survive, prob_mutation=self.prob_mutation, hash_children=self.seen)
        print(f"Mutation... Population has {len(population)} identities")
        # crossover
        population.extend(self.reproduction(num_crossover, self._crossover, survive=self.current_survive, hash_children=self.seen))
        print(f"Crossover... Population has {len(population)} identities")
        # random search
        population.extend(self.reproduction(num_random, self.search_space._sample_once, hash_children=self.seen))
        print(f"Random Select... Population has {len(population)} identities")

        self.current_epoch += 1
        return population

    def state_dict(self):
        ckpt = super(EvolutionAlgorithm, self).state_dict()
        ckpt['current_epoch'] = self.current_epoch
        ckpt['current_survive'] = self.current_survive
        return ckpt

