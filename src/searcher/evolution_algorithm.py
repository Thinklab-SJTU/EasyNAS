import os
import sys
import time
import numpy as np
import random
import functools
print = functools.partial(print, flush=True)
sys.setrecursionlimit(10000)

from .base import Searcher

class EvolutionAlgorithm(Searcher):
    def __init__(self, search_space, num_initial,
            num_epoch,
            num_survive,
            num_crossover,
            num_mutation,
            prob_mutation,
            num_reward_one_deal=-1):
        super(EvolutionAlgorithm, self).__init__(search_space, num_initial, num_reward_one_deal)
        self.num_epoch = num_epoch
        self.num_initial = num_initial
        self.num_survive = num_survive
        self.num_crossover = num_crossover
        self.num_mutation = num_mutation
        self.prob_mutation = prob_mutation

    def stop_search(self):
        return self.current_epoch >= len(self.num_epoch)

    def query_initial(self):
        self.current_epoch = 0
        self.current_survive = []
        return self.search_space.sample(self.num_initial, replace=False) 

    def natural_selection(self, cands, num_survive):
        return sorted(cands, key=lambda x: x.reward, reverse=True)[:num_survive]

    def _choose(self, cands, num, replace=False):
        idx = np.random.choice(len(cands), size=num, replace=replace)
        return [cands[i] for i in idx]

    def _mutation(self, survive_query, prob_mutation):
        cand = self._choose(survive_query, 1)

        def random_func():
            cand = list(choice(self.keep_top_k[k]))
            for i in range(self.nr_layer):
                if np.random.random_sample() < m_prob:
                    if self.nr_state == 5:
                        nr_state = self.nr_state if i not in self.downsample_layers else 4
                        cand[i] = np.random.randint(nr_state)
                    elif self.nr_state == 7:
                        if i in self.downsample_layers:
                            cand[i] = np.random.randint(5, 4 + self.nr_state) 
                        else:
                            cand[i] = np.random.randint(4, 4 + self.nr_state)
                    elif self.nr_state == 11:
                        if i in self.downsample_layers:
                            cand[i] = np.random.randint(5, 11) 
                        else:
                            cand[i] = np.random.randint(0, self.nr_state)
                    else:
                        cand[i] = np.random.randint(self.nr_state)
            return tuple(cand)


    def _crossover(self, survive_query):
        cand1 = self._choose(survive_query, 1)
        cand2 = self._choose(survive_query, 1)
        return tuple(choice([i, j]) for i, j in zip(p1, p2))

    def reproduction(self, survive_query, num_children, fn):
        children = set()
        _iter, max_iter = 0, num_children*10
        while len(children) < num_children and _iter < max_iter:
            _iter += 1
            cand = fn(survive_query)
            if cand not in children:
                children.add(cand)
        return list(children)


    def query_next(self):
        self.current_survive += self.history_reward[-1]
        self.current_survive = self.natural_selection(self.current_survive, self.num_survive)
        survive_query = [qr.query for qr in self.current_survive]
        mutation = self.get_mutation(survive_query, self.num_mutation, self.prob_mutation)
        crossover = self.get_crossover(survive_query, self.num_crossover)

        self.current_epoch += 1
        return mutation + crossover

