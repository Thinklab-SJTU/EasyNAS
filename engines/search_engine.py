from easydict import EasyDict
from collections import OrderedDict
import time

from .base import BaseEngine
from builder import create_hook, create_search_space, create_searcher, create_evaluater

class SearchEngine(BaseEngine):
    def __init__(self, search_space, searcher, evaluater, hooks):
        self.search_space, self.searcher, self.evaluator = self.build_from_cfg(search_space, searcher, evaluater, hooks)
        print(self.search_space.spaces)
        begin = time.time()
        print('space_size', self.search_space.size)
        sample_nodes = self.search_space.sample(1000, replace=False)
        print('time cost:', time.time()-begin)
#        for sample_node in sample_nodes:
#            print(sample_node.cfg, hash(sample_node))
        assert 0
        
        self.start_epoch = 0
        self.info = EasyDict({
            'history': OrderedDict({}), # {query: reward}
            })

    def build_from_cfg(self, search_space_cfg, searcher_cfg, evaluater_cfg, hooks_cfg):
        # build sampler
        print("Building search space")
        search_space = create_search_space(search_space_cfg)

        # build searcher
        print("Building searhcer")
        searcher = None

        # build evaluater
        print("Building evaluater")
        evaluater = None

        # build other hooks
        print("Building hooks")
        self._hooks = []
        gen = hooks_cfg.values() if isinstance(hooks_cfg, dict) else iter(hooks_cfg)
        for v in gen:
            print(v)
            self.register_hook(create_hook(v))
        return search_space, searcher, evaluater

    def run(self, epochs=None):
        self.info.epochs = epochs
        with hooks_run(self._hooks, self):
            next_queries = self.searcher.initial_samples()
            for epoch in range(self.start_epoch, epochs):
                self.info.current_epoch = epoch
                with hooks_epoch(self._hooks, self):
                    rewards = []
                    for query in next_queries:
                        reward = self.evaluater(query)
                        rewards.append(reward)
                        self.info.history[query] = reward
                    next_queries = self.searcher.step(next_queries, rewards)
