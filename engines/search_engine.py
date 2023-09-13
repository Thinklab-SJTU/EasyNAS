from easydict import EasyDict
from collections import OrderedDict

from .base import BaseEngine

class SearchEngine(BaseEngine):
    def __init__(self, sampler, searcher, evaluater, hooks):
        self.sampler, self.searcher, self.evaluator, hooks = self.build_from_cfg(sampler, searcher, evaluater)
        
        self.start_epoch = 0
        self.info = EasyDict({
            'history': OrderedDict({}), # {query: reward}
            })

    def build_from_cfg(self, initialize_cfg, search_cfg, evaluate_cfg, hooks_cfg):
        # build sampler

        # build searcher

        # build evaluater

        # build other hooks
        print("Building hooks")
        self._hooks = []
        gen = hooks_cfg.values() if isinstance(hooks_cfg, dict) else iter(hooks_cfg)
        for v in gen:
            print(v)
            if (not v.get('args', {}).get('only_master', False)) or self.local_rank in [-1, 0]:
            self.register_hook(create_hook(v))

    def run(self, epochs=None)
        self.info.epochs = epochs
        with hooks_run(self._hooks, self):
            next_queries = self.sampler.sample()
            for epoch in range(self.start_epoch, epochs):
                self.info.current_epoch = epoch
                with hooks_epoch(self._hooks, self):
                    rewards = []
                    for query in next_queries:
                        reward = self.evaluater(query)
                        rewards.append(reward)
                        self.info.history[query] = reward
                    next_queries = self.searcher.step(next_queries, rewards)
