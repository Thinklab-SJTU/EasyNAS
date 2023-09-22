import os
from easydict import EasyDict
from collections import OrderedDict
import time
from multiprocessing import Process, JoinableQueue, Manager

from .base import BaseEngine
from builder import create_hook, create_search_space, create_searcher, create_evaluater
from src.hook import hooks_run, hooks_epoch

class SearchEngine(BaseEngine):
    def __init__(self, search_space, searcher, evaluater, hooks, num_eval_workers=1):
        self.search_space, self.searcher, self.evaluater = self.build_from_cfg(search_space, searcher, evaluater, hooks)
        
        self.num_eval_workers = num_eval_workers

        self.info = EasyDict({
            'current_epoch': 0,
            'results': EasyDict(),
            })

    def build_from_cfg(self, search_space_cfg, searcher_cfg, evaluater_cfg, hooks_cfg):
        # build sampler
        print("Building search space")
        search_space = create_search_space(search_space_cfg)

        # build searcher
        print("Building searcher")
        searcher_cfg['args']['search_space'] = search_space
        searcher = create_searcher(searcher_cfg)

        # build evaluater
        print("Building evaluater")
        evaluater = create_evaluater(evaluater_cfg)

        # build other hooks
        print("Building hooks")
        self._hooks = []
        gen = hooks_cfg.values() if isinstance(hooks_cfg, dict) else iter(hooks_cfg)
        for v in gen:
            print(v)
            self.register_hook(create_hook(v))
        return search_space, searcher, evaluater

    def run(self):
        with hooks_run(self._hooks, self):
            sample_queue = JoinableQueue()
            reward_queue = JoinableQueue()

            eval_ps = [Process(target=self.evaluater.run, args=(sample_queue, reward_queue)) for _ in range(self.num_eval_workers)]
            for p in eval_ps:
                p.start()

            # initialize queries
            init_queries = self.searcher.query_initial()
            for q in init_queries:
                self.searcher.preprocess_cfg(q)
                sample_queue.put(q)
                self.searcher.current_queries[q] = 'waiting'

            # iterablely search
            while not self.searcher.stop_search():
                q, r = reward_queue.get()
                self.searcher.current_queries[q] = r
                if self.searcher.get_enough_rewards():
                    with hooks_epoch(self._hooks, self):
                        self.searcher.history_reward.append({})
                        for q, r in list(self.searcher.current_queries.items()):
                            if r != 'waiting': 
                                self.searcher.current_queries.pop(q)
                                self.searcher.history_reward[-1][q] = r
                        next_queries = self.searcher.query_next()
                        print(len(self.searcher.current_queries), len(self.searcher.history_reward[-1]), len(next_queries))
                        for q in next_queries:
                            self.searcher.preprocess_cfg(q)
                            sample_queue.put(q)
                        self.searcher.current_queries.update({q: 'waiting' for q in next_queries})
                        self.info.current_epoch += 1

            for i in range(self.num_eval_workers):
                sample_queue.put(None)

            # get rewards of queries in the last epoch
            self.searcher.history_reward.append({})
            while len(self.searcher.current_queries)>0:
                query, reward = reward_queue.get()
                self.searcher.current_queries.pop(query)
                self.searcher.history_reward[-1][query] = reward

            for p in eval_ps:
                p.join()

        for i, rewards in enumerate(self.searcher.history_reward):
            print("Iter", i)
            for q, r in rewards.items():
                print(hash(q), r)

        print(self.info.results.best, hash(self.info.results.best[0]))

