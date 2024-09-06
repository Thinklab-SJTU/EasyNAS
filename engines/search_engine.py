import os
from easydict import EasyDict
import time
import multiprocessing as mp

from .base import BaseEngine
from builder import create_hook, create_search_space, create_searcher, create_contractor
from src.hook import hooks_run, hooks_epoch
#from src.search_space.base import _SearchSpace
from src.util_type import QueryReward


class SearchEngine(BaseEngine):
    def __init__(self, search_space, searcher, contractor, hooks):
        self.search_space, self.searcher, self.contractor= self.build_from_cfg(search_space, searcher, contractor, hooks)

        print(f"The size of Search space is {self.search_space.size}")
#        tmp = []
#        for sample_node in self.search_space.enum_space(recurse=True):
##            print(sample_node.config)
#            tmp.append(sample_node)
#        print(len(tmp), len(set(tmp)), self.search_space.size)
#        assert 0
        
        self.num_eval_workers = self.contractor.num_workers

        self.info = EasyDict({
            'current_epoch': 0,
            'results': EasyDict(),
            })

    def build_from_cfg(self, search_space_cfg, searcher_cfg, contractor_cfg, hooks_cfg):
        # build search_space
        print("Building search space")
#        if isinstance(search_space_cfg, _SearchSpace):
#            search_space = search_space_cfg
#        else:
        search_space = create_search_space(search_space_cfg)

        # build searcher
        print("Building searcher")
        searcher_cfg['args']['search_space'] = search_space
        searcher = create_searcher(searcher_cfg)

        # build evaluater
        print("Building contractor")
        contractor = create_contractor(contractor_cfg)

        # build other hooks
        print("Building hooks")
        self._hooks = []
        gen = hooks_cfg.values() if isinstance(hooks_cfg, dict) else iter(hooks_cfg)
        for v in gen:
            print(v)
            self.register_hook(create_hook(v))
        return search_space, searcher, contractor

    def run(self):
        with hooks_run(self._hooks, self):
#            ctx = multiprocessing.get_context('spawn')
            sample_queue = mp.JoinableQueue()
            reward_queue = mp.JoinableQueue()
            error_queue = mp.JoinableQueue()

            # multiprocessing for contractor
            with self.contractor.build(sample_queue, reward_queue, error_queue) as eval_ps:

                # initialize queries
                if len(self.searcher.current_queries) == 0:
                    print("Initializing...")
                    next_queries = self.searcher.query_initial()
                else:
                    print("Loading queries")
                    next_queries = self.searcher.current_queries.keys()
                for q in next_queries:
                    q = self.searcher.preprocess_cfg(q)
                    sample_queue.put(q)
                    self.searcher.current_queries[q] = 'waiting'
    
                # iterablely search
                print("Searching...")
                while not self.searcher.stop_search():
                    q, r = reward_queue.get()
                    assert q in self.searcher.current_queries
                    self.searcher.current_queries[q] = r
                    if self.searcher.get_enough_rewards():
                        with hooks_epoch(self._hooks, self):
                            self.searcher.history_reward.append([])
                            for q, r in list(self.searcher.current_queries.items()):
                                if r != 'waiting': 
                                    self.searcher.current_queries.pop(q)
                                    self.searcher.history_reward[-1].append(QueryReward(q, r))
                            next_queries = self.searcher.query_next()
    #                        print(len(self.searcher.current_queries), len(self.searcher.history_reward[-1]), len(next_queries))
                            for q in next_queries:
                                self.searcher.preprocess_cfg(q)
                                sample_queue.put(q)
                            self.searcher.current_queries.update({q: 'waiting' for q in next_queries})
                            print(f"Num. of this queries={len(self.searcher.history_reward[-1])}; Num. of next queries={len(self.searcher.current_queries)}")
                            self.info.current_epoch += 1
                            for p in eval_ps:
                                if p.exitcode is not None:
                                    pid, e = error_queue.get(timeout=1)
                                    if e is not None:
                                        print(f"Worker-{pid} got error!")
                                        raise(e)
    
                # get rewards of queries in the last epoch
                last_reward = []
                while len(self.searcher.current_queries)>0:
                    query, reward = reward_queue.get()
                    self.searcher.current_queries.pop(query)
                    last_reward.append(QueryReward(query, reward))
                if len(last_reward) > 0:
                    self.searcher.history_reward.append(last_reward)
                for p in eval_ps:
                    if p.exitcode is not None:
                        pid, e = error_queue.get(timeout=1)
                        if e is not None:
                            print(f"Worker-{pid} got error!")
                            raise(e)

#        for i, rewards in enumerate(self.searcher.history_reward):
#            print("Epoch", i)
#            for q, r in rewards:
#                print(hash(q), r)
#                print(q.config)

