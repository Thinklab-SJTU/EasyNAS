import os
import time
from multiprocessing import Process, JoinableQueue
from builder import parse_cfg, get_submodule_by_name

class Searcher(object):
    def __init__(self, search_space, num_initial, num_reward_one_deal=-1):
        self.search_space = search_space
        self.num_initial = num_initial
        self.num_reward_one_deal = num_reward_one_deal
        self.history_reward = []
        self.current_queries = {}

    def query_initial(self):
        queries = self.search_space.sample(self.num_initial, replace=False)
        return queries

    def stop_search(self):
        raise(NotImplementedError("No Implementation."))

    def get_enough_rewards(self):
        if self.num_reward_one_deal in [-1, None]:
            return not ('waiting' in self.current_queries.values())
        else:
            return (len(self.current_queries) - sum(1 for reward in self.current_queries.values() if reward == 'waiting')) >= num_reward_one_deal

    def query_next(self, queries, rewards):
        raise(NotImplementedError("No Implementation."))

    def preprocess_cfg(self, q):
        if q.cfg.get('root_path', None):
            q.cfg['root_path'] = os.path.join(q.cfg['root_path'], 'hash%d'%(hash(q)))




