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
            return 'waiting' in self.current_queries.values()
        else:
            return (len(self.current_queries) - sum(1 for reward in self.current_queries.values() if reward == 'waiting')) >= num_reward_one_deal

    def query_next(self, queries, rewards):
        raise(NotImplementedError("No Implementation."))

    def preprocess_cfg(self, q):
        if q.cfg.get('root_path', None):
            q.cfg['root_path'] = os.path.join(q.cfg['root_path'], 'hash%d'%(hash(q)))

    def run(self, num_worker_eval, sample_queue: JoinableQueue, reward_queue: JoinableQueue, history_reward):
        init_queries = self.query_initial()
        for q in init_queries:
            self.preprocess_cfg(q)
            sample_queue.put(q)

        while not self.stop_search():
            if self.num_reward_one_deal in [-1, None]:
                num_reward_one_deal = sample_queue.qsize() + reward_queue.qsize()
            else:
                num_reward_one_deal = self.num_reward_one_deal
            if reward_queue.qsize() >= num_reward_one_deal:
                queries, rewards = [], []
                for i in range(reward_queue.qsize()):
                    q, r = reward_queue.get()
#                    reward_queue.task_done()
                    queries.append(q)
                    rewards.append(r)
                history_reward.append([(q, r) for q, r in zip(queries, rewards)])
                next_queries = self.query_next(queries, rewards)
                for q in next_queries:
                    self.preprocess_cfg(q)
                    sample_queue.put(q)
            else:
                time.sleep(10)

        for i in range(num_worker_eval):
            sample_queue.put(None)
#        sample_queue.join()
        num_None = 0
        history_reward.append([])
        while True:
            query_reward = reward_queue.get()
            if query_reward is None:
                num_None += 1
                if num_None == num_worker_eval: break
            query, reward = query_reward
            history_reward[-1].append(query_reward)




