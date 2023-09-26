import math

from .base import Searcher

class Hyperband(Searcher):
    def __init__(self, search_space, num_initial, max_resource_inner_loop={'epoch': 81}, reserve_rate=3, with_outer_loop=False, num_reward_one_deal=-1):
        """
        max_resource_inner_loop should be a dict, whose key is the resource name in training cfg, whose value is the maximum resource restriction.
        """
        self.R = max_resource_inner_loop
        self.eta = reserve_rate
        tmp = int(math.log(min(max(self.R.values()), num_initial)) / math.log(self.eta))
        self.num_inner_loop = list(range(tmp, -1, -1)) if with_outer_loop else [tmp]
        super(Hyperband, self).__init__(search_space, num_initial, num_reward_one_deal)

        self.current_outer_loop, self.current_inner_loop = 0, 0
        
    def stop_search(self):
        return self.current_outer_loop >= len(self.num_inner_loop) or len(self.current_queries) <= 0

    def _query_initial(self, n):
        return self.search_space.sample(n, replace=False)

    def query_initial(self):
        self.current_inner_loop = (self.current_inner_loop + 1) % (self.num_inner_loop[self.current_outer_loop]+1)
        if self.current_inner_loop == 0:
            self.current_outer_loop += 1

        return self._query_initial(self.num_initial)

    def get_topk(self, cands, k):
        return sorted(cands, key=lambda query: cands[query][0], reverse=True)[:k]

    def query_next(self):
        s = self.num_inner_loop[self.current_outer_loop]
        n = math.floor(self.num_initial / math.pow(self.eta, self.current_outer_loop+self.current_inner_loop) * (self.num_inner_loop[0]+1) / (s+1))
        if self.current_outer_loop > 0 and self.current_inner_loop == 0:
            next_queries = self._query_initial(n)
        else:
            last_query_reward = self.history_reward[-1]
            next_queries = self.get_topk(last_query_reward, n)

        r = {k: int(v*math.pow(self.eta, self.current_inner_loop-s)) for k, v in self.R.items()}
        for q in next_queries:
            for k, v in r.items():
                q.cfg[k] = v

        self.current_inner_loop = (self.current_inner_loop + 1) % (self.num_inner_loop[self.current_outer_loop]+1)
        if self.current_inner_loop == 0:
            self.current_outer_loop += 1

        return next_queries

