import math
from copy import deepcopy

from .base import Searcher

class Hyperband(Searcher):
    def __init__(self, search_space, num_initial, max_resource_inner_loop, min_resource_per_query=None, reserve_rate=3, with_outer_loop=False, num_reward_one_deal=-1, init_points=None):
        """
        max_resource_inner_loop should be a dict, whose key is the resource name in training cfg, whose value is the maximum resource restriction.
        """
        if min_resource_per_query is None:
            min_resource_per_query = {}
        for name, max_R in max_resource_inner_loop.items():
            min_resource_per_query.setdefault(name, 1)
        self.R = max_resource_inner_loop
        self.r = min_resource_per_query
        self.eta = reserve_rate
        tmp = math.ceil(math.log(min(max(self.R[n]/self.r[n] for n in self.R.keys()), num_initial)) / math.log(self.eta))
        print(f"There are {tmp} inner loops.")
        self.num_inner_loop = list(range(tmp, -1, -1)) if with_outer_loop else [tmp]
        super(Hyperband, self).__init__(search_space, num_initial, num_reward_one_deal, init_points=init_points)

        self.current_outer_loop, self.current_inner_loop = 0, 0
        
    def stop_search(self):
        return self.current_outer_loop >= len(self.num_inner_loop) or len(self.current_queries) <= 0

    def _query_initial(self, n):
        return super(HyperBand, self).query_initial(n)

    def query_initial(self):
        return self.query_next()

    def get_topk(self, cands, k):
        return [deepcopy(cand.query) for cand in sorted(cands, key=lambda x: x.reward, reverse=True)[:k]]

    def query_next(self):
        s = self.num_inner_loop[self.current_outer_loop]
        n = math.floor(self.num_initial / math.pow(self.eta, self.current_outer_loop+self.current_inner_loop) * (self.num_inner_loop[0]+1) / (s+1))
        if n == 0:
            return []
        if self.current_inner_loop == 0:
            next_queries = self._query_initial(n)
        else:
            last_query_reward = self.history_reward[-1]
            next_queries = self.get_topk(last_query_reward, n)

        r = {k: max(int(v*math.pow(self.eta, self.current_inner_loop-s)), self.r[k]) for k, v in self.R.items()}
        for q in next_queries:
            q.replace_setting(r)
#            for k, v in r.items():
#                q.config[k] = v

        self.current_inner_loop = (self.current_inner_loop + 1) % (self.num_inner_loop[self.current_outer_loop]+1)
        if self.current_inner_loop == 0:
            self.current_outer_loop += 1

        return next_queries

    def state_dict(self):
        ckpt = super(Hyperband, self).state_dict()
        ckpt['current_outer_loop'] = self.current_outer_loop
        return ckpt

