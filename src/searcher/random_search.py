from .base import Searcher

class RandomSearch(Searcher):
    def __init__(self, search_space, num_initial, num_reward_one_deal=-1):
        self.search_space = search_space
        self.num_initial = num_initial
        self.num_reward_one_deal = num_reward_one_deal
        self.history = {}

    def stop_search(self):
        return True

    def query_initial(self):
        return self.search_space.sample(self.num_initial, replace=False)

    def query_next(self, queries, rewards):
        return []
