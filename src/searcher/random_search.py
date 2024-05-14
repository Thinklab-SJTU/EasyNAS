from .base import Searcher

class RandomSearch(Searcher):
    def stop_search(self):
        return True

    def query_initial(self):
        return self.search_space.sample(self.num_initial, replace=False)

    def query_next(self):
        return []
