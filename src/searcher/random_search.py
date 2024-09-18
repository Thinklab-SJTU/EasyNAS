from .base import Searcher

class RandomSearch(Searcher):
    def stop_search(self):
        return True

    def query_initial(self):
        return super(RandomSearch, self).query_initial()

    def query_next(self):
        return []
