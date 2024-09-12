import numpy as np

from .base import BaseEngine

class SimulateEngine(BaseEngine):
    def run(self):
        pass
    def update(self, sample):
        pass
    def extract_performance(self):
        return np.random.randn(1)[0]

class SimulationFunctionEngine(BaseEngine):
    def __init__(self, obj):
        # build obj
        print("Building the object of the optimization")
        if isinstance(obj, dict):
            obj = create_module(obj, search_path='src.benchmark.object')
        elif isinstance(obj, str):
            obj = create_module({'submodule_name': 'Benchmark_function', 'args': obj}, search_path='src.benchmark.object')
        assert callable(obj)
        self.obj = obj

        self.info = edict({
            'results': edict({}), 
            })

    def run(self):
        self.info.results.result = self.obj()

    def update(self, sample):
        weight = sample['weight']
        self.obj.update_weight(weight)

    def extract_performance(self):
        return self.info.results.get('result', math.inf)
