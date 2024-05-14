import numpy as np

from .base import BaseEngine

class SimulateEngine(BaseEngine):
    def run(self):
        pass
    def update(self, sample):
        pass
    def extract_performance(self):
        return np.random.randn(1)[0]
