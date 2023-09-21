import os
from easydict import EasyDict
from collections import OrderedDict
import time
from multiprocessing import Process, JoinableQueue, Manager

from .base import BaseEngine
from builder import create_hook, create_search_space, create_searcher, create_evaluater
from src.hook import hooks_run

class SearchEngine(BaseEngine):
    def __init__(self, search_space, searcher, evaluater, hooks, num_eval_workers=1, root_path=None):
        self.search_space, self.searcher, self.evaluater = self.build_from_cfg(search_space, searcher, evaluater, hooks)
        
        self.num_eval_workers = num_eval_workers
        self.root_path = root_path
        self.root_path = root_path
        if root_path is not None:
            os.makedirs(root_path, exist_ok=True)

        self.info = EasyDict({
            'history_reward': Manager().list(), # {query: reward}
            })

    def build_from_cfg(self, search_space_cfg, searcher_cfg, evaluater_cfg, hooks_cfg):
        # build sampler
        print("Building search space")
        search_space = create_search_space(search_space_cfg)

        # build searcher
        print("Building searcher")
        searcher_cfg['args']['search_space'] = search_space
        searcher = create_searcher(searcher_cfg)

        # build evaluater
        print("Building evaluater")
        evaluater = create_evaluater(evaluater_cfg)

        # build other hooks
        print("Building hooks")
        self._hooks = []
        gen = hooks_cfg.values() if isinstance(hooks_cfg, dict) else iter(hooks_cfg)
        for v in gen:
            print(v)
            self.register_hook(create_hook(v))
        return search_space, searcher, evaluater

    def flatten_history(self, history):
        _history = []
        for qr_list in history:
            _history.extend(qr_list)
        print(_history)
        assert 0
        return _history


    def get_best(self, history):
        best_idx = max(history, key=lambda qr: qr[-1]['performance'])
        return history[best_idx]


    def run(self):
        with hooks_run(self._hooks, self):
            sample_queue = JoinableQueue()
            reward_queue = JoinableQueue()
            search_p = Process(target=self.searcher.run, args=(self.num_eval_workers, sample_queue, reward_queue, self.info.history_reward))
            search_p.start()

            eval_ps = [Process(target=self.evaluater.run, args=(sample_queue, reward_queue)) for _ in range(self.num_eval_workers)]
            for p in eval_ps:
                p.start()

            for p in eval_ps:
                p.join()
            search_p.join()

            history_reward = self.flatten_history(self.info.history_reward)
            for k, v in history_reward:
                print(k, v)
            best_sample, best_value = self.get_best(history_reward)
            print(best_sample, best_value)

