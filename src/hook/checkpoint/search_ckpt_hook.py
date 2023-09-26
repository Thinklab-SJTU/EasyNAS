import os
from typing import Union
import torch
import yaml

from builder import CfgDumper
from ..hook import HOOK, execute_period

class SearchCkptHOOK(HOOK):
    def __init__(self, priority=0, save_root: Union[None, str]=None, presearch: Union[None, str]=None, only_master=True):
        self.priority = priority
        self.only_master = only_master
        self.save_root = save_root
        self.presearch = presearch
        if self.save_root: 
            os.makedirs(self.save_root, exist_ok=True)

    def get_presearch_reward(self):
        if self.presearch is None: return None
        raise(NotImplementedError())
        if not os.path.exists(self.pretrain): 
            raise(ValueError(f"{self.pretrain} is not an existed file or a directory."))
        if os.path.isdir(self.pretrain):
            #TODO: get cfg and embedding
            pass

    def before_run(self, runner):
        """
        load pretrain model
        """
        history_reward = self.get_presearch_reward()
        if history_reward is not None:
            runner.searcher.history_reward = history_reward
            runner.info.results.best = self.get_best(history_reward[-1])

    def get_best(self, query_reward):
        best_query = max(query_reward, key=lambda k: query_reward[k])
        return best_query, query_reward[best_query]

    def after_epoch(self, runner):
        # get best
        current_epoch_reward = runner.searcher.history_reward[-1]
        current_epoch_best = self.get_best(current_epoch_reward)
        best = runner.info.results.get('best', None)
        if best is None or current_epoch_best[-1] > runner.info.results.best[-1]:
            runner.info.results.best = current_epoch_best

        # save reward
        self.save_reward(runner, current_epoch_reward)

    def after_run(self, runner):
        self.after_epoch(runner)

    def save_reward(self, runner, current_epoch_reward):
        if self.save_root:
            yaml_file = os.path.join(self.save_root, 'epoch%d.yaml'%runner.info.get('current_epoch', 0))
            with open(yaml_file, encoding='utf-8', mode='w') as f:
                try:
                    yaml.dump(data=[[k, v] for k, v in current_epoch_reward.items()], stream=f, allow_unicode=True, Dumper=CfgDumper, default_flow_style=False)
                except Exception as e:
                    raise(e)




