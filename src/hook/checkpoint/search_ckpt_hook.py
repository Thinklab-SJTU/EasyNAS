import os
import copy
from typing import Union
import torch
import yaml

from builder import CfgDumper
from ..hook import HOOK, execute_period
from src.util_type import QueryReward

class SearchCkptHOOK(HOOK):
    def __init__(self, priority=0, save_root: Union[None, str]=None, presearch: Union[None, str]=None, only_master=True):
        self.priority = priority
        self.only_master = only_master
        self.save_root = save_root
        self.presearch = presearch
        if self.save_root: 
            os.makedirs(self.save_root, exist_ok=True)

    def get_presearch_reward(self, presearch=None):
        if presearch is None: return None
        if not os.path.exists(self.presearch): 
            raise(ValueError(f"{self.presearch} is not an existed file or a directory."))
        if os.path.isdir(self.presearch):
            #TODO: get cfg and embedding
            raise(NotImplementedError())
        elif os.path.isfile(presearch): 
            presearch = presearch
        print('====== Load search ckpt ======')
        print(f"Loading from {presearch}")
        checkpoint = torch.load(presearch)
        return checkpoint

    def before_run(self, runner):
        """
        load presearch model
        """
        checkpoint = self.get_presearch_reward(presearch=self.presearch)
        if checkpoint is not None:
            runner.searcher.load_state_dict(checkpoint['searcher'])
            runner.info.results = checkpoint['results']

    def get_best(self, query_reward):
        best_query_reward = max(query_reward, key=lambda x: x.reward)
        return best_query_reward

    def after_epoch(self, runner):
        # get best
        current_epoch_reward = runner.searcher.history_reward[-1]
        current_epoch_best = self.get_best(current_epoch_reward)
        #TODO: runner.info is EasyDict, it will decompose namedtuple
        best = runner.info.results.get('best', None)
        if best is None or current_epoch_best.reward > best['reward']:
#            runner.info.results.best = copy.copy(current_epoch_best)
            runner.info.results.best = current_epoch_best._asdict()
            self.save_yaml(
                    data={'query': current_epoch_best.query.config, 'reward': current_epoch_best.reward.to_parsable()}, 
                    name='best.yaml')
#            self.save_yaml(runner.info.results.best['query'].config, name='best.yaml')
#            self.save_ckpt(runner, 'best.pt')
        print("Best: Query", QueryReward(**runner.info.results.best))

        # save reward
        self.save_yaml(data=[{'query': qr.query.config, 'reward': qr.reward.to_parsable()} for qr in current_epoch_reward], name='epoch%d.yaml'%runner.info.get('current_epoch', 0))
        # save results
        self.save_ckpt(runner, 'last.pt')

    def after_run(self, runner):
        self.after_epoch(runner)

    def save_ckpt(self, runner, name=None):
        name = 'ckpt_%d.pt'%runner.info.current_epoch if name is None else name
        ckpt = {
          'results': runner.info.results,
          'searcher': runner.searcher.state_dict(),
               }
        save_path = os.path.join(self.save_root, name)
        torch.save(ckpt, save_path)

    def save_yaml(self, data, name):
        if self.save_root:
            yaml_file = os.path.join(self.save_root, name)
            with open(yaml_file, encoding='utf-8', mode='w') as f:
                try:
                    yaml.dump(data=data, stream=f, allow_unicode=True, Dumper=CfgDumper, default_flow_style=False)
                except Exception as e:
                    raise(e)

