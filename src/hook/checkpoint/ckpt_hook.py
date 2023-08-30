import os
from typing import Union
import torch

from ..hook import HOOK, execute_period

class CkptHOOK(HOOK):
    def __init__(self, priority=0, save_root: Union[None, str]=None, pretrain: Union[None, str]=None, only_master=True):
        self.priority = priority
        self.only_master = only_master
        self.save_root = save_root
        self.pretrain = pretrain
        if self.save_root: 
            os.makedirs(self.save_root, exist_ok=True)
            setattr(self, 'after_epoch', self.save_model)

    def get_pretrain_model(self):
        if self.pretrain is None: return None
        if not os.path.exists(self.pretrain): 
            raise(ValueError(f"{self.pretrain} is not an existed file or a directory."))
        if os.path.isdir(self.pretrain):
            files = os.listdir(self.pretrain)
            for f in files:
              tmp = f.split('.')
              if tmp[-1] not in ['pt', 'pth']: continue
              tmp = int(tmp[0].split('_')[-1])
              if not isinstance(tmp, int): 
                  raise(ValueError(f"Please set pretrain as the path of file or name the model as *_[epoch].pt"))
              if tmp > runner.start_epoch: 
                pretrain = os.path.join(self.pretrain, f)
        elif os.path.isfile(self.pretrain): 
              pretrain = self.pretrain
        else: raise(ValueError(f"Get unknown type as pretrain. Expect path of file or directory, but get {type(self.pretrain)}"))

        print('====== Load ckpt ======')
        print(f"Loading from {pretrain}")
        checkpoint = torch.load(pretrain)
        return checkpoint

    def before_run(self, runner):
        """
        load pretrain model
        """
        checkpoint = self.get_pretrain_model()
        if checkpoint is not None:
            if runner.is_ddp():
                runner.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                runner.model.load_state_dict(checkpoint['state_dict'])
            runner.start_epoch = int(checkpoint['epoch']) + 1
            if runner.optimizer is not None:
                runner.optimizer.load_state_dict(checkpoint['optimizer'])
            if runner.lr_scheduler is not None:
                runner.lr_scheduler.load_state_dict(checkpoint['scheduler'])
            runner.info.results = checkpoint['results']
            for hook in runner.hooks:
                if hook.__class__.__name__ in checkpoint:
                    hook.load_state_dict(checkpoint[hook.__class__.__name__])

    def _save_model(self, runner, model_name: Union[None, str]=None):
        ckpt = {
          'epoch': runner.info.current_epoch,
          'state_dict': runner.model.state_dict(),
          'results': runner.info.results,
          'optimizer': runner.optimizer.state_dict(),
          'scheduler': runner.lr_scheduler.state_dict(),
               }
        for hook in runner.hooks:
            if hasattr(hook, 'state_dict'):
                ckpt[hook.__class__.__name__] = hook.state_dict(runner)
                
        model_name = 'weight_%d.pt'%epoch if model_name is None else model_name
        save_path = os.path.join(self.save_root, model_name)
        torch.save(ckpt, save_path)

    def save_model(self, runner):
#        model_name = runner.info['current_epoch']
        self._save_model(runner, 'last.pt')
        if runner.info.get('is_best', False):
            self._save_model(runner, 'best.pt')
        



