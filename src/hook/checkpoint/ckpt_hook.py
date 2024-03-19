import os
from typing import Union
import torch

from ..hook import HOOK, execute_period

class CkptHOOK(HOOK):
    def __init__(self, priority=0, save_root: Union[None, str]=None, pretrain: Union[None, str]=None, load_strict=True, resume: Union[None, str]=None, only_master=True):
        self.priority = priority
        self.only_master = only_master
        self.save_root = save_root
        self.pretrain = pretrain
        self.resume = resume
        self.load_strict = load_strict

    @classmethod
    def get_pretrain_model(cls, device='cpu', pretrain=None):
        if pretrain is None: return None
        if not os.path.exists(pretrain): 
            raise(ValueError(f"{pretrain} is not an existed file or a directory."))
        if os.path.isdir(pretrain):
            pretrain_path = pretrain
            files = os.listdir(pretrain)
            for f in files:
              tmp = f.split('.')
              if tmp[-1] not in ['pt', 'pth']: continue
              if 'last' in tmp[0]:
                pretrain = os.path.join(pretrain_path, f)
                continue
              if 'best' in tmp[0]:
                pretrain = os.path.join(pretrain_path, f)
                break

              tmp = tmp[0].split('_')[-1]
              if not isinstance(tmp, int): 
                  raise(ValueError(f"Please set pretrain as the path of file or name the model as *_[epoch].pt"))
              if int(tmp) > runner.start_epoch: 
                pretrain = os.path.join(pretrain_path, f)
        elif os.path.isfile(pretrain): 
              pretrain = pretrain
        else: raise(ValueError(f"Get unknown type as pretrain. Expect path of file or directory, but get {type(pretrain)}"))

        print('====== Load ckpt ======')
        print(f"Loading from {pretrain}")
        checkpoint = torch.load(pretrain, map_location=device)
        return checkpoint

    def load_ckpt(self, runner, checkpoint, load_epoch=True, load_opt=True, load_scheduler=True, load_hooks=True, load_results=True):
        if checkpoint is not None:
            if runner.is_ddp():
                runner.model.module.load_state_dict(checkpoint['state_dict'], strict=self.load_strict)
            else:
                runner.model.load_state_dict(checkpoint['state_dict'], strict=self.load_strict)
            if load_epoch:
                runner.start_epoch = int(checkpoint['epoch']) + 1
            if load_opt and runner.optimizer is not None:
                try:
                    runner.optimizer.load_state_dict(checkpoint['optimizer'])
                except Exception as e:
                    print("Cannot load optimizer checkpoint")
                    print(e)
            if load_scheduler and runner.lr_scheduler is not None:
                try:
                    runner.lr_scheduler.load_state_dict(checkpoint['scheduler'])
                except Exception as e:
                    print("Cannot load scheduler checkpoint")
                    print(e)
            if load_results:
                runner.info.results = checkpoint['results']
            if load_hooks:
                for hook in runner.hooks:
                    if hook.__class__.__name__ in checkpoint:
                        hook.load_state_dict(checkpoint[hook.__class__.__name__])

    def before_run(self, runner):
        """
        load resume or pretrain model
        """
        runner_root_path = getattr(runner, 'root_path', None)
        if self.save_root and not self.save_root.startswith('/') and runner_root_path: 
            self.save_root = os.path.join(runner_root_path, self.save_root)
        if self.save_root:
            os.makedirs(self.save_root, exist_ok=True)
            setattr(self, 'after_epoch', self.save_model)

        checkpoint = self.get_pretrain_model(device=runner.device, pretrain=self.resume)
        if checkpoint is not None:
            self.load_ckpt(runner, checkpoint, 
                    load_epoch=True, 
                    load_opt=True, 
                    load_scheduler=True,
                    load_hooks=True,
                    load_results=True)
        else:
            checkpoint = self.get_pretrain_model(device=runner.device, pretrain=self.pretrain)
            self.load_ckpt(runner, checkpoint, 
                    load_epoch=False, 
                    load_opt=False, 
                    load_scheduler=False,
                    load_hooks=False,
                    load_results=False)

    def _save_model(self, runner, model_name: Union[None, str]=None):
        ckpt = {
          'epoch': runner.info.current_epoch,
          'architecture': runner.model.arch_list,
          'state_dict': runner.model.state_dict(),
          'results': runner.info.results,
          'optimizer': runner.optimizer.state_dict(),
          'scheduler': runner.lr_scheduler.state_dict(),
               }
        for hook in runner.hooks:
            if hasattr(hook, 'state_dict'):
                ckpt[hook.__class__.__name__] = hook.state_dict(runner)
                
        model_name = 'weight_%d.pt'%runner.info.current_epoch if model_name is None else model_name
        save_path = os.path.join(self.save_root, model_name)
        torch.save(ckpt, save_path)

    def save_model(self, runner):
#        model_name = runner.info['current_epoch']
        self._save_model(runner, 'last.pt')
        if runner.info.results.get('is_best', False):
            self._save_model(runner, 'best.pt')
        



