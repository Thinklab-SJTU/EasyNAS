from easydict import EasyDict
from typing import Union, List
import torch

from src.hook import HOOK, OptHOOK

class Trainer(object):
    def __init__(self, dataloaders:dict, model, criterion, optimizer: Union[HOOK, torch.optim.Optimizer], lr_scheduler: HOOK, hooks: List[HOOK]=[], rank=-1):

        self.train_loader, self.val_loader, self.test_loader = dataloaders.get('train', None), dataloaders.get('val', None), dataloaders.get('test', None)
        assert self.train_loader is not None

        self.device = torch.device('cuda', max(rank, 0))
        self.rank = rank

        self.model = model.to(self.device)
        self.criterion = criterion
        self.start_epoch = 0
        self._hooks = hooks
        self.info = EasyDict({'results': {'train': {}, 'val': {}}})

        if isinstance(optimizer, HOOK):
            self.optimizer_hook = optimizer
        else:
            self.optimizer_hook = OptHOOK(optimizer)
        self.register_hook(self.optimizer_hook, 0)
        if isinstance(lr_scheduler, HOOK):
            self.lr_scheduler_hook = lr_scheduler
        else: 
            self.lr_scheduler_hook = LrScheduleHOOK(lr_scheduler)
        self.register_hook(self.lr_scheduler_hook, 0)

    def is_dpp(self):
        return self.rank == -1


    def register_hook(self, hook: HOOK, priority: int):
        """Register a hook into the hook list.
        The hook will be inserted into a priority queue, with the specified
        priority (See :class:`Priority` for details of priorities).
        For hooks with the same priority, they will be triggered in the same
        order as they are registered.
        Args:
            hook (:obj:`Hook`): The hook to be registered.
            priority (int or str or :obj:`Priority`): Hook priority.
                Lower value means higher priority.
        """
        assert isinstance(hook, HOOK)
        if hasattr(hook, 'priority'):
            raise ValueError('"priority" is a reserved attribute for hooks')
        hook.priority = priority
        # insert the hook to a sorted list
        inserted = False
        for i in range(len(self._hooks) - 1, -1, -1):
            if priority >= getattr(self._hooks[i], 'priority', len(self._hooks)):
                self._hooks.insert(i + 1, hook)
                inserted = True
                break
        if not inserted:
            self._hooks.insert(0, hook)

    def call_hook(self, fn_name:str):
        """Call all hooks.
        Args:
            fn_name (str): The function name in each hook to be called, such as
                "before_train_epoch".
        """
        for hook in self._hooks:
            getattr(hook, fn_name)(self)

    def train_one_epoch(self, train_loader, model, criterion):
        self.call_hook('before_train_epoch')
        model.train()
        for step, (input, target) in enumerate(train_loader):
            self.call_hook('before_train_iter')
            self.info.iter_step = step
            target = target.to(self.device, non_blocking=True)
            input = input.to(self.device, non_blocking=True)
            logits = model(input)
            loss = criterion(logits, target)
            loss.backward()
            self.info.train_bs_logits = logits
            self.info.train_bs_target = target
            self.info.train_bs_loss = loss
            self.call_hook('after_train_iter')

        self.call_hook('after_train_epoch')

    def val(self, val_loader, model, criterion):
        self.call_hook('before_val_epoch')
        model.eval()
        with torch.no_grad():
            for step, (input, target) in enumerate(val_loader):
                self.call_hook('before_val_iter')
                self.info.iter_step = step
                target = target.to(self.device, non_blocking=True)
                input = input.to(self.device, non_blocking=True)
            
                logits = model(input)
                loss = criterion(logits, target)
                self.info.val_bs_logits = logits
                self.info.val_bs_target = target
                self.info.val_bs_loss = loss
                self.call_hook('after_val_iter')

        self.call_hook('after_val_epoch')

    def run(self, epochs=None):
        self.call_hook('before_run')
        for epoch in range(self.start_epoch, epochs):
            self.call_hook('before_epoch')
            self.info.current_epoch = epoch
            self.train_one_epoch(self.train_loader, self.model, self.criterion)
  
            self.val(self.val_loader, self.model, self.criterion)
            self.call_hook('after_epoch')
        self.call_hook('after_run')
        
