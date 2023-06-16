from easydict import EasyDict
from typing import Union, List
import torch

from src.hook import HOOK, OptHOOK

class Trainer(object):
    def __init__(self, dataloaders:dict, model, criterion, optimizer: Union[HOOK, torch.optim.Optimizer], lr_scheduler: HOOK, hooks: List[HOOK]=[], local_rank=-1, sync_bn=False, amp=False):

        self.train_loader, self.val_loader, self.test_loader = dataloaders.get('train', None), dataloaders.get('val', None), dataloaders.get('test', None)
        assert self.train_loader is not None

        self.amp = amp
        self.scaler = torch.cuda.amp.GradScaler(enabled=True) if amp else None

        self.local_rank = local_rank
        self.device = torch.device('cuda', max(local_rank, 0))

        if self.local_rank >= 0:
#            # convert BN to SyncBN
            if sync_bn:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = model.to(self.device)
            self.model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.local_rank], output_device=self.local_rank)
#            model_without_ddp = model.module
        else:
            self.model = model.to(self.device)

        self.criterion = criterion.to(self.device)
        self.start_epoch = 0
        self._hooks = hooks
        self.info = EasyDict({
            'results': {'train': {}, 'val': {}},
            'current_iter': 0,
            'current_epoch': 0,
            })

        if isinstance(optimizer, HOOK):
            self.optimizer_hook = optimizer
        else:
            self.optimizer_hook = OptHOOK(optimizer)
        self.register_hook(self.optimizer_hook)
        if isinstance(lr_scheduler, HOOK):
            self.lr_scheduler_hook = lr_scheduler
        else: 
            self.lr_scheduler_hook = LrScheduleHOOK(lr_scheduler)
        self.register_hook(self.lr_scheduler_hook)

    def is_ddp(self):
        return self.local_rank >= 0

    def register_hook(self, hook: HOOK, priority: int=-1):
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
        if priority < 0:
            assert hasattr(hook, 'priority')
        else:
            hook.priority = priority
#        if hasattr(hook, 'priority'):
#            raise ValueError('"priority" is a reserved attribute for hooks')
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
            if getattr(hook, 'only_master', False) and self.local_rank not in [-1, 0]: continue
            getattr(hook, fn_name)(self)

    def train_one_epoch(self, train_loader, model, criterion):
        self.call_hook('before_train_epoch')
        model.train()
        for step, (input, target) in enumerate(train_loader):
            self.call_hook('before_train_iter')
            self.info.current_iter = step
            target = target.to(self.device, non_blocking=True)
            input = input.to(self.device, non_blocking=True)
            self.info.train_bs_input = input
            self.info.train_bs_target = target
            with torch.cuda.amp.autocast(enabled=self.amp):
                logits = model(input)
                loss = criterion(logits, target)
                self.info.train_bs_logits = logits
                self.info.train_bs_loss = loss
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            self.call_hook('after_train_iter')

        self.call_hook('after_train_epoch')

    def val(self, val_loader, model, criterion):
        self.call_hook('before_val_epoch')
        model.eval()
        with torch.no_grad():
            for step, (input, target) in enumerate(val_loader):
                self.call_hook('before_val_iter')
                self.info.current_iter = step
                target = target.to(self.device, non_blocking=True)
                input = input.to(self.device, non_blocking=True)
            
                with torch.cuda.amp.autocast(enabled=self.amp):
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
#            if self.is_ddp():
#                if self.train_loader.cfg.get('use_dist', True): self.train_loader.sampler.set_epoch(epoch)
#                if self.val_loader.cfg.get('use_dist', True): self.val_loader.sampler.set_epoch(epoch)
            self.info.current_epoch = epoch
            self.call_hook('before_epoch')
            self.train_one_epoch(self.train_loader, self.model, self.criterion)
  
            if self.local_rank in [-1, 0] or self.val_loader.cfg.get('use_dist', True):
                self.val(self.val_loader, self.model, self.criterion)
            self.call_hook('after_epoch')
        self.call_hook('after_run')
        
