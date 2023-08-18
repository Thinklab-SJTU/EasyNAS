from easydict import EasyDict
from typing import Union, List
import bisect
import torch

from builder import create_dataloader, create_model, create_optimizer, create_criterion, create_hook, create_scheduler
from src.hook import HOOK, OptHOOK, hooks_run, hooks_epoch, hooks_train_epoch, hooks_val_epoch, hooks_train_iter, hooks_val_iter

class NNEngine(object):
    def __init__(self, data, model, criterion, optimizer, lr_scheduler, hooks=tuple(), local_rank=-1, sync_bn=False, amp=False, amp_val=False):

        self.local_rank = local_rank
        self.device = torch.device('cuda', max(local_rank, 0))
        self.dataloaders, model, self.criterion, self.optimizer, self.lr_scheduler, hooks = self.build_from_cfg(data, model, criterion, optimizer, lr_scheduler, hooks)

        self.train_loader, self.val_loader, self.test_loader = self.dataloaders.get('train', None), self.dataloaders.get('val', None), self.dataloaders.get('test', None)
        assert self.train_loader is not None

#        for i, (img, target, path, shapes) in enumerate(self.val_loader):
#            print(path)
#            model.train()
#            logits = model(img.to(self.device))
#            logits = [torch.ones_like(l)*0.1 for l in logits]
#            loss = self.criterion(logits, target.to(self.device))
#            print(loss)
#            if i == 0: break
#        assert 0

        self.amp, self.amp_val = amp, amp_val
        self.scaler = torch.cuda.amp.GradScaler(enabled=True) if amp else None

        self.model_without_ddp = model
        if self.local_rank >= 0:
#            # convert BN to SyncBN
            if sync_bn:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            self.model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.local_rank], output_device=self.local_rank)
        else:
            self.model = model
#            self.model = model.to(self.device)

        self.start_epoch = 0
        self._hooks = []
        for hook in hooks: self.register_hook(hook)
        self.info = EasyDict({
            'results': {'train': {}, 'val': {}},
            'current_iter': 0,
            'current_epoch': 0,
            })


    def build_from_cfg(self, data_cfg, model_cfg, criterion_cfg, optimizer_cfg, lr_scheduler_cfg, hooks_cfg):
        # build data
        print("Building dataloader")
        datasets, dataloaders = create_dataloader(data_cfg)

        # build model
        print("Building model")
        model = create_model(model_cfg, input_size=data_cfg.get('input_size', None), local_rank=self.local_rank)

        # build criterion
        print("Building criterion")
        criterion = create_criterion(criterion_cfg, local_rank=self.local_rank).to(self.device)

        # build optimizer
        print("Building optimizer")
        optimizer = create_optimizer(model, optimizer_cfg, criterion)

        # build scheduler
        print("Building lr scheduler")
        lr_scheduler_cfg['args']['optimizer'] = optimizer
        lr_scheduler = create_scheduler(lr_scheduler_cfg)

        # build other hooks
        print("Building hooks")
        hooks = []
        gen = hooks_cfg.values() if isinstance(hooks_cfg, dict) else iter(hooks_cfg)
        for v in gen:
            print(v)
            if (not v.get('args', {}).get('only_master', False)) or self.local_rank in [-1, 0]:
                hooks.append(create_hook(v))
        return dataloaders, model, criterion, optimizer, lr_scheduler, hooks

    @property
    def hooks(self):
        return self._hooks

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
        # insert the hook to a sorted list
        idx = bisect.bisect_right([h.priority for h in self._hooks], hook.priority)
        self._hooks.insert(idx, hook)

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
#        if self.amp: model.half()
        for step, (input, target, *bs_args) in enumerate(train_loader):
#            self.call_hook('before_train_iter')
            with hooks_train_iter(self._hooks, self):
                self.info.current_iter = step
                target = target.to(self.device, non_blocking=True)
                input = input.to(self.device, non_blocking=True)
#                if self.amp: input = input.half()
                self.info.train_bs_input = input
                self.info.train_bs_target = target
                self.info.train_bs_others = bs_args
                with torch.cuda.amp.autocast(enabled=self.amp):
                    logits = model(input)
                    loss_items = criterion(logits, target)
                    if isinstance(loss_items, (list, tuple)):
                        loss, loss_items = loss_items[0], loss_items[1:]
                    else:
                        loss, loss_items = loss_items, []
                    self.info.train_bs_logits = logits
                    self.info.train_bs_loss = loss
                    self.info.train_bs_loss_items = loss_items
                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
#        if self.amp: model.float()

    def val(self, val_loader, model, criterion):
        with torch.no_grad():
            if self.amp_val: model.half()
            for step, (input, target, *bs_args) in enumerate(val_loader):
#                self.call_hook('before_val_iter')
                with hooks_val_iter(self._hooks, self):
                    self.info.current_iter = step
                    target = target.to(self.device, non_blocking=True)
                    input = input.to(self.device, non_blocking=True)
                    if self.amp_val: input = input.half()
            
                    with torch.cuda.amp.autocast(enabled=self.amp_val):
                        logits = model(input)
#                        loss = criterion(logits, target)
                        self.info.val_bs_logits = logits
                        self.info.val_bs_input = input
                        self.info.val_bs_target = target
                        self.info.val_bs_others = bs_args
#                        self.info.val_bs_loss = loss
#                self.call_hook('after_val_iter')
            if self.amp_val: model.float()

    def run(self, epochs=None):
        self.info.epochs = epochs
#        self.call_hook('before_run')
        with hooks_run(self._hooks, self):
            for epoch in range(self.start_epoch, epochs):
                self.info.current_epoch = epoch
                with hooks_epoch(self._hooks, self):
                    self.model.train()
                    with hooks_train_epoch(self.hooks, self):
                        self.train_one_epoch(self.train_loader, self.model, self.criterion)
          
                    if self.local_rank in [-1, 0] or self.val_loader.cfg.get('use_dist', True):
                        self.model.eval()
                        with hooks_val_epoch(self._hooks, self):
                            self.val(self.val_loader, self.model, self.criterion)
#        self.call_hook('after_run')
        
