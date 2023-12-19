from easydict import EasyDict
from typing import Union, List
import bisect
from itertools import chain
import torch

from builder import create_dataloader, create_model, create_optimizer, create_criterion, create_hook, create_scheduler, create_search_space
from src.hook import HOOK, OptHOOK, hooks_run, hooks_epoch, hooks_train_epoch, hooks_val_epoch, hooks_train_iter, hooks_val_iter
from .base import BaseEngine

class NNEngine(BaseEngine):
    def __init__(self, data, model, criterion=None, optimizer=None, lr_scheduler=None, hooks=tuple(), local_rank=-1, sync_bn=False, amp=False, amp_val=False):

        self.local_rank = local_rank
        self.device = torch.device('cuda', max(local_rank, 0))
        self.search_space = create_search_space(model) # an instance of _Searchspace
        self.dataloaders, model, self.criterion, self.optimizer, self.lr_scheduler, hooks = self.build_from_cfg(data, model, criterion, optimizer, lr_scheduler, hooks)

        self.train_loader, self.val_loader, self.test_loader = self.dataloaders.get('train', None), self.dataloaders.get('val', None), self.dataloaders.get('test', None)
        assert self.train_loader is not None

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
            'results': {'train': {'best': 0}, 'val': {'best': 0}},
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
        if criterion_cfg:
            print("Building criterion")
            criterion = create_criterion(criterion_cfg, local_rank=self.local_rank).to(self.device)
        else: criterion = None

        # build optimizer
        if optimizer_cfg:
            print("Building optimizer")
            optimizer = create_optimizer(model, optimizer_cfg, criterion)
        else: optimizer = None

        # build scheduler
        if lr_scheduler_cfg:
            print("Building lr scheduler")
            lr_scheduler_cfg['args']['optimizer'] = optimizer
            lr_scheduler = create_scheduler(lr_scheduler_cfg)
        else: lr_scheduler = None

        # build other hooks
        print("Building hooks")
        hooks = []
        gen = hooks_cfg.values() if isinstance(hooks_cfg, dict) else iter(hooks_cfg)
        for v in gen:
            print(v)
            if (not v.get('args', {}).get('only_master', False)) or self.local_rank in [-1, 0]:
                hooks.append(create_hook(v))
        return dataloaders, model, criterion, optimizer, lr_scheduler, hooks

    def is_ddp(self):
        return self.local_rank >= 0

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
                    loss = self.scaler.scale(loss)
                params_require_grad = []
                for pg in self.optimizer.param_groups:
                    params_require_grad.extend(pg['params'])
                loss.backward(inputs=params_require_grad)

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

    def train(self, epochs):
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

    def validate(self):
#        import json
#        alpha_file = "runs/coco_EAutoDet-s/arch/alpha_49.json"
#        with open(alpha_file, 'r') as f:
#            arch_param = json.load(f)
#        with torch.no_grad():
#            for n, p in self.model.named_arch_parameters():
#                assert n in arch_param
#                p.copy_(torch.tensor(arch_param[n]))

        with hooks_run(self._hooks, self):
            self.model.eval()
            with hooks_val_epoch(self._hooks, self):
                self.val(self.val_loader, self.model, self.criterion)

    def run(self, epochs=None):
        if epochs is None or epochs <=0:
            self.validate()
        else:
            self.train(epochs)

    def extract_performance(self):
        return self.info.results.val.best
        
