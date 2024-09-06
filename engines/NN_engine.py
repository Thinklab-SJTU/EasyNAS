from easydict import EasyDict
from typing import Union, List
import bisect
import math
from itertools import chain
import torch
import torch.nn as nn

from builder import create_dataloader, create_model, create_optimizer, create_criterion, create_hook, create_scheduler, create_search_space
from src.hook import HOOK, OptHOOK, hooks_run, hooks_epoch, hooks_train_epoch, hooks_val_epoch, hooks_train_iter, hooks_val_iter
from .base import BaseEngine

class NNEngine(BaseEngine):
    def __init__(self, data, model, criterion=None, optimizer=None, lr_scheduler=None, hooks=tuple(), local_rank=-1, sync_bn=False, amp=False, amp_val=False, root_path=None, eval_names=('val.best', )):

        self.local_rank = local_rank
        self.sync_bn = sync_bn
        self.device = torch.device('cuda', max(local_rank, 0))
        self.search_space = create_search_space(model) # an instance of _Searchspace
        self.default_cfg = {}
        self.build_all(data, model, criterion, optimizer, lr_scheduler, hooks)

        self.amp, self.amp_val = amp, amp_val
        self.scaler = torch.cuda.amp.GradScaler(enabled=True) if amp else None

        self.start_epoch = 0
        self.eval_names = eval_names
        self.info = EasyDict({
            'results': {'train': {'best': 0}, 'val': {'best': 0}},
            'current_iter': 0,
            'current_epoch': 0,
            })

        self.root_path = root_path


    def _build_dataset(self, data):
        if data is None:
            assert hasattr(self, 'dataloaders')
            return 
        # build from cfg
        elif 'dataloader' in data:
            self.default_cfg['data'] = data
            print("Building dataloader")
            _, self.dataloaders = create_dataloader(data)
        else:
            self.dataloaders = data
        self.train_loader, self.val_loader, self.test_loader = self.dataloaders.get('train', None), self.dataloaders.get('val', None), self.dataloaders.get('test', None)

    def _build_model(self, model, input_size=None):
        if model is None:
            assert hasattr(self, 'model')
            return 
        # build from cfg
        elif isinstance(model, dict):
            if hasattr(self, 'model'): 
                delattr(self, 'model')
                delattr(self, 'model_without_ddp')
            self.default_cfg['model'] = model
            print("Building model")
            model = create_model(model, input_size=input_size, local_rank=self.local_rank)
            self.model_without_ddp = model
            if self.local_rank >= 0:
#                # convert BN to SyncBN
                if self.sync_bn:
                    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                self.model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.local_rank], output_device=self.local_rank)
            else:
                self.model = model
#                self.model = model.to(self.device)
        else: 
            assert(isinstance(model, nn.Module))
            self.model_without_ddp = self.model = model
            #TODO: What about local_rank > 0

    def _build_criterion(self, criterion):
        if criterion is None:
            self.criterion = getattr(self, 'criterion', None)
            return 
        # build from cfg
        elif isinstance(criterion, dict):
            self.default_cfg['criterion'] = criterion
            print("Building criterion")
            self.criterion = create_criterion(criterion, local_rank=self.local_rank).to(self.device)
        else: 
            assert(isinstance(criterion, nn.Module))
            self.criterion = criterion

    def _build_optimizer(self, optimizer, model, criterion=None):
        if optimizer is None:
            self.optimizer = getattr(self, 'optimizer', None)
            return 
        # build from cfg
        elif isinstance(optimizer, dict):
            self.default_cfg['optimizer'] = optimizer
            print("Building optimizer")
            self.optimizer = create_optimizer(optimizer, model, criterion)
        else: 
            assert(isinstance(optimizer, torch.optim.Optimizer))
            self.optimizer = optimizer

    def _build_scheduler(self, scheduler, optimizer):
        if scheduler is None:
            self.lr_scheduler = getattr(self, 'lr_scheduler', None)
            return 
        # build from cfg
        elif isinstance(scheduler, dict):
            self.default_cfg['lr_scheduler'] = scheduler
            print("Building lr scheduler")
            scheduler['args']['optimizer'] = optimizer
            self.lr_scheduler = create_scheduler(scheduler)
        else: 
            assert(isinstance(scheduler, torch.optim.LRScheduler))
            self.lr_scheduler = scheduler

    def _build_hooks(self, hooks_cfg):
        if hooks_cfg:
            self.default_cfg['hooks'] = hooks_cfg
            print("Building hooks")
            self._hooks = []
            gen = hooks_cfg.values() if isinstance(hooks_cfg, dict) else iter(hooks_cfg)
            for v in gen:
                print(v)
                if (not v.get('args', {}).get('only_master', False)) or self.local_rank in [-1, 0]:
                    self.register_hook(create_hook(v))
        else: self._hooks = getattr(self, '_hooks', [])

    def build_all(self, data=None, model=None, criterion=None, optimizer=None, lr_scheduler=None, hooks=None):
        # build data
        self._build_dataset(data)
        # build model
        input_size = data.get('input_size', None) if isinstance(data, dict) else None
        self._build_model(model, input_size=input_size)
        # build criterion
        self._build_criterion(criterion)
        # build optimizer
        if model and not optimizer:
            optimizer = self.default_cfg.get('optimizer', None)
        self._build_optimizer(optimizer, self.model_without_ddp, self.criterion)
        # build scheduler
        if optimizer and not lr_scheduler:
            lr_scheduler = self.default_cfg.get('lr_scheduler', None)
        self._build_scheduler(lr_scheduler, self.optimizer)
        # build other hooks
        if not hooks:
            hooks = self.default_cfg.get('hooks', None)
        self._build_hooks(hooks)

    def is_ddp(self):
        return self.local_rank >= 0

    def train_one_epoch(self, train_loader, model, criterion, early_stop=None):
#        if self.amp: model.half()
        for step, (input, target, *bs_args) in enumerate(train_loader):
            if early_stop is not None and step==early_stop: break
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

    def train(self, epochs, max_iter):
        self.info.epochs = epochs
#        self.call_hook('before_run')
        with hooks_run(self._hooks, self):
            for epoch in range(self.start_epoch, epochs):
                self.info.current_epoch = epoch
                with hooks_epoch(self._hooks, self):
                    self.model.train()
                    with hooks_train_epoch(self.hooks, self):
                        self.train_one_epoch(self.train_loader, self.model, self.criterion, early_stop=max_iter)
          
                    if self.local_rank in [-1, 0] or self.val_loader.cfg.get('use_dist', True):
                        self.model.eval()
                        with hooks_val_epoch(self._hooks, self):
                            self.val(self.val_loader, self.model, self.criterion)
                max_iter -= len(self.train_loader)
                if max_iter <=0: break
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

    def run(self, epochs=0, max_iter=0):
        epochs = max(epochs, 0)
        max_iter = max(max_iter, 0)
        if epochs or max_iter:
            if epochs == 0: 
                epochs =  math.ceil(max_iter/len(self.train_loader))
            elif max_iter == 0:
                max_iter = math.ceil(epochs * len(self.train_loader))
            else:
                epochs, max_iter = min(math.ceil(epochs), math.ceil(max_iter/len(self.train_loader))), min(math.ceil(epochs*len(self.train_loader)), max_iter)
            self.train(epochs, max_iter)
        else:
            self.validate()

    def update(self, sample):
        self.build_all(**sample)
        if self.scaler is not None: self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        self.start_epoch = 0
        self.info = EasyDict({
            'results': {'train': {'best': 0}, 'val': {'best': 0}},
            'current_iter': 0,
            'current_epoch': 0,
            })

    def extract_performance(self, eval_names=None):
        super(NNEngine, self).extract_performance(eval_names)
#        return self.info.results.val.best

        
