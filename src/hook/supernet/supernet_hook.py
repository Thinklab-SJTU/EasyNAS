import os
import inspect
from functools import partial
import torch
import json
import yaml

from builder import get_submodule_by_name, create_criterion, CfgDumper
from ..hook import HOOK, execute_period, only_master, hooks_train_iter
from .. import OptHOOK, ZOOptHOOK
from src.scheduler.utils import one_cycle

#from src.searcher.first_order_opt import set_temperature, to_device

def set_temperature(space, temp):
    if hasattr(space, 'sampler') and hasattr(space.sampler, 'norm_fn'):
        norm_fn = space.sampler.norm_fn
        if isinstance(norm_fn, partial):
            norm_fn = norm_fn.func
        if 'temperature' in inspect.getfullargspec(space.sampler.norm_fn).args:
            space.sampler.norm_fn = partial(norm_fn, temperature=temp)

def to_device(x, device):
    with torch.no_grad():
        return x.to(device).requires_grad_(x.requires_grad)

class DARTSHOOK(HOOK):
    def __init__(self, optimizer_cfg, dataloader_name, criterion_cfg=None, grad_clip=None,  update_freq=1, accumulate_gradient=1, priority=0, save_root=None,
           warmup=10,
           temperature_start=1.,
           temperature_end=1.,
           replace_settings={}
            ):
        self.priority = priority
        self.optimizer_cfg = optimizer_cfg
        self.grad_clip = grad_clip
        self.dataloader_name = dataloader_name
        self.criterion_cfg = criterion_cfg
        self.update_freq = update_freq  
        self.accumulate_gradient = accumulate_gradient
        self.warmup = warmup
        self.save_root = save_root
        self.temperature_start = temperature_start
        self.temperature_end = temperature_end
        self.replace_settings = replace_settings

    def before_run(self, runner):
        runner.search_space.apply_sampler_weights(lambda x: to_device(x, runner.device), recurse=True)

        self.model = runner.model_without_ddp
        self.optimizer_cfg['args']['params'] = runner.search_space.sampler_weights()
        self.optimizer = get_submodule_by_name(self.optimizer_cfg.get('submodule_name'), search_path=('torch.optim',))(**self.optimizer_cfg['args'])
        self.optimizer.zero_grad()
        self.optimizer_hook = OptHOOK(self.optimizer, self.accumulate_gradient, grad_clip=self.grad_clip)
        if self.criterion_cfg is not None:
            self.criterion = create_criterion(self.criterion_cfg).to(runner.device)
        else:
            self.criterion = runner.criterion
        self.dataloader = runner.dataloaders[self.dataloader_name]
        self.dataiter = self.data_generator(self.dataloader)

        runner_root_path = getattr(runner, 'root_path', None)
        if self.save_root and not self.save_root.startswith('/') and runner_root_path: 
            self.save_root = os.path.join(runner_root_path, self.save_root)
        if self.save_root:
            os.makedirs(self.save_root, exist_ok=True)

        self.scaler = None
#        self.scaler = torch.cuda.amp.GradScaler(enabled=True) if runner.amp else None
        runner.search_space.apply(partial(set_temperature, temp=self.temperature_start))
        self.after_train_epoch(runner)

    def after_run(self, runner):
        self.after_train_epoch(runner)

    def data_generator(self, dataloader):
        while True:
            yield from dataloader

    def postprocess_loss(self, loss_items):
        if isinstance(loss_items, (list, tuple)):
            loss, loss_items = loss_items[0], loss_items[1:]
        else:
            loss, loss_items = loss_items, []
        return loss, loss_items

    def backward_arch_param(self, runner):
        arch_param = list(runner.search_space.sampler_weights())
#        try:
#            input_valid, target_valid = self.dataiter.next()
#        except StopIteration:
#            self.dataiter = iter(self.dataloader)
#            input_valid, target_valid = self.dataiter.next()
        input_valid, target_valid, *others = next(self.dataiter)

        target_valid = target_valid.to(runner.device, non_blocking=True)
        input_valid = input_valid.to(runner.device, non_blocking=True)
#        if runner.amp: 
#            input_valid = input_valid.half()
        with torch.cuda.amp.autocast(enabled=runner.amp):
            logits = runner.model(input_valid)
        loss_items = self.criterion(logits, target_valid)
        loss, loss_items = self.postprocess_loss(loss_items)

        if getattr(self, 'scaler', None):
            loss = self.scaler.scale(loss)
        loss.backward(inputs=arch_param)
#        for n, v in runner.search_space.named_sampler_weights():
#            if v.grad is None or torch.isnan(v.grad).any() or torch.isinf(v.grad).any():
#                print(n, v, id(v))

        for v in arch_param:
          if torch.isnan(v.grad).any() or torch.isinf(v.grad).any():
            raise(ValueError("gradient of architecture has NaN..."))

#        grads =  torch.autograd.grad(loss, arch_param, grad_outputs=torch.ones_like(loss), allow_unused=True)
#        for v, g in zip(arch_param, grads):
#          if torch.isnan(g).any() or torch.isinf(g).any():
#            raise(ValueError("gradient of architecture has NaN..."))
#          if g is not None:
#              if v.grad is None:
#                  v.grad = g.data.clone().detach()
#              else:
#                  v.grad.data.add_(g.data.detach())

    @execute_period("update_freq")
    def before_train_iter(self, runner):
        if runner.info.current_epoch < self.warmup:
            return 
        with hooks_train_iter([self.optimizer_hook], self):
            self.backward_arch_param(runner)

    def before_train_epoch(self, runner):
        temp = self.temperature_start - (self.temperature_start-self.temperature_end) * runner.info.current_epoch / (runner.info.epochs-1)
        print(f"Set softmax temperature for arch parameters as {temp}")

        runner.search_space.apply(partial(set_temperature, temp=temp))

    @only_master
    def after_train_epoch(self, runner):
        out_model_yaml = runner.search_space.discretize(**self.replace_settings)
        yaml_file = os.path.join(self.save_root, "architecture_%d.yaml"%runner.info.current_epoch)
        with open(yaml_file, encoding='utf-8', mode='w') as f:
            try:
                yaml.dump(data=out_model_yaml, stream=f, allow_unicode=True, Dumper=CfgDumper, default_flow_style=False)
            except Exception as e:
                raise(e)
        print('='*10+' Show Arch Parameters Begin '+'='*10)
        runner.search_space.show_info()
        print('='*10+' Show Arch Parameters Done '+'='*10)


class ZARTSHOOK(DARTSHOOK):
    def __init__(self, optimizer_cfg, dataloader_name, criterion_cfg=None, grad_clip=None, update_freq=10, accumulate_gradient=1, priority=0, save_root=None,
           warmup=10,
           temperature_start=1.,
           temperature_end=1.,
           replace_settings={},
           train_w_iter=10,
           train_w_optimizer_cfg=None,
           train_w_grad_clip=5.,
           train_w_dataloader_name='train',
           val_w_iter=1,
           ):
        super(ZARTSHOOK, self).__init__(optimizer_cfg, dataloader_name, criterion_cfg, grad_clip, update_freq, accumulate_gradient, priority, save_root,
           warmup, temperature_start, temperature_end, replace_settings)
        self.train_w_optimizer_cfg = train_w_optimizer_cfg
        self.train_w_dataloader_name = train_w_dataloader_name
        self.train_w_iter = train_w_iter
        self.train_w_grad_clip = train_w_grad_clip
        self.val_w_iter = val_w_iter

    def before_run(self, runner):
        runner.search_space.apply_sampler_weights(lambda x: to_device(x, runner.device), recurse=True)

        self.model = runner.model_without_ddp
        self.optimizer_cfg['args']['params'] = runner.search_space.sampler_weights()
        self.optimizer = get_submodule_by_name(self.optimizer_cfg.get('submodule_name'), search_path=('torch.optim',))(**self.optimizer_cfg['args'])
        assert hasattr(self.optimizer, 'ZO')
        self.optimizer.zero_grad()
        self.optimizer_hook = ZOOptHOOK(self.optimizer, self.accumulate_gradient, grad_clip=self.train_w_grad_clip)

        self.opt_sample_norm_decay = partial(one_cycle, start=self.optimizer.sample_norm, end=1e-5, steps=runner.info.epochs-self.warmup)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=runner.info.epochs-self.warmup, eta_min=1e-5)

        if self.criterion_cfg is not None:
            self.criterion = create_criterion(self.criterion_cfg).to(runner.device)
        else:
            self.criterion = runner.criterion
        self.dataloader = runner.dataloaders[self.dataloader_name]
        self.dataiter = self.data_generator(self.dataloader)
        # train_weight
        self.train_w_dataloader = runner.dataloaders[self.train_w_dataloader_name]
        self.train_w_dataiter = self.data_generator(self.train_w_dataloader)
#        if self.train_w_optimizer_cfg is None:
#            self.train_w_optimizer_cfg = {
#                    'submodule_name': 'torch.optim.SGD',
#                    'args': {'lr':0.001, 'momentum':0., 'weight_decay':3e-4}
#                    }
#        self.train_w_optimizer_cfg['args']['params'] = self.model.parameters()
#        self.train_w_optimizer = get_submodule_by_name(self.train_w_optimizer_cfg.get('submodule_name'), search_path=('torch.optim',))(**self.train_w_optimizer_cfg['args'])
#        self.train_w_optimizer.zero_grad()
#        self.train_w_optimizer_hook = OptHOOK(self.train_w_optimizer, self.accumulate_gradient, grad_clip=self.train_w_grad_clip)
        self.train_w_optimizer_hook = OptHOOK(runner.optimizer, self.accumulate_gradient, grad_clip=self.train_w_grad_clip)
        
        runner_root_path = getattr(runner, 'root_path', None)
        if self.save_root and not self.save_root.startswith('/') and runner_root_path: 
            self.save_root = os.path.join(runner_root_path, self.save_root)
        if self.save_root:
            os.makedirs(self.save_root, exist_ok=True)

        self.scaler = None
#        self.scaler = torch.cuda.amp.GradScaler(enabled=True) if runner.amp else None
        runner.search_space.apply(partial(set_temperature, temp=self.temperature_start))
        self.after_train_epoch(runner)

    def before_train_epoch(self, runner):
        if runner.info.current_epoch >= self.warmup:
            self.optimizer.sample_norm = self.opt_sample_norm_decay(runner.info.current_epoch-self.warmup)
            print(f'Sample Norm decay to: {self.optimizer.sample_norm}')
            self.lr_scheduler.step()
            print(f'LR decay to: {self.lr_scheduler.get_lr()}')
        super(ZARTSHOOK, self).before_train_epoch(runner)

    def _closure(self, val_queue, train_queue, model, criterion, optimizer_hook, amp=False):
        # train
        train_loss = 0.
#        ps = [p.clone(memory_format=torch.contiguous_format) for p in model.parameters()]
        state_dict = {k: v.clone().detach() for k, v in model.state_dict().items() if isinstance(v, torch.Tensor)}
        with torch.enable_grad():
          model.train()
          for step, (input, target) in enumerate(train_queue):
            with hooks_train_iter([optimizer_hook], self):
              # update weight
              with torch.cuda.amp.autocast(enabled=amp):
                logits = model(input)
              loss_items = criterion(logits, target)
              loss, loss_items = self.postprocess_loss(loss_items)
              loss.backward()
              train_loss += loss
          train_loss /= (step+1)
        # val
        val_loss = 0.
        model.eval()
        with torch.no_grad():
          for step, (val_input, val_target) in enumerate(val_queue):
            with torch.cuda.amp.autocast(enabled=amp):
              logits = model(val_input)
            loss_items = criterion(logits, val_target)
            loss, loss_items = self.postprocess_loss(loss_items)
            val_loss += loss
          val_loss /= (step+1)
        model.train()

#        for p, pdata in zip(model.parameters(), ps):
#            p.copy_(pdata)
        model.load_state_dict(state_dict)
        return val_loss #train_loss


    def backward_arch_param(self, runner):
        arch_param = list(runner.search_space.sampler_weights())
        val_queue = []
        for _ in range(self.val_w_iter):
            input_valid, target_valid, *others = next(self.dataiter)
            target_valid = target_valid.to(runner.device, non_blocking=True)
            input_valid = input_valid.to(runner.device, non_blocking=True)
            val_queue.append((input_valid, target_valid))
        train_queue = []
        for _ in range(self.train_w_iter):
            input_train, target_train, *others = next(self.train_w_dataiter)
            target_train = target_train.to(runner.device, non_blocking=True)
            input_train = input_train.to(runner.device, non_blocking=True)
            train_queue.append((input_train, target_train))
        self.closure = partial(self._closure, val_queue=val_queue, train_queue=train_queue, model=self.model, criterion=self.criterion, optimizer_hook=self.train_w_optimizer_hook, amp=runner.amp)

