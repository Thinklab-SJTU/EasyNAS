import os
import torch
import json
import yaml

from builder import get_submodule_by_name, create_criterion, CfgDumper
from ..hook import HOOK, execute_period, only_master
from .. import OptHOOK

class DARTSHOOK(HOOK):
    def __init__(self, optimizer_cfg, dataloader_name, criterion_cfg=None, update_freq=1, accumulate_gradient=1, priority=0, save_root=None):
        self.priority = priority
        self.optimizer_cfg = optimizer_cfg
        self.dataloader_name = dataloader_name
        self.criterion_cfg = criterion_cfg
        self.update_freq = update_freq  
        self.accumulate_gradient = accumulate_gradient
        self.save_root = save_root
        if self.save_root: 
            os.makedirs(self.save_root, exist_ok=True)

#    def _initialize_arch_param(arch_params):
#        for p in arch_params:
#            torch.nn.init.normal_(p, mean=0.0, std=1e-6)

    def before_run(self, runner):
        self.optimizer_cfg['args']['params'] = runner.model_without_ddp.arch_parameters()
#        self._initialize_arch_param(arch_param)
        self.optimizer = get_submodule_by_name(self.optimizer_cfg.get('submodule_name'), search_path=('torch.optim',))(**self.optimizer_cfg['args'])
        self.optimizer_hook = OptHOOK(self.optimizer, self.accumulate_gradient)
        if self.criterion_cfg is not None:
            self.criterion = create_criterion(self.criterion_cfg)
        else:
            self.criterion = runner.criterion
        self.dataloader = runner.dataloaders[self.dataloader_name]
#        self.dataiter = iter(self.dataloader)
        self.dataiter = self.data_generator(self.dataloader)

        self.after_train_epoch(runner)

    def after_run(self, runner):
        self.after_train_epoch(runner)

    def data_generator(self, dataloader):
        while True:
            yield from dataloader

    def backward_arch_param(self, runner):
        arch_param = runner.model_without_ddp.arch_parameters()
#        try:
#            input_valid, target_valid = self.dataiter.next()
#        except StopIteration:
#            self.dataiter = iter(self.dataloader)
#            input_valid, target_valid = self.dataiter.next()
        input_valid, target_valid = next(self.dataiter)

        target_valid = target_valid.to(runner.device, non_blocking=True)
        input_valid = input_valid.to(runner.device, non_blocking=True)
        logits = runner.model(input_valid)
        loss = self.criterion(logits, target_valid)

        grads =  torch.autograd.grad(loss, arch_param, grad_outputs=torch.ones_like(loss), allow_unused=True)
        for v, g in zip(arch_param, grads):
          if torch.isnan(g).any() or torch.isinf(g).any():
            raise(ValueError("gradient of architecture has NaN..."))
          if v.grad is None:
            if not (g is None):
              v.grad = Variable(g.data)
          else:
            if not (g is None):
              v.grad.data.add_(g.data)

#    def before_train_epoch(self, runner):
#        self.dataiter = iter(self.dataloader)

    @execute_period("update_freq")
    def before_train_iter(self, runner):
#        self.tmp = getattr(self, 'tmp', 5)
#        if self.tmp == 1:
#            self.after_train_epoch(runner)
#            assert 0
#        else: self.tmp += 1

        self.optimizer_hook.before_train_iter(runner)
        self.backward_arch_param(runner)
        self.optimizer_hook.after_train_iter(runner)

    @only_master
    def after_train_epoch(self, runner):
        arch_param = {k:v.data.cpu().numpy().tolist() for k, v in runner.model_without_ddp.named_arch_parameters()}
        alpha_file = os.path.join(self.save_root, "alpha_%d.json"%runner.info.current_epoch)
        with open(alpha_file, 'w') as f:
          json.dump(arch_param, f)
        out_model_yaml = runner.model_without_ddp.discretize(depth_multiple=5, width_multiple=2.25)
        yaml_file = os.path.join(self.save_root, "architecture_%d.yaml"%runner.info.current_epoch)
        with open(yaml_file, encoding='utf-8', mode='w') as f:
            try:
                yaml.dump(data=out_model_yaml, stream=f, allow_unicode=True, Dumper=CfgDumper, default_flow_style=False)
            except Exception as e:
                raise(e)
        runner.model_without_ddp.info_arch()

