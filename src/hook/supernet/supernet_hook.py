import os
from functools import partial
import torch
import json
import yaml

from builder import get_submodule_by_name, create_criterion, CfgDumper
from ..hook import HOOK, execute_period, only_master, hooks_train_iter
from .. import OptHOOK

def set_temperature(m, temp):
    if hasattr(m, 'set_temperature'):
        m.set_temperature('all', temp)

class DARTSHOOK(HOOK):
    def __init__(self, optimizer_cfg, dataloader_name, criterion_cfg=None, grad_clip=None,  update_freq=1, accumulate_gradient=1, priority=0, save_root=None, discretize_depth=1., discretize_width=1.,
            temperature_start=1.,
            temperature_end=1.,
            ):
        self.priority = priority
        self.optimizer_cfg = optimizer_cfg
        self.grad_clip = grad_clip
        self.dataloader_name = dataloader_name
        self.criterion_cfg = criterion_cfg
        self.update_freq = update_freq  
        self.accumulate_gradient = accumulate_gradient
        self.save_root = save_root
        if self.save_root: 
            os.makedirs(self.save_root, exist_ok=True)
        self.discretize_depth = discretize_depth
        self.discretize_width = discretize_width
        self.temperature_start = temperature_start
        self.temperature_end = temperature_end

    def before_run(self, runner):
        self.model = runner.model_without_ddp
        self.optimizer_cfg['args']['params'] = runner.model_without_ddp.arch_parameters()
        self.optimizer = get_submodule_by_name(self.optimizer_cfg.get('submodule_name'), search_path=('torch.optim',))(**self.optimizer_cfg['args'])
        self.optimizer.zero_grad()
        self.optimizer_hook = OptHOOK(self.optimizer, self.accumulate_gradient, grad_clip=self.grad_clip)
        if self.criterion_cfg is not None:
            self.criterion = create_criterion(self.criterion_cfg).to(runner.device)
        else:
            self.criterion = runner.criterion
        self.dataloader = runner.dataloaders[self.dataloader_name]
        self.dataiter = self.data_generator(self.dataloader)

        self.scaler = None
#        self.scaler = torch.cuda.amp.GradScaler(enabled=True) if runner.amp else None
        runner.model.apply(partial(set_temperature, temp=self.temperature_start))
        self.after_train_epoch(runner)

    def after_run(self, runner):
        self.after_train_epoch(runner)

    def data_generator(self, dataloader):
        while True:
            yield from dataloader

    def backward_arch_param(self, runner):
        arch_param = list(runner.model_without_ddp.arch_parameters())
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
        if isinstance(loss_items, (list, tuple)):
            loss, loss_items = loss_items[0], loss_items[1:]
        else:
            loss, loss_items = loss_items, []

        if getattr(self, 'scaler', None):
            loss = self.scaler.scale(loss)
        loss.backward(inputs=arch_param)
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
        with hooks_train_iter([self.optimizer_hook], self):
            self.backward_arch_param(runner)

    def before_train_epoch(self, runner):
        temp = self.temperature_start - (self.temperature_start-self.temperature_end) * runner.info.current_epoch / (runner.info.epochs-1)
        print(f"Set softmax temperature for arch parameters as {temp}")

        runner.model.apply(partial(set_temperature, temp=temp))

    @only_master
    def after_train_epoch(self, runner):
#        arch_param = {k:v.data.cpu().numpy().tolist() for k, v in runner.model_without_ddp.named_arch_parameters()}
#        alpha_file = os.path.join(self.save_root, "alpha_%d.json"%runner.info.current_epoch)
#        with open(alpha_file, 'w') as f:
#          json.dump(arch_param, f)
        out_model_yaml = runner.model_without_ddp.discretize(depth_multiple=self.discretize_depth, width_multiple=self.discretize_width)
        yaml_file = os.path.join(self.save_root, "architecture_%d.yaml"%runner.info.current_epoch)
        with open(yaml_file, encoding='utf-8', mode='w') as f:
            try:
                yaml.dump(data=out_model_yaml, stream=f, allow_unicode=True, Dumper=CfgDumper, default_flow_style=False)
            except Exception as e:
                raise(e)
        runner.model_without_ddp.info_arch()

    def state_dict(self, runner):
        return {k: v.detach() for k, v in self.model.named_arch_parameters()}

    def load_state_dict(self, ckpt):
        with torch.no_grad():
            for name, p in self.model.named_arch_parameters():
                p.copy_(ckpt[name])


