import inspect
from functools import partial
import torch
from .base import Searcher

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

class FirstOrderOpt(Searcher):
    def __init__(self, search_space, optimizer_cfg, dataloader_name, criterion_cfg=None, grad_clip=None, update_freq=1, accumulate_gradient=1, save_root=None, temperature_start=1.,
           temperature_end=1.,
           replace_settings={}
            ):
        self.search_space = search_space
        self.optimizer_cfg = optimizer_cfg
        self.grad_clip = grad_clip
        self.dataloader_name = dataloader_name
        self.criterion_cfg = criterion_cfg
        self.update_freq = update_freq  
        self.accumulate_gradient = accumulate_gradient
        self.save_root = save_root
        if self.save_root: 
            os.makedirs(self.save_root, exist_ok=True)
        self.temperature_start = temperature_start
        self.temperature_end = temperature_end
        self.replace_settings = replace_settings

    def initialize(self):
        self.search_space.apply_sampler_weights(lambda x: to_device(x, runner.device), recurse=True)

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

        self.scaler = None
#        self.scaler = torch.cuda.amp.GradScaler(enabled=True) if runner.amp else None
        runner.search_space.apply(partial(set_temperature, temp=self.temperature_start))
        self.after_train_epoch(runner)

    def stop_search(self):
        return True

    def query_initial(self):
        return self.search_space

    def query_next(self):
        return []

