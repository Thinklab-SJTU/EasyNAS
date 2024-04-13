import inspect
from functools import partial
import torch

from .particle_swarm_opt import PSO
from src.hook import OptHOOK, hooks_train_iter

class Particle_GD:
    def __init__(self, space, optimizer_cfg, grad_clip=None, accumulate_gradient=1):
        self.position = [w for w in space.sampler_weights()]
        self.optimizer_cfg['args']['params'] = self.position 
        self.optimizer = get_submodule_by_name(self.optimizer_cfg.get('submodule_name'), search_path=('torch.optim',))(**self.optimizer_cfg['args'])
        self.optimizer.zero_grad()
        self.optimizer_hook = OptHOOK(optimizer, self.accumulate_gradient, grad_clip=self.grad_clip)

        self.reward = None               
        self.best_pos = None          # best position individual

    def update_position(self, reward, pos_best_global):
        if self.reward is None or reward > self.reward: 
            self.reward = reward
            self.best_pos = deepcopy(self.position)
        with hooks_train_iter([self.optimizer_hook], self):
            reward[0].backward(inputs=self.position)

class FirstOrderOpt(PSO):
    def __init__(self, search_space, 
            optimizer_cfg, grad_clip=None, accumulate_gradient=1, 
            num_particle=1, particle=Particle_GD, num_iters=1000, num_reward_one_deal=1):
        super(FirstOrderOpt, self).__init__(search_space, num_particle, particle, num_iters, num_reward_one_deal)
        self.optimizer_cfg = optimizer_cfg
        self.grad_clip = grad_clip
        self.accumulate_gradient = accumulate_gradient

    def build_particle(self, space):
        return self.particle_cls(space, self.optimizer_cfg, self.grad_clip, self.accumulate_gradient)


#class FirstOrderOpt(Searcher):
#    def __init__(self, search_space, num_initial, optimizer_cfg, grad_clip=None, accumulate_gradient=1, num_iters=1000, device='cuda',
#           save_root=None, temperature_start=1.,
#           temperature_end=1.,
#            ):
#        super(FirstOrderOpt, self).__init__(search_space, num_initial, num_reward_one_deal=1)
#        self.optimizer_cfg = optimizer_cfg
#        self.grad_clip = grad_clip
#        self.accumulate_gradient = accumulate_gradient
#        self.num_iters = num_iters
#        self.device = torch.device(device)
#        self.save_root = save_root
#        if self.save_root: 
#            os.makedirs(self.save_root, exist_ok=True)
#        self.temperature_start = temperature_start
#        self.temperature_end = temperature_end
#
#    def stop_search(self):
#        return self.current_iter < self.num_iters
#
#    def query_initial(self):
#        self.search_spaces = []
#        self.optimizer_hooks = []
#        for i in range(self.num_initial):
#            space = self.search_space.new_space()
#            space.apply_sampler_weights(lambda x: to_device(x, self.device), recurse=True)
#
#            self.optimizer_cfg['args']['params'] = runner.search_space.sampler_weights()
#            optimizer = get_submodule_by_name(self.optimizer_cfg.get('submodule_name'), search_path=('torch.optim',))(**self.optimizer_cfg['args'])
#            optimizer.zero_grad()
#            optimizer_hook = OptHOOK(optimizer, self.accumulate_gradient, grad_clip=self.grad_clip)
#            space.apply(partial(set_temperature, temp=self.temperature_start))
#            assert not hasattr(space, f'__OrderOpt__')
#            setattr(space, f'__OrderOpt__', i)
#            self.optimizer_hooks.append(optimizer)
#            self.search_spaces.append(space)
#        self.current_iter = 0
#
#        return self.search_spaces
#
#    def query_next(self):
#        """
#        There should be two rewards for each query: [criterion/loss, semantized_search_space (e.g. nn.module based on the search space)]
#        """
#        QRs = self.history_reward[-1]
#        next_queries = []
#        for qr in QRs:
#            idx = qr.query.__OrderOpt__
#            with hooks_train_iter([self.optimizer_hooks[idx]], self):
#                qr.reward[0].backward(inputs=qr.query.sampler_weights())
#            next_queries.append(qr.reward[1])
#        return next_queries
