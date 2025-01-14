import inspect
from functools import partial
import torch

from .particle_swarm_opt import PSO
from .first_order_opt import Particle_GD
from src.hook import ZOOptHOOK, hooks_train_iter

class Particle_ZO(Particle_GD):
    def __init__(self, space, optimizer_cfg, grad_clip=None, accumulate_gradient=1):
        super(Particle_ZO, self).__init__(space, optimizer_cfg, grad_clip, accumulate_gradient)
        assert hasattr(self.optimizer, 'ZO')
        del self.optimizer_hook

    def update_position(self):
        while True:
            rewards = []
            directions = self.optimizer.get_samples()
            for d in directions:
                self.optimizer.update_params(d)
                reward, pos_best_global = yield 
                if self.reward is None or reward > self.reward: 
                    self.reward = reward
                    self.best_pos = deepcopy(self.position)
                self.optimizer.update_params(-d)
                rewards.append(reward)
            self.optimizer._step(rewards)
            reward, pos_best_global = yield 
            if self.reward is None or reward > self.reward: 
                self.reward = reward
                self.best_pos = deepcopy(self.position)


class ZeroOrderOpt(PSO):
    #TODO: separate the direction sample and update
    def __init__(self, search_space, 
            optimizer_cfg, grad_clip=None, accumulate_gradient=1, 
            num_particle=1, particle=Particle_ZO, num_iters=1000, num_reward_one_deal=1):
        super(ZeroOrderOpt, self).__init__(search_space, num_particle, particle, num_iters, num_reward_one_deal)
        self.optimizer_cfg = optimizer_cfg
        self.grad_clip = grad_clip
        self.accumulate_gradient = accumulate_gradient
        self.num_iters *= optimizer_cfg['args']['num_sample_per_step']

    def build_particle(self, space):
        return self.particle_cls(space, self.optimizer_cfg, self.grad_clip, self.accumulate_gradient)

    def query_initial(self):
        queries = super(ZeroOrderOpt, self).query_initial()
        self.update_position_gens = [p.update_position() for p in self.particles]

    def stop_search(self):
        out = self.current_iter >= self.num_iters
        for g in self.update_position_gens:
            g.close()
        return out

    def query_next(self):
        current_qr = self.history_reward[-1]
        # get best, (maximize)
        best_query_global = None
        best_reward_global = None
        for qr in current_qr:
            if (best_reward_global is None or qr.reward > best_reward_global):
                best_query_global, best_reward_global = qr.query, qr.reward 
                best_pos_global = deepcopy(self.particle[best_query_global.__PSO_INDEX__].position) 
        # update particle position
        next_queries = []
        for qr in current_qr:
            query, reward = qr.query, qr.reward
            particle_idx = query.__PSO_INDEX__
            self.update_position_gens[particle_idx].send(reward, best_pos_global)
            # obtain the next query
            q = self.search_spaces[particle_idx].sample(1, replace=False)[0]
            assert not hasattr(q, f'__PSO_INDEX__')
            setattr(q, f'__PSO_INDEX__', i)
            new_queries.append(q)
        return new_queries


#class ZeroOrderOpt(Searcher):
#    def __init__(self, search_space, num_initial, optimizer_cfg, grad_clip=None, accumulate_gradient=1, num_iters=1000, device='cuda',
#           save_root=None, temperature_start=1.,
#           temperature_end=1.,
#            ):
#        super(ZeroOrderOpt, self).__init__(search_space, num_initial, num_reward_one_deal=-1)
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
#            assert getattr(optimizer, 'ZO', False):
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
#                pass
#            next_queries.append(qr.reward[1])
#        return next_queries

