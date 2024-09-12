#------------------------------------------------------------------------------+
# Update according to https://github.com/nathanrooy/particle-swarm-optimization
#------------------------------------------------------------------------------+

import numpy
import torch

from .base import Searcher

class Particle:
    def __init__(self, space, momentum=0.5, local_acceleration=1, global_acceleration=2):
        self.momentum = momentum
        self.local_acceleration = local_acceleration
        self.global_acceleration = global_acceleration
        self.position = [w for w in space.sampler_weights()]
        self.velocity = [torch.rand(*w.shape)*2-1 for w in self.position]          # particle velocity
#        self.velocity = [np.random.uniform(-1, 1, size=w.shape) for w in self.position]          # particle velocity
        self.reward = None               
        self.best_pos = None          # best position individual

#    def traversal(self, sample):
#        variables = []
#        label_samples = {}
#        stack = [sample]
#        while len(stack) > 0:
#            _sample = stack.pop()
#            if _sample.space.label in label_samples:
#                _sample.sample = _sample.space.sample_from_node(label_samples[_sample.space.label], label_samples).sample
#                continue
#            if isinstance(_sample.space, IIDSpace):
#                stack.extend(list(_sample.sample.values()))
#            else:
#        print("\033[1;91m NOTICE:\033[0m PSO algorithm now only supports ContinousSpace. In the future, we will update Particle:traversal to support more spaces")


    # update new particle velocity
    def _update_velocity(self, pos_best_global):
        w = self.momentum       # constant inertia weight (how much to weigh the previous velocity)
        c1 = self.local_acceleration        # cognative constant
        c2 = self.global_acceleration        # social constant

        for i, sub_pos in enumerate(self.position):
            r1 = torch.rand(*sub_pos.shape)
            r2 = torch.rand(*sub_pos.shape)
#            r1 = np.random.uniform(0,1,size=sub_pos.shape)
#            r2 = np.random.uniform(0,1,size=sub_pos.shape)
            vel_cognitive = c1*r1*(self.best_pos[i] - sub_pos)
            vel_social = c2*r2*(pos_best_global[i] - sub_pos)
            self.velocity[i] = w*self.velocity[i] + vel_cognitive + vel_social

    # update the particle position based on new velocity updates
    def update_position(self, reward, pos_best_global):
        if self.reward is None or reward > self.reward: 
            self.reward = reward
            self.best_pos = deepcopy(self.position)
        self._update_velocity(pos_best_global)
        for i, sub_vel in enumerate(self.velocity):
            self.position[i].add_(sub_vel)
        
################### 
# all sampler should be weighted, even for ContinuousSpace
################### 
class PSO(Searcher):
    def __init__(self, search_space, num_particle, particle=Particle, num_iters=1000, num_reward_one_deal=1):
        super(PSO, self).__init__(search_space, num_particle, num_reward_one_deal=num_reward_one_deal)
        self.num_particle = num_particle
        self.num_iters = num_iters
        self.particle_cls = particle
        self.particls = []
        self.current_iter = 1

    def build_particle(self, space):
        return self.particle_cls(space)

    def query_initial(self):
        self.search_spaces = []
        queries = []
        for i in range(self.num_particle):
            # randomize sampler weights
            space = self.search_space.new_space(label=None)
            self.search_spaces.append(space)
            # sample one query
            q = space.sample(1, replace=False)[0]
            assert not hasattr(q, f'__PSO_INDEX__')
            setattr(q, f'__PSO_INDEX__', i)
            queries.append(q)
            # build particle
            self.particles.append(self.build_particle(space))
        return queries

    def stop_search(self):
        return self.current_iter >= self.num_iters

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
            particle = self.particles[particle_idx]
            particle.update_position(reward, best_pos_global)
            # obtain the next query
            q = self.search_spaces[particle_idx].sample(1, replace=False)[0]
            assert not hasattr(q, f'__PSO_INDEX__')
            setattr(q, f'__PSO_INDEX__', i)
            new_queries.append(q)
        return new_queries
        
