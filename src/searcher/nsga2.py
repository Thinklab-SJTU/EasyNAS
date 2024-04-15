import math
from .evolution_algorithm import EvolutionAlgorithm

class NSGA2(EvolutionAlgorithm):
    def __init__(self, search_space,
            num_epoch,
            num_survive,
            num_crossover,
            num_mutation,
            prob_mutation,
            num_population=None,
            num_reward_one_deal=-1):
        super(NSGA2, self).__init__(search_space, num_epoch, num_survive, num_crossover, num_mutation, prob_mutation, num_population, num_reward_one_deal)

    def domiates(self, cand1, cand2):
        ''' Whether cand1 dominates cand2
        Input:
            cand1[src/util_type:QueryReward]: candidate solution 1
            cand2[src/util_type:QueryReward]: candidate solution 2
        '''
        out_ = True
#        equal_flag = True
        for r1, r2 in zip(cand1.reward, cand2.reward):
            # assume larger is better,
            # and no equal exists
            out_ = out_ and (r1 > r2)
#            equal_flag = equal_flag and (r1 == r2)
        return out_

    #Function to carry out NSGA-II's fast non dominated sort
    def fast_non_dominated_sort(self, candidates):
        '''
        Input:
            candidates: list of candidate solutions, each solution is a dictionary
        Output:
            fronts: list of pareto front (index form of candidates), each front is a list of candidate indices
        '''


        S = [[] for i in range(len(candidates))]
        fronts = [[]]
        n = [0 for i in range(len(candidates))]
        rank = [0 for i in range(len(candidates))]

        for p in range(len(candidates)):
            S[p] = []
            n[p] = 0
            for q in range(len(candidates)):
                if self.domiates(candidates[p], candidates[q]):
                    if q not in S[p]:
                        S[p].append(q)
                elif self.domiates(candidates[q], candidates[p]):
                    n[p] = n[p] + 1

            if n[p] == 0:
                rank[p] = 0
                if p not in fronts[0]:
                    fronts[0].append(p)

        i = 0
        while(fronts[i] != []):
            Q=[]
            for p in fronts[i]:
                for q in S[p]:
                    n[q] =n[q] - 1
                    if( n[q]==0):
                        rank[q]=i+1
                        if q not in Q:
                            Q.append(q)
            i = i+1
            fronts.append(Q)

        del fronts[len(fronts)-1]
        return fronts

    #Function to calculate crowding distance
    def crowding_distance(self, candidates, front):
        '''
            candidates: list of candidate solutions, each solution is a dictionary containing objective results
            front: a list of candidate indices in pareto front ranking K
        '''
        sort_list = []
        for i in front:
            sort_list.append(
                                {
                                  'idx': i,
                                  'distance': 0,
                                  'reward': candidates[i].reward
                                 }
                            )

        for r_i in range(len(candidates[0].reward)):
            sort_list.sort(key=lambda x: x['reward'][r_i], reverse=False)
            sort_list[0]['distance'] = math.inf
            sort_list[-1]['distance'] = math.inf
            normalize = abs(sort_list[-1]['reward'][r_i] - sort_list[0]['reward'][r_i])
            for i in range(1, len(sort_list)-1):
                sort_list[i]['distance'] += (sort_list[i+1]['reward'][r_i] - sort_list[i-1]['reward'][r_i])/(normalize + 1e-6)

        sort_list.sort(key=lambda x: x['distance'], reverse=True)
        return [candidates[x['idx']] for x in sort_list]

    def natural_selection(self, cands, num_survive):
        '''Elitist Non-Dominated Sorting
        '''
        print('select with Elitist Non-Dominated Sorting')
        fronts_idx = self.fast_non_dominated_sort(cands)
        survive = []
        for front in fronts_idx:
            if len(front) + len(survive) <= num_survive:
                survive += [cands[i] for i in front]
            else:
                sort_list = self.crowding_distance(cands, front)
                #survive += sort_list[:num_survive-len(self.keep_top_k[k])]
                survive += sort_list[:num_survive-len(survive)]

        self.pareto_front = [cands[i] for i in fronts_idx[0]]
        self.pareto_front.sort(key=lambda x: x.reward)
        return survive

    def state_dict(self):
        ckpt = super(EvolutionAlgorithm, self).state_dict()
        ckpt['current_epoch'] = self.current_epoch
        ckpt['current_survive'] = self.current_survive
        ckpt['pareto_front'] = self.pareto_front
        return ckpt

    def load_state_dict(self, state_dict):
        super(EvolutionAlgorithm, self).load_state_dict(state_dict)
        for qr in self.pareto_front:
            print(qr.reward)
