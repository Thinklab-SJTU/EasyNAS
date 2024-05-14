from src.hook.hook import HOOK
from functools import reduce

scip_result_getter = {
        'sol': 'getBestSol',
        'optimal_val': 'getObjVal',
        'time': 'getSolvingTime',
        'primal_bound': 'getPrimalbound',
        'dual_bound': 'getDualbound',
        'gap': 'getGap',
        }

fn_getter = {
        'sum': sum,
        'geoMean': lambda rewards: reduce(lambda x, y: x*y, rewards)**(1/len(rewards)),
        'mean': lambda rewards: sum(rewards)/len(rewards)
        }

def get_result_by_name(model, name):
    return getattr(model, scip_result_getter[name])()

class EvalHOOK(HOOK):
    def __init__(self, priority=0, eval_names=('neg-sum-gap',), default_gather_fn='sum'):
        self.priority = priority
        self.eval_names = eval_names
        self.default_gather_fn = default_gather_fn

    def _get_fn_name(self, name):
        name = name.split('-')
        if len(name) == 1: 
            return 1 , self.default_gather_fn, name[0]
        elif len(name) == 2:
            return 1, name[0], name[1]
        else:
            return -1, name[1], name[2]

    def after_iter(self, runner):
        computed_name = set()
        for i, eval_name in enumerate(self.eval_names):
            sign, gather_fn, name = self._get_fn_name(eval_name)
#            sign_label = '' if sign == 1 else 'neg-'
            if name not in computed_name:
                runner.info.results.setdefault(name, []).append(get_result_by_name(runner.info.current_model, name))
                computed_name.add(name)
            runner.info.results[eval_name] = fn_getter[gather_fn](runner.info.results[name]) * sign
#            runner.info.results[f'{sign_label}{gather_fn}-{name}'] = fn_getter[gather_fn](runner.info.results[name]) * sign

    def after_run(self, runner):
        for i, name in enumerate(self.eval_names):
            sign, gather_fn, name = self._get_fn_name(name)
            sign_label = '' if sign == 1 else 'neg-'
            runner.info.results.setdefault('best', []).append(runner.info.results[f'{sign_label}{gather_fn}-{name}'])

