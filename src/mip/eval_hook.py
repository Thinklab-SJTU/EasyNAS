from src.hook.hook import HOOK

scip_result_getter = {
        'sol': 'getBestSol',
        'optimal_val': 'getObjVal',
        'time': 'getSolvingTime',
        'primal_bound': 'getPrimalbound',
        'dual_bound': 'getDualbound',
        }

def get_result_by_name(model, name):
    return getattr(model, scip_result_getter[name])()

class SumEvalHOOK(HOOK):
    def __init__(self, priority=0, eval_names=('time',)):
        self.priority = priority
        self.eval_names = eval_names

    def after_iter(self, runner):
        for name in self.eval_names:
            runner.info.results[name] = get_result_by_name(runner.info.current_model, name)
            runner.info.results[f'sum_{name}'] = runner.info.results.get(f'sum_{name}', 0) + runner.info.results[name]

