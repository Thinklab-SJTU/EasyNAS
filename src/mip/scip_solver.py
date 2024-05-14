import pyscipopt
from pyscipopt import Model as SCIPModel

class SCIPSolver(object):
    def __init__(self, solver_params=None):
        self.solver_params = {} if solver_params is None else solver_params
        self.solver_params.setdefault('limits/memory', 12*1024)
        self.solver_params.setdefault('limits/time', 15*60)
        self.solver_params.setdefault('estimation/restarts/restartpolicy', 'n')

    def preprocess(self, model):
        model.setPresolve(pyscipopt.SCIP_PARAMSETTING.OFF)
        model.setHeuristics(pyscipopt.SCIP_PARAMSETTING.OFF)
        model.disablePropagation()
        model.setParams(self.solver_params)

        model.hideOutput()


    def solve(self, instance):
        if isinstance(instance, str):
            model = SCIPModel()
            model.readProblem(instance)
        self.preprocess(model)
        model.optimize()
        return model

#        # solution
#        result = {
##                'sol': model.getBestSol(),
#                'optimal_val': model.getObjVal(),
#                'time': model.getSolvingTime(),
#                'primal_bound': model.getPrimalbound(),
#                'dual_bound': model.getDualbound(),
#                }
#
#        return result
