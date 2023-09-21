import pyscipopt
from pyscipopt import Model as SCIPModel

class SCIPSolver(object):
    def __init__(self, solver_params):
        self.solver_params = solver_params
        if "limits/memory" not in self.solver_params: self.solver_params["limits/memory"] = 12*1024
        if "limits/time" not in self.solver_params: self.solver_params["limits/time"] = 15*60

        self.model = SCIPModel()
        self.model.setPresolve(pyscipopt.SCIP_PARAMSETTING.OFF)
        self.model.setHeuristics(pyscipopt.SCIP_PARAMSETTING.OFF)
        self.model.disablePropagation()
        self.model.setParams(self.solver_params)

        self.model.hideOutput()


    def solve(self, instance):
        self.model.readProblem(instance)
        self.model.optimize()

        # solution
        sol = model.getBestSol()
        primal = model.getPrimalbound()
        dual = model.getDualbound()
        time = model.getSolvingTime()

        return sol, primal, time
