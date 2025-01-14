import sys, io
import torch
import torch.nn as nn
import pycutest

class PyCUTEst_F(torch.autograd.Function):
    """
    utilize PyCUTEst to return the gradient
    """
    @staticmethod
    def forward(self, input, problem):
        self.problem = problem
        self.save_for_backward(input)
        return torch.tensor(self.problem.obj(input.detach().numpy()))

    @staticmethod
    def backward(self, grad_output):
        input, = self.saved_tensors
        obj, grad_input = self.problem.obj(input.detach().numpy(), gradient=True)
        return torch.tensor(grad_input), None

class PyCUTEst_func(nn.Module):
    def __init__(self, problem_name, sifParams=None):
        super(PyCUTEst_func, self).__init__()
        self.problem = pycutest.import_problem(problem_name, sifParams=sifParams)
        self.weight = nn.Parameter(torch.from_numpy(self.problem.x0))

    def update_weight(weight):
        if isinstance(weight, np.array):
            weight = torch.from_numpy(weight, device=self.weight.device, dtype=self.weight.dtype)
        self.weight.data.copy_(weight)

    def forward(self):
        return PyCUTEst_F().apply(self.weight, self.problem)

    def display_info(self):
        print(f"PyCUTEst problem name: {self.problem.name}")
        print(f"No. parameters: {self.problem.n}; No. constraints: {self.problem.m}")


#TODO
def parse_availabel_sif_params(message):
    all_sifParams = []
    sifParams = {}
    comments = {}
    for line in message.split('\n'):
        if '=' in line:
            vals = line.split()
            var_name = vals[0]
            value = vals[2]
            dtype = vals[3].split(',')
            if len(dtype) > 1:
                dtype, comment = dtype[0], dtype[1]
            else: comment = None
            dtype = dtype.strip('(').strip(')')
            comment.strip(')').strip()
            if dtype == 'int':
                value = int(value)
            elif dtype == 'float':
                value = float(value)
            sifParams.setdefault(var_name, []).append(value)
            comments.setdefault(var_name, []).append(comment)
#    all_sifParams = 
    return all_sifParams


def PyCUTEst_benchmark(fn_names=None, sifParams=None, objective=None, constraints=None, regular=None, degree=None, origin=None, internal=None, n=None, userN=None, m=None, userM=None, 
    with_sifParams=False,
    num_problem=None):
    if fn_names is None:
        problems = pycutest.find_problems(
            objective=objective, constraints=constraints,
            regular=regular, degree=degree,
            origin=origin, internal=internal,
            n=n, userN=userN,
            m=m, userM=userM
            )
        if num_problem is not None:
            problems = sorted(problems)[:num_problem]

        #TODO
        if with_sifParams:
            assert 0
            new_problems = []
            old_stdout = sys.stdout
            for p in problems:
                sys.stdout = mystdout = io.StringIO()
                pycutest.print_available_sif_params(p)
                message = mystdout.getvalue()
                all_sifParams = parse_availabel_sif_params(message)
            sys.stdout = old_stdout
        else:
            sifParams = [None for _ in range(len(problems))]
    else:
        problems = fn_names
        if sifParams is None:
            sifParams = [None for _ in range(len(problems))]
        sifParams = sifParams

    fns = []
    for idx, p in enumerate(problems):
        fn = PyCUTEst_func(p, sifParams[idx])
        fns.append(fn)
    return fns

