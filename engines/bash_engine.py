import atexit
import time
from easydict import EasyDict
from typing import Union, Optional, List, Tuple
from .base import BaseEngine
import subprocess
import psutil

class BashEngine(BaseEngine):
    def __init__(self, 
            bash_cmd, 
            visible_cuda: Optional[List[int]] = None, 
            eval_names: Union[List[str], Tuple[str], str] = 'last', 
            parse_fn = None,
            **kwargs):

        self.bash_cmd = bash_cmd
        self.visible_cuda = visible_cuda
        self.kwargs = kwargs
        if isinstance(eval_names, str):
            self.eval_names = (eval_names, )
        else:
            self.eval_names = eval_names
        self.info = EasyDict({
            'results': {n: 0 for n in self.eval_names},
            })
        self.parse_fn = parse_fn or self.default_parse_fn
        atexit.register(self.kill_process)
                 
    def kill_process(self):
        process = getattr(self, 'process', None)
        if process:
            print("Killing popen process...")
            process.kill()
            process.wait()
            print("Popen process Killed...")

    def default_parse_fn(self, stdout):
        stdout = stdout.split('\n')
        result_keys = set(self.eval_names)
        results = {}
        if 'last' in result_keys:
            result_keys.remove('last')
            results['last'] = float(stdout[-1])
        for line in stdout[::-1]:
            for n in result_keys:
                if n in line:
                    results[n] = float(line.split(' ')[-1])
            for k in results.keys():
                result_keys.discard(k)
            if len(result_keys) == 0: break
        return results

    def run(self, *args, **kwargs):
        kwargs.update({k:v for k, v in self.kwargs.items() if k not in kwargs})
        bash_cmd = ' '.join([self.bash_cmd] + [f"--{k} {v}" for k, v in kwargs.items()])
        if self.visible_cuda:
            bash_cmd = "CUDA_VISIBLE_DEVICES=" + ','.join([str(i) for i in self.visible_cuda]) + ' ' + bash_cmd
        print(f"Runing bash cmd as: {bash_cmd}")
        self.process = psutil.Popen( # use psutil.Popen instead of subprocess.Popen, so that we can get children processes by psutil
                bash_cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, #subprocess.PIPE, subprocess.STDOUT, 
                shell=True, 
                bufsize=1,
                universal_newlines=True)
        stdout = ""
        while True:
            exit_code = self.process.poll()
            output = self.process.stdout.readline()
            if output:
                output = output.rstrip() #.decode('utf-8') #decode is used to transfer byte to str
                print(output)
                stdout += output.strip() + "\n"
            if not output and exit_code is not None: 
                break
        print("Bash cmd done!")
        results = self.parse_fn(stdout)
        self.info.results.update(results)

    def update(self, sample):
        self.kwargs.update(sample)
        self.info = EasyDict({
            'results': {n: 0 for n in self.eval_names},
            })

        
