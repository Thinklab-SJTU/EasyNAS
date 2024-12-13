import yaml
import atexit
import time
from easydict import EasyDict
from typing import Union, Optional, List, Tuple
import subprocess
import psutil

from .base import BaseEngine
from builder.yaml_parser import CfgLoader

class BashEngine(BaseEngine):
    def __init__(self, 
            bash_cmd, 
            visible_cuda: Optional[List[int]] = None, 
            save_info_names: Union[List[str], Tuple[str], str] = None, 
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
        if isinstance(save_info_names, str):
            self.save_info_names = (save_info_names, )
        else:
            self.save_info_names = save_info_names
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

    def default_parse_fn(self, stdout, result_keys, val_type=None):
        stdout = stdout.split('\n')
        results = {}
        if 'last' in result_keys:
            result_keys.remove('last')
            if val_type is not None:
                results['last'] = val_type(stdout[-1])
            else:
                results['last'] = yaml.load(stdout[-1], Loader=CfgLoader)
        for line in stdout[::-1]:
            line = line.split(' ', 1)
            for n in result_keys:
                if line[0].startswith(n):
#                if n in line[0]:
                    if val_type is not None:
                        results[n] = float(line[-1])
                    else:
                        results[n] = yaml.load(line[-1], Loader=CfgLoader)
                    break
            for k in results.keys():
                result_keys.discard(k)
            if len(result_keys) == 0: break
        return None if len(results) == 0 else results

    def run(self, *args, **kwargs):
        kwargs.update({k:v for k, v in self.kwargs.items() if k not in kwargs})
        command = [self.bash_cmd]
        for k, v in kwargs.items():
            if isinstance(v, list):
                v = [f'--{k}'] + v
                command.append(' '.join(map(str, v)))
            else:
                command.append(f"--{k} {v}")
#        bash_cmd = ' '.join([self.bash_cmd] + [f"--{k} {v}" for k, v in kwargs.items()])
        bash_cmd = ' '.join(command)
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
        results = self.parse_fn(stdout, result_keys = set(self.eval_names), val_type=float)
        if results is None:
            self.info.results = None
        else:
            self.info.results.update(results)
        
        save_infos = self.parse_fn(stdout, result_keys = set(self.save_info_names))
        if save_infos is not None:
            self.info.save_infos = save_infos

        if exit_code == 0:
            print("Bash cmd done!")
        else:
            print("Bash cmd meet error and exit!")
            raise(ValueError("Bash cmd meet error and exit!"))

    def update(self, sample):
        self.kwargs.update(sample)
        self.info = EasyDict({
            'results': {n: 0 for n in self.eval_names},
            })

        
