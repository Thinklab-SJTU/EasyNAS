import os
import sys
from copy import deepcopy
import time
from collections import UserList
from multiprocessing import Process, JoinableQueue, Queue
import logging
from functools import partial

from builder import parse_cfg, get_submodule_by_name
from engines.base import BaseEngine
from src.search_space.base import SampleNode

class Reward(UserList):
    def to_parsable(self):
        return [float(tmp) for tmp in self]

class Contractor(object):
    def __init__(self, eval_engines, resource=None, log_dir=None, num_workers=1):
        self.num_workers = num_workers
        self.eval_engines = eval_engines
        self.resource = resource
        self.worker_id = {}
        self.log_dir = log_dir
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)

    def create_worker(self, Worker, resource=None, log_dir=None, worker_id=None):
        if worker_id is None: worker_id = len(self.worker_id)
        worker = Worker(self.eval_engines, resource=resource, log_dir=log_dir)
        self.worker_id[worker_id] = worker
        worker._ID = worker_id
        if worker.log_dir is not None:
            worker.config_logger(f'worker-{worker_id}', os.path.join(worker.log_dir, f'worker-{worker_id}'))
            import builtins as __builtin__
            builtin_print = __builtin__.print
            __builtin__.print = worker.logger.info
        return worker 

    def dismiss_worker(self, worker):
        if worker.log_dir is not None:
            __builtin__.print = builtin_print
        del self.worker_id[worker._ID]

    def dispatch(self, sample_queue, reward_queue, worker_id=None):
        evaluater = self.create_worker(Evaluater, log_dir=self.log_dir, worker_id=worker_id)
        while True:
            task = sample_queue.get()
            if task is None:
#                sample_queue.task_done()
                break
            rewards = evaluater.do_one_task(task)
            reward_queue.put((task, rewards))
        self.dismiss_worker(evaluater)

class Evaluater(object):
    def __init__(self, eval_engines, resource=None, log_dir=None):
        self.resource = resource
        self.task_id = 0
        self.eval_engines = self.get_eval_engines(eval_engines)
        self.log_dir = log_dir
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)

    def config_logger(self, logger_name, log_path=None):
        log_format = '[%(asctime)s] [%(name)s] [%(levelname)s]: %(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
        self.logger = logging.getLogger(logger_name)
        if log_path:
            fh = logging.FileHandler(log_path)
            fh.setFormatter(logging.Formatter(log_format))
            self.logger.addHandler(fh)

    def get_eval_engines(self, eval_engines):
        _eval_engines = []

        if not isinstance(eval_engines, (list, tuple)):
            eval_engines = [eval_engines]

        for engine in eval_engines:
            if isinstance(engine, dict) and 'submodule_name' in engine:
                _engine = get_submodule_by_name(engine['submodule_name'], search_path='src.evaluater')(
                              **engine['args'],
                              )
                if engine.get('run_args', False):
                    _engine.run = partial(_engine.run, **engine['run_args'])
                _eval_engines.append(_engine)
            elif isinstance(engine, BaseEngine):
                _eval_engines.append(engine)
        return _eval_engines

    def do_one_task(self, task):
        print('='*20+f"Task-{self.task_id} Begin"+'='*20)
        rewards = Reward()
        for _idx, engine in enumerate(self.eval_engines):
            print(f"Running {_idx}-th evaluation engine as {engine}")
            if isinstance(task, SampleNode):
                task = deepcopy(task.config)
            engine.update(task)
            engine.run()
            reward = engine.extract_performance() #engine.info.results.val.best
            if isinstance(reward, (list, tuple)): rewards.extend(list(reward))
            else: rewards.append(reward)
        print(f'Get reward = {rewards}')
        print('='*20+f"Task-{self.task_id} End"+'='*20)
        self.task_id += 1
        return rewards

class Evaluater_ori(object):
    def __init__(self, eval_fns, resource=None, log_dir=None):
        self.resource = resource
        self.eval_fns = self.get_eval_fn(eval_fns)
        self.log_dir = log_dir
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)

    def config_logger(self, logger_name, log_path=None):
        log_format = '[%(asctime)s] [%(name)s] [%(levelname)s]: %(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
        self.logger = logging.getLogger(logger_name)
        if log_path:
            fh = logging.FileHandler(log_path)
            fh.setFormatter(logging.Formatter(log_format))
            self.logger.addHandler(fh)

    def get_eval_fn(self, eval_fn):
        eval_fns = []
        if isinstance(eval_fn, (list, tuple)):
            for fn in eval_fn:
                if isinstance(fn, dict):
                    fn = get_submodule_by_name(fn['submodule_name'], search_path='src.evaluater')(
                                  **fn['args'],
                                  )
                assert callable(fn)
                eval_fns.append(fn)
        elif isinstance(eval_fn, dict) and 'submodule_name' in eval_fn:
             eval_fns.append(get_submodule_by_name(eval_fn['submodule_name'], search_path='src.evaluater')(
                           **eval_fn['args'],
                           ))
        elif callable(eval_fn):
            eval_fns.append(eval_fn)
        return eval_fns

    def run(self, sample_queue, reward_queue, worker_id=None):
        if self.log_dir is not None and worker_id is not None:
            self.config_logger(f'EVAL_WORKER#{worker_id}', os.path.join(self.log_dir, f'worker-{worker_id}'))
            import builtins as __builtin__
            builtin_print = __builtin__.print
            __builtin__.print = self.logger.info
        self.task_id = -1
        while True:
            task = sample_queue.get()
            if task is None:
#                sample_queue.task_done()
                break
            self.task_id += 1
            print('='*20+f"Task-{self.task_id} Begin"+'='*20)
            rewards = Reward()
            for fn_idx, fn in enumerate(self.eval_fns):
                print(f"Get {fn_idx}-th evaluation fn as {fn}")
                reward = fn(deepcopy(task))
                if isinstance(reward, (list, tuple)): rewards.extend(list(reward))
                else: rewards.append(reward)
            print(f'Get reward = {rewards}')
            print('='*20+f"Task-{self.task_id} End"+'='*20)
            reward_queue.put((task, rewards))
        if self.log_dir is not None and worker_id is not None:
            __builtin__.print = builtin_print

#    def run(self, sample_queue: JoinableQueue, reward_queue: JoinableQueue):
#        while True:
#            if len(self._children_p) < self.num_worker and not sample_queue.empty():
#                task = sample_queue.get()
#                if task is None: 
#                    sample_queue.task_done()
#                    break
#
#                if self.root_path:
#                    task.cfg['root_path'] = os.path.join(self.root_path, 'task%d'%(self._num_evaluated))
#
#                self._children_p.append(Process(target=self.eval_one, args=(task, sample_queue, reward_queue)))
#                self._children_p[-1].start()
#                self._num_evaluated += 1
#            else:
#                for p_idx in range(len(self._children_p)):
#                    self._children_p[p_idx].join(1)
#                    if not self._children_p[p_idx].is_alive():
#                        self._children_p.pop(p_idx)
#                        break
#        for p in self._children_p:
#            p.join()
##        reward_queue.join()
#
#
#    def eval_one(self, task, sample_queue: JoinableQueue, reward_queue: JoinableQueue):
#        rewards = {}
#        for key, fn in self.eval_fns.items():
#            rewards[key] = fn(task)
#        reward_queue.put((task, rewards))
#        print(hash(task), sample_queue.qsize(), rewards)
##        sample_queue.task_done()


