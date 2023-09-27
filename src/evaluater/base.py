from copy import deepcopy
import time
from collections import UserList
from multiprocessing import Process, JoinableQueue, Queue

from builder import parse_cfg, get_submodule_by_name

class Reward(UserList):
    pass


class Evaluater(object):
    def __init__(self, eval_fns, resource=None):
        self.resource = resource
        self.eval_fns = self.get_eval_fn(eval_fns)

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

    def run(self, sample_queue, reward_queue):
        while True:
            task = sample_queue.get()
            if task is None:
#                sample_queue.task_done()
                break
            rewards = Reward()
            for fn in self.eval_fns:
                rewards.append(fn(task))
            reward_queue.put((task, rewards))

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


