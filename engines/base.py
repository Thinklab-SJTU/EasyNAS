import bisect

from src.hook.hook import HOOK 

class BaseEngine(object):
    def __init__(self, *args, **kwargs):
        self._hooks = []

    @property
    def hooks(self):
        return self._hooks

    def call_hook(self, fn_name:str):
        """Call all hooks.
        Args:
            fn_name (str): The function name in each hook to be called, such as
                "before_train_epoch".
        """
        for hook in self._hooks:
            if getattr(hook, 'only_master', False) and self.local_rank not in [-1, 0]: continue
            getattr(hook, fn_name)(self)

    def register_hook(self, hook: HOOK, priority: int=-1):
        """Register a hook into the hook list.
        The hook will be inserted into a priority queue, with the specified
        priority (See :class:`Priority` for details of priorities).
        For hooks with the same priority, they will be triggered in the same
        order as they are registered.
        Args:
            hook (:obj:`Hook`): The hook to be registered.
            priority (int or str or :obj:`Priority`): Hook priority.
                Lower value means higher priority.
        """
        assert isinstance(hook, HOOK)
        if priority < 0:
            assert hasattr(hook, 'priority')
        else:
            hook.priority = priority
        # insert the hook to a sorted list
        idx = bisect.bisect_right([h.priority for h in self._hooks], hook.priority)
        self._hooks.insert(idx, hook)

    def update(self, sample):
        raise(NotImplementedError("No implementation"))

    def run(self, epochs=None):
        raise(NotImplementedError("No implementation"))

    def extract_performance(self, eval_names=None):
        if eval_names is None: eval_names = self.eval_names
        performance = []
        for i, eval_name in enumerate(eval_names):
            eval_name = eval_name.split('-')
            sign = 1 if len(eval_name) == 1 else -1
            eval_name = eval_name[-1].split('.')
            tmp = self.info.results
            for _n in eval_name:
                tmp = tmp.get(_n)
            performance.append(sign * tmp)
#            performance = sign * self.info.results[eval_name]
        return performance if len(performance)>1 else performance[0]

