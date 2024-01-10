from contextlib import contextmanager

def only_master(func):
    def inner(self, runner, *args, **kwargs):
        if (not hasattr(runner, 'local_rank')) or runner.local_rank in [-1, 0]:
            return func(self, runner, *args, **kwargs)
    return inner

def execute_period(attr_name=None, n=None):
    def wrapper(func):
        def inner(self_, *args, **kwargs):
            setattr(self_, 'execute_period_count', getattr(self_, 'execute_period_count', {}))
            count = self_.execute_period_count.get(func.__name__, 0)
#            count = getattr(self_, 'execute_period_count', {}).get(func.__name__, 0)
            if count == 0:
                func(self_, *args, **kwargs)

            attr = '_execute_period_'+func.__name__ if attr_name is None else attr_name
            execute_period = getattr(self_, attr, n if n is not None else 1)
            self_.execute_period_count[func.__name__] = (count + 1) % execute_period
        return inner
    return wrapper


class HOOK(object):
    def __init__(self, only_master=False, ns=[2, 3]):
        self.only_master = only_master
        self.n = 1
        self.set_period('before_run', ns[0])
        self.set_period('after_run', ns[1])

    def set_period(self, name, n):
        setattr(self, '_execute_period_'+name, n)

    def reset_period(self):
        for k in getattr(self, 'execute_period_count', {}).keys():
            self.execute_period_count[k] = 0

#    @execute_period('n')
    def before_run(self, runner):
#        print("before run")
        pass
 
#    @execute_period()
    def after_run(self, runner):
#        print("after run")
        pass
 
    def before_epoch(self, runner):
        pass
 
    def after_epoch(self, runner):
        self.reset_period()
 
    def before_train_epoch(self, runner):
        pass
 
    def before_val_epoch(self, runner):
        pass
 
    def after_train_epoch(self, runner):
        pass
 
    def after_val_epoch(self, runner):
        pass
 
    def before_train_iter(self, runner):
        pass
 
    def before_val_iter(self, runner):
        pass
 
    def after_train_iter(self, runner):
        pass
 
    def after_val_iter(self, runner):
        pass

    def before_iter(self, runner):
        pass
 
    def after_iter(self, runner):
        pass
 

@contextmanager
def hooks_ctx(fn_name, hooks, runner):
    for hook in hooks: getattr(hook, 'before_'+fn_name)(runner)
    yield
    for hook in hooks: getattr(hook, 'after_'+fn_name)(runner)

@contextmanager
def hooks_run(hooks, runner):
    for hook in hooks: hook.before_run(runner)
    yield
    for hook in hooks: hook.after_run(runner)
@contextmanager
def hooks_epoch(hooks, runner):
    for hook in hooks: hook.before_epoch(runner)
    yield
    for hook in hooks: 
        hook.after_epoch(runner)
        hook.reset_period()
@contextmanager
def hooks_train_epoch(hooks, runner):
    for hook in hooks: hook.before_train_epoch(runner)
    yield
    for hook in hooks: hook.after_train_epoch(runner)
@contextmanager
def hooks_val_epoch(hooks, runner):
    for hook in hooks: hook.before_val_epoch(runner)
    yield
    for hook in hooks: hook.after_val_epoch(runner)
@contextmanager
def hooks_train_iter(hooks, runner):
    for hook in hooks: hook.before_train_iter(runner)
    yield
    for hook in hooks: hook.after_train_iter(runner)
@contextmanager
def hooks_val_iter(hooks, runner):
    for hook in hooks: hook.before_val_iter(runner)
    yield
    for hook in hooks: hook.after_val_iter(runner)
def hooks_iter(hooks, runner):
    return hooks_ctx('iter', hooks, runner)

 
if __name__ == '__main__':
    hook = HOOK([2,4])
    for i in range(10):
        print(i)
        hook.before_epoch('')
        hook.before_run('')
        hook.after_run('')
