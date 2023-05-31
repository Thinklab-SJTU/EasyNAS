
def execute_period(n):
    def wrapper(func):
        def inner(self, *args, **kwargs):
            setattr(self, 'count', getattr(self, 'count', {}))
            count = self.count.get(func.__name__, 0)
            if count == 0:
                func(self, *args, **kwargs)
            self.count[func.__name__] = (count + 1) % n
        return inner
    return wrapper


class HOOK(object):
    def __init__(self, ns=[2, 3]):
        self.ns = ns

    def before_run(self, runner):
 
    def after_run(self, runner):
 
    def before_epoch(self, runner):
        pass
 
    def after_epoch(self, runner):
        pass
 
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

class _HOOK(object):
    @execute_period(self.ns[0])
    def before_run(self, runner):
        print("before run")
        pass

    @execute_period(self.ns[1])
    def after_run(self, runner):
        print("after run")
        pass

    def before_epoch(self, runner):
        pass

    def after_epoch(self, runner):
        pass

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

 
if __name__ == '__main__':
    hook = HOOK([2,3])
    for i in range(10):
        print(i)
        hook.before_epoch('')
        hook.before_run('')
        hook.after_run('')

