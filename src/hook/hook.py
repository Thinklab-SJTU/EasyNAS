
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
    def __init__(self):

    def before_run(self, runner):
        pass
 
    def after_run(self, runner):
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
    hook = HOOK()
    for i in range(10):
        print(i)
        hook.before_epoch('')
        hook.before_run('')
        hook.after_run('')
