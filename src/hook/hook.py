
def execute_period(attr_name=None, n=None):
    def wrapper(func):
        def inner(self_, *args, **kwargs):
            setattr(self_, 'count', getattr(self_, 'count', {}))
            count = self_.count.get(func.__name__, 0)
            if count == 0:
                func(self_, *args, **kwargs)

            attr = '_execute_period_'+func.__name__ if attr_name is None else attr_name
            execute_period = getattr(self_, attr, n if n is not None else 1)
            self_.count[func.__name__] = (count + 1) % execute_period
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
    hook = HOOK([2,4])
    for i in range(10):
        print(i)
        hook.before_epoch('')
        hook.before_run('')
        hook.after_run('')
