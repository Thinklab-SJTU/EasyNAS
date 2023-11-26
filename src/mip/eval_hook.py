from src.hook.hook import HOOK

class TotalTimeHOOK(HOOK):
    def __init__(self, priority=0):
        self.priority = priority

    def after_iter(self, runner):
        runner.info.results['total_time'] = runner.info.results.get('total_time', 0) + runner.info.current_result['time']

