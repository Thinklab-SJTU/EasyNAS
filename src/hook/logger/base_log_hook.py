import os
import sys
import logging
from typing import Union

from ..hook import HOOK, execute_period

class BaseLogHOOK(HOOK):
    def __init__(self, priority=0, logger_name='Run', log_path: Union[None, str]=None, only_master=True):
        self.priority = priority
        self.only_master = only_master
        self.logger_name = logger_name
        self.log_path = log_path

    def before_run(self, runner):
        runner_root_path = getattr(runner, 'root_path', None)
        if self.log_path and not self.log_path.startswith('/') and runner_root_path:
            self.log_path = os.path.join(runner_root_path, self.log_path)

        if self.log_path and not os.path.exists(os.path.dirname(self.log_path)):
            os.makedirs(os.path.dirname(self.log_path))
        self.config_logger(self.logger_name, self.log_path)

    def config_logger(self, logger_name, log_path=None):
        log_format = '[%(asctime)s] [%(name)s] [%(levelname)s]: %(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
        self.logger = logging.getLogger(logger_name)
        if log_path:
            fh = logging.FileHandler(log_path)
            fh.setFormatter(logging.Formatter(log_format))
            self.logger.addHandler(fh)
        import builtins as __builtin__
        self.builtin_print = __builtin__.print
        __builtin__.print = self._print

    def _print(self, *args, sep=' ', end='\n', file=sys.stdout, flush=False):
        if len(args) == 1: args = args[0]
        self.logger.info(args)

    def after_run(self, runner):
        import builtins as __builtin__
        __builtin__.print = self.builtin_print

