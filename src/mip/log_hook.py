import os
import sys
import logging
from typing import Union

from src.hook.hook import HOOK, execute_period

class LogHOOK(HOOK):
    def __init__(self, priority=0, logger_name='MIPEngine', log_path: Union[None, str] = None, print_freq: int = 1):
        self.priority = priority
        self.print_freq = print_freq
        if log_path and not os.path.exists(os.path.dirname(log_path)):
            os.makedirs(os.path.dirname(log_path))
        self.config_logger(logger_name, log_path)

    def config_logger(self, logger_name, log_path=None):
        log_format = '[%(asctime)s] [%(name)s] [%(levelname)s]: %(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
        self.logger = logging.getLogger(logger_name)
        if log_path:
            fh = logging.FileHandler(log_path)
            fh.setFormatter(logging.Formatter(log_format))
            self.logger.addHandler(fh)

    @execute_period("print_freq")
    def after_iter(self, runner):
#        string = 'train %03d lr %e' % (runner.info.current_iter, runner.lr_scheduler.get_lr()[0])
        string = 'Instance %03d:' % (runner.info.current_iter)
        for k, v in runner.info.results.items():
            if isinstance(v, list):
                string += f' {k}: {v[-1]}'
            else:
                string += f' {k}: {v}'
        self.logger.info(string)

