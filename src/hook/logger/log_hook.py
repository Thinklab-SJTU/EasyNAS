import os
import sys
import logging
from typing import Union

from ..hook import HOOK, execute_period

class LogHOOK(HOOK):
    def __init__(self, logger_name='TrainPip', log_path: Union[None, str] = None, print_freq: int = 1):
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

#    def before_run(self, runner):
#        self.logger.info("param size = %fMB", count_parameters_in_MB(runner.model))

    @execute_period("print_freq")
    def after_train_iter(self, runner):
        string = 'train %03d lr %e' % (runner.info.iter_step, runner.lr_scheduler_hook.lr_scheduler.get_lr()[0])
        for k, v in runner.info.results.train.items():
            string += ' %s: %f' % (k, v)
        self.logger.info(string)

    def after_train_epoch(self, runner):
        string = 'Epoch %03d train' % (runner.info.current_epoch)
        for k, v in runner.info.results.train.items():
            string += ' %s: %f' % (k, v)
        self.logger.info(string)

    @execute_period("print_freq")
    def after_val_iter(self, runner):
        string = 'val %03d' % runner.info.iter_step
        for k, v in runner.info.results.val.items():
            string += ' %s: %f' % (k, v)
        self.logger.info(string)

    def after_val_epoch(self, runner):
        string = 'Epoch %03d val' % (runner.info.current_epoch)
        for k, v in runner.info.results.val.items():
            string += ' %s: %f' % (k, v)
        self.logger.info(string)
