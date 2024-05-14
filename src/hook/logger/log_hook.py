import os
import sys
import logging
from typing import Union

from ..hook import HOOK, execute_period, only_master

class LogHOOK(HOOK):
    def __init__(self, priority=0, logger_name='TrainPip', log_path: Union[None, str] = None, print_freq: int = 1, only_master=True):
        self.priority = priority
        self.only_master = only_master
        self.print_freq = print_freq
        self.log_path = log_path
        self.logger_name = logger_name

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

#    def before_run(self, runner):
#        self.logger.info("param size = %fMB", count_parameters_in_MB(runner.model))

    @only_master
    @execute_period("print_freq")
    def after_train_iter(self, runner):
#        string = 'train %03d lr %e' % (runner.info.current_iter, runner.lr_scheduler.get_lr()[0])
        string = 'train %03d lr' % (runner.info.current_iter)
        for group in runner.optimizer.param_groups:
            string += ' %4.3e' % group['lr']
        for k, v in runner.info.results.train.items():
            if 'ignore' not in k:
                string += ' %s: %f' % (k, v)
        self.logger.info(string)

    @only_master
    def after_train_epoch(self, runner):
        string = 'Epoch %03d train' % (runner.info.current_epoch)
        for k, v in runner.info.results.train.items():
            string += ' %s: %f' % (k, v)
        self.logger.info(string)

    @only_master
    @execute_period("print_freq")
    def after_val_iter(self, runner):
        string = 'val %03d' % runner.info.current_iter
        for k, v in runner.info.results.val.items():
            if 'ignore' not in k:
                string += ' %s: %f' % (k, v)
        self.logger.info(string)

    @only_master
    def after_val_epoch(self, runner):
        string = 'Epoch %03d val' % (runner.info.current_epoch)
        for k, v in runner.info.results.val.items():
            if 'ignore' not in k:
                string += ' %s: %f' % (k, v)
        self.logger.info(string)
        self.logger.info('='*10+f'Epoch {runner.info.current_epoch} Done'+'='*10)

    @only_master
    @execute_period("print_freq")
    def after_iter(self, runner):
        string = '%03d lr' % (runner.info.current_iter)
        for group in runner.optimizer.param_groups:
            string += ' %4.3e' % group['lr']
        for k, v in runner.info.results.items():
            if 'ignore' not in k:
                string += ' %s: %f' % (k, v)
        self.logger.info(string)
