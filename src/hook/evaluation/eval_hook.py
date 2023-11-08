from ..hook import HOOK, execute_period
from .utils import AverageMeter, accuracy

class EvalAccHOOK(HOOK):
    def __init__(self, priority=0, only_master=False):
        self.priority = priority
        self.only_master = only_master
        self.loss = AverageMeter()
        self.top1 = AverageMeter()
        self.top5 = AverageMeter()
#        self.val_loss = AverageMeter()
        self.val_top1 = AverageMeter()
        self.val_top5 = AverageMeter()

    def before_train_epoch(self, runner):
        self.loss.reset()
        self.top1.reset()
        self.top5.reset()

    def before_val_epoch(self, runner):
#        self.val_loss.reset()
        self.val_top1.reset()
        self.val_top5.reset()

    def after_train_iter(self, runner):
        logits, target, iter_loss = runner.info.train_bs_logits, runner.info.train_bs_target, runner.info.train_bs_loss
        if isinstance(logits, (list, tuple)): logits = logits[-1]
        prec1, prec5 = accuracy(logits, target, topk=(1, 5))
        n = target.size(0)
        self.loss.update(iter_loss.item(), n)
        self.top1.update(prec1.item(), n)
        self.top5.update(prec5.item(), n)
        runner.info.results.train.loss = self.loss.avg
        runner.info.results.train.top1 = self.top1.avg
        runner.info.results.train.top5 = self.top5.avg

    def after_val_iter(self, runner):
        logits, target = runner.info.val_bs_logits, runner.info.val_bs_target
        if isinstance(logits, (list, tuple)): logits = logits[-1]
        prec1, prec5 = accuracy(logits, target, topk=(1, 5))
        n = target.size(0)
#        self.val_loss.update(loss.item(), n)
        self.val_top1.update(prec1.item(), n)
        self.val_top5.update(prec5.item(), n)
#        runner.info.results.val.loss = self.val_loss.avg
        runner.info.results.val.top1 = self.val_top1.avg
        runner.info.results.val.top5 = self.val_top5.avg

    def after_val_epoch(self, runner):
        best = runner.info.results.val.get('best', 0)
        runner.info.results.is_best = best < runner.info.results.val.top1
        if runner.info.results.is_best:
            runner.info.results.val['best'] = runner.info.results.val.top1
        


