from ..hook import HOOK, execute_period
from .utils import AverageMeter, accuracy

class EvalAccHOOK(HOOK):
    def __init__(self):
        loss = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

    def after_train_iter(self, runner):
        logits, target, loss = runner.info.train_bs_logits, runner.info.train_bs_target, runner.info.train_bs_loss
        prec1, prec5 = accuracy(logits, target, topk=(1, 5))
        n = target.size(0)
        loss.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)
        runner.info.results.train.loss = loss.avg
        runner.info.results.train.top1 = top1.avg
        runner.info.results.train.top5 = top5.avg

    def after_val_iter(self, runner):
        logits, target, loss = runner.info.val_bs_logits, runner.info.val_bs_target, runner.info.val_bs_loss
        prec1, prec5 = accuracy(logits, target, topk=(1, 5))
        n = target.size(0)
        loss.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)
        runner.info.results.val.loss = loss.avg
        runner.info.results.val.top1 = top1.avg
        runner.info.results.val.top5 = top5.avg

