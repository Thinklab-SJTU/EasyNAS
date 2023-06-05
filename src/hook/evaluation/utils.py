import torch.distributed as dist

from app.distribute_utils import is_dist_avail_and_initialized

class AverageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt

  def synchronize_between_processes(self):
      """
      Warning: does not synchronize the deque!
      """
      if not is_dist_avail_and_initialized():
          return
      t = torch.tensor([self.cnt, self.sum], dtype=torch.float64, device='cuda')
      dist.barrier()
      dist.all_reduce(t)
      t = t.tolist()
      self.cnt = int(t[0])
      self.sum = t[1]
      self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].reshape(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res
