import os
from contextlib import contextmanager
import torch
import torch.nn as nn
import torch.distributed as dist

@contextmanager
def ddp_ctx(args):
    init_distributed_mode(args)
    yield
    if is_dist_avail_and_initialized():
        dist.barrier()
        dist.destroy_process_group()

@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    if local_rank not in [-1, 0]:
        dist.barrier()
    yield
    if local_rank == 0:
        dist.barrier()

def setup_for_distributed(is_master, logger=None):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print if logger is None else logger.info

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    if logger is None:
        __builtin__.print_ddp = __builtin__.print
        __builtin__.print = print
    else: logger.info = print

def is_parallel(model):
    return isinstance(model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel))


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() in [0, -1]


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.local_rank = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.local_rank = -1
        return

    torch.cuda.set_device(args.local_rank)
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    dist.barrier()
    setup_for_distributed(args.rank in [0, -1])

def synchronize_between_processes(var, device='cuda', mode='sum'):
    """
    Warning: does not synchronize the deque!
    """
    if not is_dist_avail_and_initialized():
        return var
    t = var if isinstance(var, torch.Tensor) else torch.tensor([var], device=device)
    dist.barrier()
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    if mode == 'avg': 
        t.div_(get_world_size())
    return t if isinstance(var, torch.Tensor) else t.tolist()[0]
