import torch
import torch.nn as nn
from builder import create_model, create_criterion

default_aux_cfg = dict(
           submodule_name='src.models.BaseModel',
           args=dict(
               architecture=[
                   dict(input_idx=-1,
                        submodule_name='PoolBNAct',
                        args=dict(pool='avg', kernel=5, bn=False, act=False, stride=3, pad=0, count_include_pad=False)),
                   dict(input_idx=-1,
                        submodule_name='ConvBNAct',
                        args=dict(out_channel=128, kernel=1, bn='torch.nn.BatchNorm2d', act='torch.nn.ReLU')),
                   dict(input_idx=-1,
                        submodule_name='ConvBNAct',
                        args=dict(out_channel=768, kernel=2, bn='torch.nn.BatchNorm2d', act='torch.nn.ReLU')),
                   dict(input_idx=-1,
                        submodule_name='LinearBNAct',
                        is_outlayer=True, 
                        args=dict(out_channel=10, bn=False, act=False)),
                   ]
               )
           )

def refine_aux_cfg(input_ch, output_ch, cfg):
    cfg['args'].update(input_ch=input_ch, output_ch=output_ch)
    return cfg

class CriterionWithAux(nn.Module):
    def __init__(self, criterion_or_cfg, in_channel=None, out_channel=None, auxiliary_or_cfg=default_aux_cfg, auxiliary_weight=0.4, sync_bn=False, local_rank=-1):
        super(CriterionWithAux, self).__init__()

        self.auxiliary_weight = auxiliary_weight if isinstance(auxiliary_weight, (list, tuple)) else (auxiliary_weight,)
        self.device = torch.device('cuda', max(local_rank, 0))
        if not auxiliary_or_cfg:
            self.auxiliary = None
        elif isinstance(auxiliary_or_cfg, list):
            self.auxiliary = nn.ModuleList([self.build_auxiliary(cin, out_channel, aux, local_rank) for aux, cin in zip(auxiliary_or_cfg, in_channel)])
        else: 
            self.auxiliary = nn.ModuleList([self.build_auxiliary(in_channel, out_channel, auxiliary_or_cfg, local_rank)])

        if self.auxiliary and local_rank >= 0:
            for i in range(len(self.auxiliary)):
#            # convert BN to SyncBN
                if sync_bn:
                    self.auxiliary[i] = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.auxiliary[i])
                self.auxiliary[i] = torch.nn.parallel.DistributedDataParallel(self.auxiliary[i], device_ids=[local_rank], output_device=local_rank)

        if isinstance(criterion_or_cfg, dict):
            self.criterion = create_criterion(criterion_or_cfg)
        elif isinstance(criterion_or_cfg, nn.Module):
            self.criterion = criterion_or_cfg
        else:
            raise(ValueError(f"criterion_or_cfg should be an instance of dict or nn.Module, but got {type(auxiliary_or_cfg)}"))

    def build_auxiliary(self, in_channel, out_channel, auxiliary_or_cfg, local_rank):
        if isinstance(auxiliary_or_cfg, dict):
            aux = refine_aux_cfg(in_channel, out_channel, auxiliary_or_cfg)
            return create_model(auxiliary_or_cfg, local_rank=local_rank)
        elif isinstance(auxiliary_or_cfg, nn.Module):
            return auxiliary_or_cfg
        else:
            raise(ValueError(f"auxiliary_or_cfg should be an instance of dict or nn.Module, but got {type(auxiliary_or_cfg)}"))

    def forward(self, input, target):
        loss = self.criterion(input[-1] if isinstance(input, (list, tuple)) else input, target)
        if self.auxiliary:
            for i, (aux, aux_weight) in enumerate(zip(self.auxiliary, self.auxiliary_weight)):
                loss += aux_weight * self.criterion(aux(input[i]), target)
        return loss

