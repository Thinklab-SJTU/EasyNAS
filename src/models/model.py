import sys
from collections.abc import Iterable
import inspect
from copy import deepcopy
from functools import partial
import logging
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')

import numpy as np
import torch
import torch.nn as nn
import thop

from .utils import count_parameters_in_MB, make_divisible, default_init_weights
from .layers.utils import get_layer, gumbel_softmax
from .layers.base import SearchModule
from app.distribute_utils import setup_for_distributed

def get_outchannel(cin, layer_name, args):
    if layer_name in ['Concat']:
        return sum(cin)
    elif layer_name in ['Contract']:
        return cin * args['gain']**2
    elif layer_name in ['Expand']:
        return cin // args['gain']**2
    else: return cin

class BaseModel(nn.Module):
    def __init__(self, architecture, output_ch, input_ch=3, input_size=None, depth_multiple=1., width_multiple=1., width_divisible=1, log_path=None, init_func=None, local_rank=-1):
        super(BaseModel, self).__init__()
        self.local_rank = local_rank
        self.device = torch.device('cuda', max(local_rank, 0))

        self.logger = logging.getLogger('model_builder')
        if log_path and local_rank in [-1, 0]:
            fh = logging.FileHandler(log_path)
            fh.setFormatter(logging.Formatter(log_format))
            self.logger.addHandler(fh)
        setup_for_distributed(local_rank in [-1, 0], self.logger)

        self.output_ch = output_ch
        self.input_ch = input_ch
        self.log_path = log_path
        self.arch_list = architecture
        self.gw = width_multiple
        self.gd = depth_multiple
        self.model, self.save = self.parse_model(deepcopy(self.arch_list), ch=[input_ch], width_divisible=width_divisible)  # model, savelist

        # Init weights, biases
        if init_func is not None: self.apply(init_func)
        else: self.apply(default_init_weights)

        if input_size:
            self.input_size = [input_size, input_size] if isinstance(input_size, int) else input_size
        else: self.input_size = None

        self.info(self.input_size)

        self.to(self.device)

    def info(self, input_size=None):
        if input_size:
            self.logger.info("param size = %fMB, FLOPS=%fG", count_parameters_in_MB(self), thop.profile(self, inputs=(torch.zeros(1, self.input_ch, *input_size),), verbose=False)[0] / 1E9 if thop else 0)
        else:
            self.logger.info("param size = %fMB", count_parameters_in_MB(self))

    def forward(self, x):
        def get_input(feature_list, idx):
            if len(feature_list) == 0: return x
            return feature_list[idx] if isinstance(idx, int) else [feature_list[j] for j in idx]  # from earlier layers

        y, outputs = [], []  
        for m in self.model:
            if isinstance(m, nn.ModuleList):
                tmp_y = y.copy()
                for m_ in m:
                    tmp_y.append(m_(get_input(tmp_y, m.in_idx)))
                x = tmp_y[-1]
                del tmp_y
            else:
                x = m(get_input(y, m.in_idx))  # run
            y.append(x)
            if m.arch_yaml.get('is_outlayer', False): outputs.append(x)
        if len(outputs) == 0: 
            return x
        elif len(outputs) == 1:
            return outputs[0]
        return outputs

    def parse_model(self, arch_list, ch, width_divisible):  # model_dict, input_channels(3)
        self.logger.info('%3s%10s%10s%10s  %-20s%-40s' % ('', 'input_idx', 'repeat', 'params', 'layer', 'arguments'))
        gd, gw = self.gd, self.gw

        layers, save, out_ch = [], [], ch[-1]  # layers, savelist, ch out
        for layer_idx, v in enumerate(arch_list):
            in_idx = v['input_idx']
            if 'num_repeat' in v:
                v['num_repeat'] = num_repeat = max(round(v.get('num_repeat') * gd), 1) 
            else:
                num_repeat = 1
            freeze_ch = v.get('freeze_ch', False)
            layer = get_layer(v['submodule_name'])
            args = v['args']
            if 'num_repeat' in args.keys(): args['num_repeat'] = max(round(args['num_repeat'] * gd), 1)

            cin = [ch[idx] for idx in in_idx] if isinstance(in_idx, Iterable) else ch[in_idx]
            arg_names = inspect.getfullargspec(layer.__init__).args
            if 'input_idx' in arg_names:
                args['input_idx'] = in_idx
            if 'in_channel' in arg_names:
                args['in_channel'] = cin
            cout = args.get('out_channel', None)

            if not cout:
                cout = get_outchannel(cin, v['submodule_name'], args)
            elif not freeze_ch:
                if isinstance(cout, Iterable):
                    for i in range(len(cout)):
                        cout[i] = int(make_divisible(cout[i] * gw, width_divisible))
                else: 
                    cout = make_divisible(cout*gw, width_divisible)
#                    cout = [int(make_divisible(c * gw, width_divisible)) for c in cout] if isinstance(cout, list) else make_divisible(cout*gw, width_divisible)
                args['out_channel'] = cout
            if isinstance(cout, Iterable): cout = max(cout)

            m_ = layer(**args)

            if num_repeat > 1:
                if 'in_channel' in args and 'out_channel' in args: 
                    if isinstance(in_idx, Iterable):
                        cin = ch + [cout]
                        args['in_channel'] = [cin[idx] for idx in in_idx]
                    else:
                        args['in_channel'] = cout
                m_ = nn.ModuleList([m_] + [layer(**args) for _ in range(num_repeat-1)])

            num_param = sum([x.numel() for x in m_.parameters()])  # number params
            self.logger.info('%3s%10s%10s%10.0f  %-20s%-40s' % (layer_idx, in_idx, num_repeat, num_param, v['submodule_name'], args))  # print

            m_.idx, m_.in_idx, m_.type, m_.np, m_.arch_yaml = layer_idx, in_idx, layer, num_param, deepcopy(v)  # attach index, 'from' index, type, number params

#            save.extend(x % layer_idx for x in ([in_idx] if isinstance(in_idx, int) else in_idx) if x != -1)  # append to savelist
            layers.append(m_)
            if layer_idx == 0:
                ch = []
            ch.append(cout)
#        return nn.Sequential(*layers), sorted(set(save))
        return nn.ModuleList(layers), sorted(set(save))




class SearchModel(BaseModel, SearchModule):
    def __init__(self, architecture, output_ch, input_ch=3, input_size=None, depth_multiple=1., width_multiple=1., log_path=None, init_func=None, local_rank=-1):
        super(SearchModel, self).__init__(architecture, output_ch, input_ch=input_ch, input_size=input_size, 
                depth_multiple=depth_multiple, width_multiple=width_multiple,
                log_path=log_path, init_func=init_func, local_rank=local_rank)

        self.init_arch_parameters()
        self.info_arch()


    def init_arch_parameters(self):
        for i, m_ in enumerate(self.model):
            layer, arch_yaml = m_.type, m_.arch_yaml
            if issubclass(layer, SearchModule):
                arch_idx = arch_yaml.get('arch_idx', None)
                num_repeat = arch_yaml.get('num_repeat', 1)
                if arch_idx is not None: # same architecture as the (arch_idx)-th layer
                    module = self.model[arch_idx if arch_idx >=0 else i+arch_idx]
                    assert(type(module) == type(m_))
                    if num_repeat > 1:
                        for l in range(num_repeat):
                            m_[l].set_arch_parameters(module[l], recurse=True)
                    else:
                        m_.set_arch_parameters(module, recurse=True)
                elif arch_yaml.get('repeat_arch', False) and num_repeat > 1: # repeat the same architecture
                    m_[0].apply_arch_parameters(lambda x: x.to(self.device))
                    for l in range(1, num_repeat):
                        m_[l].set_arch_parameters(m_[0], recurse=True)
                else:
                    if num_repeat > 1:
                        for l in range(num_repeat):
                            m[l].apply_arch_parameters(lambda x: x.to(self.device))
                    else:
                        m_.apply_arch_parameters(lambda x: x.to(self.device))


    def info_arch(self): 
        self.logger.info("="*10+"Search Layers arch_parameters"+"="*10) 
        self.logger.info('%3s%20s%10s%20s  %-40s' % ('idx', 'layer', 'repeat', 'repeat_arch', 'arch_parameters')) 
        hash_param = {}
        for i, m_ in enumerate(self.model): 
            if issubclass(m_.type, SearchModule):
                arch_yaml = m_.arch_yaml 
                num_repeat = arch_yaml.get('num_repeat', 1) 
                repeat_arch = arch_yaml.get('repeat_arch', False) 
                self.logger.info('%3s%20s%10s%20s' % (i, arch_yaml['submodule_name'], num_repeat, repeat_arch)) 
                if num_repeat == 1 or repeat_arch: 
                    for name, v in m_.named_arch_parameters(recurse=True):
                        if v in hash_param: 
                            self.logger.info(f"{name} is the same as {hash_param[v]}")
                        else:
                            self.logger.info(name)
#                            self.logger.info(v.cpu().data.numpy())
                            self.logger.info(torch.softmax(v, dim=-1).cpu().data.numpy())
                            hash_param[v] = f"Layer{i}:{name}"
#                        self.logger.info('%10s  %-40s' % (name, v.data.numpy().tolist()))
                else: 
                    for l in range(num_repeat):
                        for name, v in m_[l].named_arch_parameters(recurse=True):
                            if v in hash_param: 
                                self.logger.info(f"{name} is the same as {hash_param[v]}")
                            else:
                                self.logger.info(name)
                                self.logger.info(v.data.numpy())
                                hash_param[v] = f"Layer{i}:repeat{l}:{name}"
#                            self.logger.info('%10s  %-40s' % (name, v.data.numpy().tolist()))
        self.logger.info("="*40)

    def discretize(self, outOp_name='BaseModel', depth_multiple=1., width_multiple=1.):
        out_model_yaml = self.init_output_yaml(
                outOp_name=outOp_name, 
                depth_multiple=depth_multiple, 
                width_multiple=width_multiple,
                output_ch=self.output_ch,
                input_ch=self.input_ch,
                log_path=self.log_path
                )

        new_arch = []
        for i, m_ in enumerate(self.model):
            if issubclass(m_.type, SearchModule):
                if isinstance(m_, nn.ModuleList):
                    if m_.arch_yaml.get('repeat_arch', False):
                        new_arch.append(m_[0].discretize(m_.arch_yaml))
                    else:
                        for l, tmp_m in enumerate(m_):
                            tmp_arch = tmp_m.discretize(m_.arch_yaml)
                            tmp_arch['num_repeat'] = 1
                            new_arch.append(tmp_arch)
                else:
                    new_arch.append(m_.discretize(m_.arch_yaml))

            else:
                new_arch.append(m_.arch_yaml)
        out_model_yaml['args']['architecture'] = new_arch

        return out_model_yaml



