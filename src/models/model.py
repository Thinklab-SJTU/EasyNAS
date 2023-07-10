import sys
import inspect
from copy import deepcopy
from functools import partial
import logging
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')

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
        self.logger = logging.getLogger('model_builder')
        if log_path and local_rank in [-1, 0]:
            fh = logging.FileHandler(log_path)
            fh.setFormatter(logging.Formatter(log_format))
            self.logger.addHandler(fh)
        setup_for_distributed(local_rank in [-1, 0], self.logger)

        self.output_ch = output_ch
        self.input_ch = input_ch
        self.arch_list = architecture
        self.gw = width_multiple
        self.gd = depth_multiple
        self.model, self.save = self.parse_model(deepcopy(self.arch_list), ch=[input_ch], width_divisible=width_divisible)  # model, savelist

        # Init weights, biases
        if init_func is not None: self.apply(init_func)
        else: self.apply(default_init_weights)

        self.info(input_size)


    def info(self, input_size=None):
        if input_size:
            input_size = [input_size, input_size] if isinstance(input_size, int) else input_size
            self.logger.info("param size = %fMB, FLOPS=%10.1fG", count_parameters_in_MB(self), thop.profile(self.model, inputs=(torch.ones(1, self.input_ch, *input_size),), verbose=False)[0] / 1E9 if thop else 0)
        else:
            self.logger.info("param size = %fMB", count_parameters_in_MB(self))

    def forward(self, x):
        y = []  # outputs
        for m in self.model:
            if m.in_idx != -1:  # if not from previous layer
                x = y[m.in_idx] if isinstance(m.in_idx, int) else [x if j == -1 else y[j] for j in m.in_idx]  # from earlier layers
            x = m(x)  # run
            y.append(x if m.idx in self.save else None)  # save output
        return x

    def parse_model(self, arch_list, ch, width_divisible):  # model_dict, input_channels(3)
        self.logger.info('%3s%10s%10s%10s  %-20s%-40s' % ('', 'input_idx', 'repeat', 'params', 'layer', 'arguments'))
        gd, gw = self.gd, self.gw

        layers, save, out_ch = [], [], ch[-1]  # layers, savelist, ch out
        for i, v in enumerate(arch_list):
            in_idx = v['input_idx']
            num_repeat = max(round(v.get('num_repeat', 0) * gd), 1) 
            v['num_repeat'] = num_repeat
            is_outlayer = v.get('is_outlayer', False)
            layer = get_layer(v['submodule_name'])
            args = v['args']
            if 'num_repeat' in args.keys(): args['num_repeat'] = max(round(args['num_repeat'] * gd), 1)

            cin = [ch[idx] for idx in in_idx] if isinstance(in_idx, (list, tuple)) else ch[in_idx]
            arg_names = inspect.getfullargspec(layer.__init__).args
            if 'in_channel' in arg_names:
                args['in_channel'] = cin
            cout = args.get('out_channel', None)
            if cout:
                if not is_outlayer: 
                    cout = [int(make_divisible(c * gw, width_divisible)) for c in cout] if isinstance(cout, list) else make_divisible(cout*gw, width_divisible)
                    args['out_channel'] = cout
                else:
                    if self.output_ch: args['out_channel'] = self.putput_ch
            else:
                cout = get_outchannel(cin, v['submodule_name'], args)

            m_ = layer(**args)
            if num_repeat > 1:
                if 'in_channel' in args and 'out_channel' in args: 
                    args['in_channel'] = cout
                m_ = nn.Sequential(*[m_] + [layer(**args) for _ in range(num_repeat-1)])

            num_param = sum([x.numel() for x in m_.parameters()])  # number params
            self.logger.info('%3s%10s%10s%10.0f  %-20s%-40s' % (i, in_idx, num_repeat, num_param, v['submodule_name'], args))  # print

            m_.idx, m_.in_idx, m_.type, m_.np, m_.arch_yaml = i, in_idx, layer, num_param, deepcopy(v)  # attach index, 'from' index, type, number params

            save.extend(x % i for x in ([in_idx] if isinstance(in_idx, int) else in_idx) if x != -1)  # append to savelist
            layers.append(m_)
            if i == 0:
                ch = []
            ch.append(cout)
        return nn.Sequential(*layers), sorted(set(save))




class SearchModel(BaseModel, SearchModule):
    def __init__(self, architecture, output_ch, input_ch=3, input_size=None, depth_multiple=1., width_multiple=1., log_path=None, init_func=None, local_rank=-1):
        super(SearchModel, self).__init__(architecture, output_ch, input_ch=3, input_size=None, log_path=None, init_func=None, local_rank=-1)
        self.init_arch_parameters()
        self.info_arch()

    def init_arch_parameters(self):
        for i, m_ in enumerate(self.model):
            layer, arch_yaml = m_.type, m_.arch_yaml
            if issubclass(layer, SearchModule):
                arch_idx = arch_yaml.get('arch_idx', None)
                num_repeat = arch_yaml.get('num_repeat', 1)
                if arch_idx is not None:
                    module = self.model[arch_idx if arch_idx >=0 else i+arch_idx]
                    assert(type(module) == type(m_))
                    if num_repeat > 1:
                        for l in range(num_repeat):
                            m_[l].set_arch_parameters(module[l], recurse=True)
                    else:
                        m_.set_arch_parameters(module, recurse=True)
                elif arch_yaml.get('repeat_arch', False) and num_repeat > 1:
                    for l in range(1, num_repeat):
                        m_[l].set_arch_parameters(m_[0], recurse=True)


    def info_arch(self): 
        self.logger.info("="*20+"\n Search Layers") 
        self.logger.info('%3s%20s%10s%10s  %-40s' % ('idx', 'layer', 'repeat', 'repeat_arch', 'arch_parameters')) 
        for i, m_ in enumerate(self.model): 
            if issubclass(m_.type, SearchModule):
                arch_yaml = m_.arch_yaml 
                num_repeat = arch_yaml.get('num_repeat', 1) 
                repeat_arch = arch_yaml.get('repeat_arch', False) 
                self.logger.info('%3s%20s%10s%10s' % (i, m_.type, num_repeat, repeat_arch)) 
                if num_repeat == 1: 
                    for name, v in m_.named_arch_parameters(recurse=True):
                        self.logger(name, v)
                else: 
                    for l in range(num_repeat):
                        for name, v in m_[l].named_arch_parameters(recurse=True):
                            self.logger(name, v)
        self.logger.info("="*20)

    def discretize(self, outOp_name='BaseModel', depth_multiple=1., width_multiple=1.):
        new_cfg = self.init_output_yaml(outOp_name=outOp_name, depth_multiple=depth_multiple, width_multiple=width_multiple)

        new_arch = []
        for i, m_ in enumerate(self.model):
            if issubclass(m_, SearchModule):
                if isinstance(m_, nn.Sequential):
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
        out_model_yaml['architecture'] = new_arch
        return out_model_yaml



#class SearchModel(BaseModel, SearchModule):
#    def __init__(self, cfg, output_ch, input_ch=3, input_size=None, log_path=None, init_func=None, local_rank=-1):
#        super(SearchModel, self).__init__(cfg, output_ch, input_ch=3, input_size=None, log_path=None, init_func=None, local_rank=-1)
#        self.arch_param_list = self.init_arch_param()
#        self.info_arch()
#
#    def set_arch_param_layer(self, layer, ch_alphas, op_alphas, edge_alphas):
#        if ch_alphas is not None: layer.set_arch_param('ch_alphas', ch_alphas)
#        if op_alphas is not None: layer.set_arch_param('op_alphas', op_alphas)
#        if edge_alphas is not None: layer.set_arch_param('edge_alphas', edge_alphas)
#        return {
#            'ch_alphas': layer.get_ch_arch_param(),
#            'op_alphas': layer.get_op_arch_param(),
#            'edge_alphas': layer.get_edge_arch_param()
#        }
#
#    def init_arch_param(self):
#        arch_param_list = []
#        for i, m_ in enumerate(self.model):
#            layer, arch_yaml = m_.type, m_.arch_yaml
#            if issubclass(layer, SearchLayer):
#                arch_param_idx = arch_yaml.get('arch_param_idx', None)
#                ch_arch_param_idx = arch_yaml.get('ch_arch_param_idx', arch_param_idx)
#                op_arch_param_idx = arch_yaml.get('op_arch_param_idx', arch_param_idx)
#                edge_arch_param_idx = arch_yaml.get('edge_arch_param_idx', arch_param_idx)
#                # TODO: What if arch_param_idx is a repeated module?
#                set_arch_param = partial(
#                      self.set_arch_param_layer,
#                      ch_alphas=arch_param_list[ch_arch_param_idx]['ch_alphas'] if ch_arch_param_idx is not None else None,
#                      op_alphas=arch_param_list[op_arch_param_idx]['op_alphas'] if op_arch_param_idx is not None else None,
#                      edge_alphas=arch_param_list[edge_arch_param_idx]['edge_alphas'] if edge_arch_param_idx is not None else None
#                )
#                num_repeat = arch_yaml.get('num_repeat', 1)
#                if num_repeat > 1:
#                    if arch_yaml.get('repeat_arch', False):
#                        arch_param_list.append(set_arch_param(m_[0]))
#                        for l in range(1, num_repeat):
#                            self.set_arch_param_layer(m_[l], **arch_param_list[-1])
#                    else:
#                        arch_param_list.append(
#                              [set_arch_param(m_[l]) for l in range(num_repeat)]
#                        )
#
#                else:
#                    arch_param_list.append(set_arch_param(m_))
#
#            else:
#                arch_param_list.append(None)
#        return arch_param_list
#
#    def get_arch_param_layer(self, arch_yaml, p):
#        arch_param = {}
#        if arch_yaml.get('ch_arch_param_idx') is not None: 
#            arch_param['ch_alphas'] = "Idx {%d}"%arch_yaml.get('ch_arch_param_idx')
#            assert(p['ch_alphas'] == self.arch_param_list[arch_yaml.get('ch_arch_param_idx')]['ch_alphas'])
#        else: arch_param['ch_alphas'] = p['ch_alphas']
#        if arch_yaml.get('op_arch_param_idx') is not None: 
#            arch_param['op_alphas'] = "Idx {%d}"%arch_yaml.get('op_arch_param_idx')
#            assert(p['op_alphas'] == self.arch_param_list[arch_yaml.get('op_arch_param_idx')]['op_alphas'])
#        else: arch_param['op_alphas'] = p['op_alphas']
#        if arch_yaml.get('edge_arch_param_idx') is not None: 
#            arch_param['edge_alphas'] = "Idx {%d}"%arch_yaml.get('edge_arch_param_idx')
#            assert(p['edge_alphas'] == self.arch_param_list[arch_yaml.get('edge_arch_param_idx')]['edge_alphas'])
#        else: arch_param['edge_alphas'] = p['edge_alphas']
#        return arch_param
#
#    def info_arch(self): self.logger.info("="*20+"\n Search Layers") self.logger.info('%3s%20s%10s%10s  %-40s' % ('idx', 'layer', 'repeat', 'repeat_arch', 'arch_parameters')) for i, (m_, p) in zip(self.model, self.arch_param_list): if p is not None: assert(issubclass(m_.type, SearchLayer)) arch_yaml = m_.arch_yaml num_repeat = arch_yaml.get('num_repeat', 1) repeat_arch = arch_yaml.get('repeat_arch', False) if num_repeat == 1 or repeat_arch: arch_param = self.get_arch_param_layer(arch_yaml, p) else: assert isinstance(p, list) arch_param = [self.get_arch_param_layer(arch_yanl, p[l]) for l in range(num_repeat)] self.logger.info('%3s%20s%10s%10s  %-40s' % (i, m_.type, num_repeat, repeat_arch, arch_param)) self.logger.info("="*20)
#
#    def genotype(self):
#        out_model_yaml = {
#            'submodule_name': 'BaseModel',
#            'output_ch': self.output_ch,
#            'input_ch': self.input_ch,
#            'depth_multiple': 1.,
#            'width_multiple': 1.,
#            'log_path': self.log_path
#            }
#
#        new_arch = []
#
#        for i, (m_, p) in enumerate(zip(self.model, self.arch_param_list)):
#            if issubclass(m_, SearchLayer):
#                if isinstance(m_, nn.Sequential):
#                    if m_.arch_yaml.get('repeat_arch', False):
#                        new_arch.append(m_[0].genotype(m_.arch_yaml, **p))
#                    else:
#                        for l, tmp_m in enumerate(m_):
#                            tmp_arch = tmp_m.genotype(m_.arch_yaml, **p[l])
#                            tmp_arch['num_repeat'] = 1
#                            new_arch.append(tmp_arch)
#                else:
#                    new_arch.append(m_.genotyp(m_.arch_yaml, **p))
#
#            else:
#                new_arch.append(m_.arch_yaml)
#        out_model_yaml['architecture'] = new_arch
#        return out_model_yaml

