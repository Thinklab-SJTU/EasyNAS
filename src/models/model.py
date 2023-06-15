import sys
import inspect
from copy import deepcopy
import logging
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')

import torch
import torch.nn as nn
import thop

from .utils import count_parameters_in_MB, make_divisible, default_init_weights
from .layers.utils import get_submodule
from app.distribute_utils import setup_for_distributed
#from builder.utils import get_submodule as utils_get_submodule

def get_outchannel(cin, module_name, module_args):
    if module_name in ['Concat']:
        return sum(cin)
    elif module_name in ['Contract']:
        return cin * module_args['gain']**2
    elif module_name in ['Expand']:
        return cin // module_args['gain']**2
    else: return cin

class BaseModel(nn.Module):
    def __init__(self, cfg, output_ch, input_ch=3, input_size=None, log_path=None, init_func=None, local_rank=-1):
        super(BaseModel, self).__init__()
        self.logger = logging.getLogger('model_builder')
        if log_path and local_rank in [-1, 0]:
            fh = logging.FileHandler(log_path)
            fh.setFormatter(logging.Formatter(log_format))
            self.logger.addHandler(fh)
        setup_for_distributed(local_rank in [-1, 0], self.logger)

        assert isinstance(cfg, dict)
        self.output_ch = output_ch
        self.input_ch = input_ch
        self.cfg = cfg
        self.model, self.save = self.parse_model(deepcopy(self.cfg), ch=[input_ch])  # model, savelist

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

    def parse_model(self, cfg, ch):  # model_dict, input_channels(3)
        self.logger.info('%3s%10s%10s%10s  %-20s%-40s' % ('', 'input_idx', 'repeat', 'params', 'module', 'arguments'))
        gd, gw = cfg.get('depth_multiple', 1), cfg.get('width_multiple', 1)

        layers, save, out_ch = [], [], ch[-1]  # layers, savelist, ch out
        for i, v in enumerate(cfg['architecture']):
            in_idx = v['input_idx']
            num_repeat = max(round(v.get('num_repeat', 0) * gd), 1) 
            is_outlayer = v.get('is_outlayer', False)
            module = get_submodule(v['module'])
            args = v['module_args']
            if 'num_repeat' in args.keys(): args['num_repeat'] = max(round(args['num_repeat'] * gd), 1)

            cin = [ch[idx] for idx in in_idx] if isinstance(in_idx, (list, tuple)) else ch[in_idx]
            arg_names = inspect.getfullargspec(module.__init__)
            if 'in_channel' in arg_names.args:
                args['in_channel'] = cin
            cout = args.get('out_channel', None)
            if cout:
                if not is_outlayer: 
                    cout = [int(make_divisible(c * gw, 8)) for c in cout] if isinstance(cout, list) else make_divisible(cout*gw, 8)
                    args['out_channel'] = cout
                else:
                    if self.output_ch: args['out_channel'] = self.putput_ch
            else:
                cout = get_outchannel(cin, v['module'], args)

            m_ = module(**args)
            if num_repeat > 1:
                if 'in_channel' in args and 'out_channel' in args: 
                    args['in_channel'] = cout
                m_ = nn.Sequential(*[m_] + [module(**args) for _ in range(num_repeat-1)])
#            m_ = nn.Sequential(*[module(**args) for _ in range(num_repeat)]) if num_repeat > 1 else module(**args)  # module

            num_param = sum([x.numel() for x in m_.parameters()])  # number params
            self.logger.info('%3s%10s%10s%10.0f  %-20s%-40s' % (i, in_idx, num_repeat, num_param, v['module'], args))  # print

            m_.idx, m_.in_idx, m_.type, m_.np, m_.cfg = i, in_idx, v['module'], num_param, deepcopy(v)  # attach index, 'from' index, type, number params

            save.extend(x % i for x in ([in_idx] if isinstance(in_idx, int) else in_idx) if x != -1)  # append to savelist
            layers.append(m_)
            if i == 0:
                ch = []
            ch.append(cout)
        return nn.Sequential(*layers), sorted(set(save))

class SearchModel(BaseModel):
    def __init__(self, cfg, output_ch, input_ch=3, input_size=None, log_path=None, init_func=None, local_rank=-1):
        super(SearchModel, self).__init__(cfg, output_ch, input_ch=3, input_size=None, log_path=None, init_func=None, local_rank=-1)

    def genotype(self):
        out_cfg = deepcopy(self.cfg)
        new_arch = []
        for cfg, (name, layer) in zip(out_cfg['architecture'], self.model.named_children()):
            new_arch.append(self.genotype_layer(layer, cfg))
        out_cfg['architecture'] = new_arch
        self.display_genotype(out_cfg)
        return out_cfg

    def genotype_layer(self, layer, cfg):
        if isinstance(layer, SearchLayer):
            return layer.genotype(cfg)
        elif isinstance(layer, nn.Sequential):
            for n, l in layer.named_children():
                self.genotype_layer(l, cfg)
        else: return cfg
           
