from copy import deepcopy
import logging
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')

from .utils import count_parameters_in_MB
from builder.utils import get_submodule as utils_get_submodule

def get_outchannel(cin, module_name, module_args):
    if module_name in ['Concat']:
        return sum(cin)
    elif module_name in ['Contract']:
        return cin * module_args['gain']**2
    elif module_name in ['Expand']:
        return cin // module_args['gain']**2
    else: return cin

class BaseModel(nn.Module):
    def __init__(self, cfg, output_ch, input_ch=3, input_size=None, log_path=None):
        self.logger = logging.getLogger('model_builder')
        if log_path:
            fh = logging.FileHandler(log_path)
            fh.setFormatter(logging.Formatter(log_format))
            self.logger.addHandler(fh)

        assert isinstance(cfg, dict)
        self.output_ch = output_ch
        self.input_ch = input_ch
        self.cfg = cfg
        self.model, self.save = self.parse_model(deepcopy(self.cfg), ch=[input_ch])  # model, savelist

        # Init weights, biases
        self.initialize_weights()
        self.info(input_size)

    def info(self, input_size=None):
        if input_size:
            self.logger.info("param size = %fMB, FLOPS=%10.1f", count_parameters_in_MB(self), thop.profile(m, inputs=(torch.ones(1, self.input_ch, *input_size),), verbose=False)[0] / 1E9 * 2 if thop else 0)
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

    def get_submodule(self, submodule_name):
        self.submodule_map = getattr(self, 'submodule_map', {})
        submodule = utils_get_submodule(submodule_name, '.layers', package_path='src.models', loaded_submodule=self.submodule_map)
        return submodule

    def parse_model(self, cfg, ch):  # model_dict, input_channels(3)
        self.logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'input_idx', 'repeat', 'params', 'module', 'arguments'))
        gd, gw = cfg.get('depth_multiple', 1), cfg.get('width_multiple', 1)

        layers, save, out_ch = [], [], ch[-1]  # layers, savelist, ch out
        for i, v in enumerate(cfg['architecture']):
            in_idx = v['input_idx']
            num_repeat = max(round(v.get('num_repeat', 0) * gd), 1) 
            is_outlayer = v.get('is_outlayer', False)
            module = self.get_submodule(v['module'])
            args = v['module_args']
            if 'num_repeat' in args.keys(): args['num_repeat'] = max(round(args['num_repeat'] * gd), 1)

            cin = [ch[idx] for idx in in_idx] if isinstance(in_idx, (list, tuple)) else ch[in_idx]
            args['in_channels'] = cin
            cout = args.get('out_channels', None)
            if cout:
                if not is_outlayer: 
                    cout = [int(make_divisible(c * gw, 8)) for c in cout] if isinstance(cout, list) else make_divisible(cout*gw, 8)
                    args['out_channels'] = cout
            else:
                cout = get_outchannel(cin, v['module'], args)

            m_ = nn.Sequential(*[module(**args) for _ in range(num_repeat)]) if num_repeat > 1 else module(**args)  # module
            num_param = sum([x.numel() for x in m_.parameters()])  # number params

            m_.idx, m_.in_idx, m_.type, m_.np, m.cfg = i, in_idx, v['module'], num_param, deepcopy(v)  # attach index, 'from' index, type, number params
            self.logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, in_idx, num_repeat, num_param, m_.type, args))  # print

            save.extend(x % i for x in ([in_idx] if isinstance(in__idx, int) else in_idx) if x != -1)  # append to savelist
            layers.append(m_)
            if i == 0:
                ch = []
            ch.append(cout)
        return nn.Sequential(*layers), sorted(set(save))

class SearchModel(BaseModel):
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
           
