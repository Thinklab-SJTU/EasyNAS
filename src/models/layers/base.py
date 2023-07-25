from collections import namedtuple
from copy import deepcopy
import inspect
from inspect import isfunction
from easydict import EasyDict as edict
import torch
import torch.nn as nn

from .utils import get_layer

#OP_CFG = namedtuple('OP_CFG', ['submodule_name', 'args'])


class SearchModule(nn.Module):
    def __init__(self):
        super(SearchModule, self).__init__()
        self._arch_parameters = {}

    def set_outOp(self, name=None):
        setattr(self, 'outOp_name', self.__class__.__name__.rstrip("_search") if name is None else name)
        setattr(self, 'outOp', get_layer(self.outOp_name))

    def init_output_yaml(self, arch_yaml=None, outOp_name=None, input_idx=-1, **kwargs):
        if outOp_name is not None:
            outOp = get_layer(outOp_name)
        elif hasattr(self, 'outOp_name'):
            outOp_name, outOp = self.outOp_name, self.outOp
        else:
            self.set_outOp()
            outOp_name, outOp = self.outOp_name, self.outOp

#            arch_yaml = {
#                'args': {k: getattr(self, k) for k in inspect.signature(self.__init__).parameters.keys() if hassttr(self, k)}
#                }
        if arch_yaml is None:
            new_arch = dict(submodule_name=outOp_name, input_idx=input_idx, args={})
        else:
            new_arch = deepcopy(arch_yaml)
            new_arch['submodule_name'] = outOp_name
            new_arch['input_idx'] = input_idx

        # del unused variables
        need_key = inspect.signature(outOp.__init__).parameters.keys()
        if arch_yaml is not None:
            for k in arch_yaml['args'].keys():
                if k not in need_key: del new_arch['args'][k]

        for k, v in kwargs.items():
            assert k in need_key
            new_arch['args'][k] = v

        return new_arch

    def init_arch_parameters(self, arch_name, *shape):
#        arch_param = 1e-3*torch.randn(*shape, requires_grad=True)
        arch_param = torch.normal(mean=0, std=1e-3, size=shape, requires_grad=True)
        if arch_name in self._arch_parameters: 
            del self._arch_parameters[arch_name]
            delattr(self, arch_name)
        setattr(self, arch_name, arch_param)
        self._arch_parameters[arch_name] = arch_param

    def apply_arch_parameters(self, fn, recurse=True, memo=None):
        if memo is None:
            memo = set()
        memo.add(self)
        for key, param in self._arch_parameters.items():
            if param is None:
                continue
            # Tensors stored in modules are graph leaves, and we don't want to
            # track autograd history of `param_applied`, so we have to use
            # `with torch.no_grad():`
            with torch.no_grad():
                param_applied = fn(param)
#            param.data = param_applied
#            out_param = param
            assert param.is_leaf
            out_param = param_applied.requires_grad_(param.requires_grad)
            self._arch_parameters[key] = out_param
            if hasattr(self, key): 
                delattr(self, key)
                setattr(self, key, out_param)

            if param.grad is not None:
                with torch.no_grad():
                    grad_applied = fn(param.grad)
                assert param.grad.is_leaf
                out_param.grad = grad_applied.requires_grad_(param.grad.requires_grad)

        if recurse:
            for module in self.modules():
                if isinstance(module, SearchModule) and module not in memo:
                    module.apply_arch_parameters(fn, memo)



    def set_arch_parameters(self, module_or_dict, recurse=True, memo=None):
        if memo is None:
            memo = set()
        if isinstance(module_or_dict, dict):
            for na, np in module_or_dict.items():
                assert hasattr(self, na) and na in self._arch_parameters
                del self._arch_parameters[na]
                self._arch_parameters[na] = np
                delattr(self, na)
                setattr(self, na, np)

        elif isinstance(module_or_dict, SearchModule):
            if module_or_dict in memo: return 
            memo.add(module_or_dict)
            self.set_arch_parameters(module_or_dict._arch_parameters)
            if recurse:
                src_named_modules = {k: v for k, v in module_or_dict.named_modules(prefix="") if isinstance(v, SearchModule)}
                for dist_name, dist_module in self.named_modules(prefix=""):
                    if dist_module not in memo and isinstance(dist_module, SearchModule):
                        assert dist_name in src_named_modules
                        dist_module.set_arch_parameters(src_named_modules[dist_name], memo=memo)

    def arch_parameters(self, recurse=True):
        for name, param in self.named_arch_parameters(recurse=recurse):
            yield param

    def named_arch_parameters(self, prefix='', recurse=True, remove_duplicate=True):
        gen = self._named_members(
            lambda module: module._arch_parameters.items() if isinstance(module, SearchModule) else {},
            prefix=prefix, recurse=recurse,
#            remove_duplicate=remove_duplicate
            )
        yield from gen

    def norm_arch_parameters(self, alphas, gumbel=False):
        return gumbel_softmax(F.log_softmax(alphas, dim=-1), hard=True) if gumbel else nn.functional.softmax(alphas, dim=-1)

    def get_norm_layer(self, ch_alphas, bn, gumbel_channel=True):
        return bn[ch_alphas.argmax()] if gumbel_channel else bn

    def get_reserved_idx(self, num_reserved, weight):
        return [x.item() for x in torch.topk(weight, k=num_reserved, dim=-1)[1]]

    def discretize(self):
        raise(NotImplementedError(f"No Implementation of genotype func for {self}"))

    def forward(self, x):
        raise(NotImplementedError("No implementation"))

class OpBuilder(object):
    def __init__(self, auto_refine=False, adjust_ch_op=None, upsample_op=None): 
        self.auto_refine = auto_refine
        self.adjust_ch_op = edict(submodule_name='ConvBNAct', args=dict(kernel=1, dilation=1, bn=False, act=None)) if adjust_ch_op is None else adjust_ch_op
        self.upsample_op = edict(submodule_name=nn.Upsample, args=dict(size=None, scale_factor=None, mode='nearest', align_corners=None)) if upsample_op is None else upsample_op

    def refine_C_stride(self, op_config, in_channel, out_channel, stride, **update_args):
        if isinstance(op_config, edict):
            op_config = [op_config]
        refined_op_config = []
        if isinstance(in_channel, int): in_channel = (in_channel,)*len(op_config)
        if isinstance(out_channel, int): out_channel = (out_channel,)*len(op_config)
        if isinstance(stride, int): stride = (stride,)*len(op_config)
        for idx, (cin, cout, s, op) in enumerate(zip(in_channel, out_channel, stride, op_config)):
            refined_op = deepcopy(op)
            tmp_update_args = {}
            for k, v in update_args:
                tmp_update_args[k] = v[idx] if isinstance(v, (list, tuple)) else v
            if isinstance(op, edict):
                up_s, s = int(1./s), max(1, s)
                adjust_ch = False
                refined_op.args.update(stride=s, **tmp_update_args)
                tmp_module = get_layer(refined_op.submodule_name)
                if isfunction(tmp_module):
                    arg_names = inspect.getfullargspec(tmp_module).args
                else:
                    arg_names = inspect.getfullargspec(tmp_module.__init__).args
                if 'in_channel' in arg_names: 
                    refined_op.args.update(in_channel=cin)
                if 'out_channel' in arg_names: 
                    refined_op.args.update(out_channel=cout)
                if cin != cout and ('in_channel' not in arg_names or 'out_channel' not in arg_names):
                    Warning("Input channel should be the same as output channel. Otherwise, you should set auto_refine as True")
                    adjust_ch = True
                if self.auto_refine and up_s > 1: 
                    upsample_op = deepcopy(self.upsample_op)
                    upsample_op.args.update(scale_factor=up_s)
                    refined_op = [refined_op]
                    refined_op.append(upsample_op)
                if self.auto_refine and adjust_ch: 
                    adjust_ch_op = deepcopy(self.adjust_ch_op)
                    adjust_ch_op.args.update(in_channel=cin, out_channel=cout)
                    if isinstance(refined_op, edict):
                        refined_op = [refined_op]
                    refined_op.append(adjust_ch_op)
            elif isinstance(op, (tuple, list)):
                Warning("Sequential op will set the out_channel of last op as cout, and set the out_channel and in_channel of other ops as cin; set stride of first op as stride and set others' as 1.")
                refined_op = list(refined_op)
                refined_op[0].args.update(stride=s, in_channel=cin, out_channel=cin, **tmp_update_args)
                for i in range(1, len(op)-1):
                    refined_op[i].args.update(stride=1, in_channel=cin, out_channel=cin, **tmp_update_args)
                refined_op[-1].args.update(in_channel=cin, out_channel=cout, stride=1, **tmp_update_args)
            else: raise(ValueError(f"No implementation for op as type {type(op)}"))

            refined_op_config.append(refined_op)
        return tuple(refined_op_config)

    def _build_op(self, op_config):
        ops = nn.ModuleList([])
        for config in op_config:
            if isinstance(config, edict):
                module = get_layer(config.submodule_name) 
                op = module(**config.args)
            elif isinstance(config, (tuple, list)):
                op = nn.Sequential()
                for idx, sub_config in enumerate(config):
                    module = get_layer(sub_config.submodule_name) 
                    op.add_module(str(idx), module(**sub_config.args))
            else: 
                raise(TypeError("op_config should be either easydict or sequence"))
            ops.append(op)
        return ops

    def build_op(self, op_config, in_channel, out_channel, stride, **update_args):
        op_config = self.refine_C_stride(op_config, in_channel, out_channel, stride, **update_args)
        return self._build_op(op_config)



class OpLayer(nn.Module):
    def __init__(self, in_channel, out_channel, stride, op, act=nn.ReLU(), bn=True, auto_refine=False, adjust_ch_op=None, upsample_op=None):
        super(OpLayer, self).__init__()
        op_builder = OpBuilder(auto_refine=auto_refine, adjust_ch_op=adjust_ch_op, upsample_op=upsample_op)
        self.op = OpBuilder.build_op(op, in_channel, out_channel, stride)
        self.act = get_act(act)
        self.bn = nn.BatchNorm2d(self.cout) if bn else None

    def forward(self, x):
       out = sum(op(x) for op in self.op)
       if self.bn: out = self.bn(out)
       if self.act: out = self.act(out)
       return out
        

#class ParallelOpLayer(SearchLayer, OpLayer):
#    def __init__(self, in_channel, out_channel, candidate_op=darts_candidate_op, candidate_ch=[1.], gumbel_op=False, gumbel_channel=True, stride=1, act=nn.ReLU(), bn=True, independent_ch_arch_param=True, independent_op_arch_param=True):
#        super(ParallelOpLayer, self).__init__()
#        self.set_outOp("SingleOpLayer")
#        self.candidate_op = candidate_op
#        self.candidate_ch = candidate_ch
#        self.gumbel_op = gumbel_op 
#        self.gumbel_channel = gumbel_channel and len(candidate_ch)>1
#
#        self.adjust_ch_op = OP(submodule_name='ConvBNAct_search', args=dict(candidate_op=[(1,1)], candidate_ch=candidate_ch, gumbel_channel=gumbel_channel, stride=1, bn=False, act=None, independent_ch_arch_param=False))
#
#        self.candidate_op = candidate_op
#        self.refined_candidate_op = self.refine_C_stride(candidiate_op, in_channel=in_channel, out_channnel=out_channel, stride=stride)
#        self.ops = self.build_op(self.refined_candidate_op)
#
#        self.num_alphas_each_op = []
#        for op in candidate_op:
#            self.num_alphas_each_op.append(
#                 len(op.args['candidate_op']) if hasattr(op.args, 'candidate_op') else -1)
#        self.num_op_alphas = sum(abs(x) for x in self.num_alphas_each_op)
#
#        self.act = get_act(act)
#        if self.gumbel_channel: self.bn = nn.ModuleList([nn.BatchNorm2d(int(self.cout*e)) for e in candidate_ch]) if bn else [None for _ in candidate_ch]
#        else: self.bn = nn.BatchNorm2d(self.cout) if bn else None
#
#        self.init_arch_param(independent_ch_arch_param, independent_op_arch_param)
#
#    def init_arch_param(self, ind_ch_alpha, ind_op_alpha):
#        if self.num_op_alphas > 1 and ind_op_alpha:
#            super().init_arch_param('op_alphas', len(self.num_op_alphas))
#
#        if len(self.candidate_ch) > 1 and ind_ch_alpha:
#            super().init_arch_param('ch_alphas', len(self.candidate_ch))
#
#    def forward(self, x, op_alphas=None, ch_alphas=None):
#        op_alphas = op_alphas if op_alphas is not None else (self.norm_arch_param(self.op_alphas, self.gumbel_op) if hasattr(self, 'op_alphas') else [1.])
#        ch_alphas = ch_alphas if ch_alphas is not None else (self.norm_arch_param(self.ch_alphas, self.gumbel_channel) if hasattr(self, 'ch_alphas') else [1.])
#        bn = self.get_norm_layer(ch_alphas, self.bn, self.gumbel_channel)
#
#        out, ptr = 0., 0
#        for idx, (op, num_alphas_each_op) in enumerate(zip(op_alphas, self.ops, self.num_alphas_each_op)):
#            if num_alphas_each_op > 0: 
#                end_ptr = ptr + num_alphas_each_op
#                out = out + op(x, op_alphas=op_alpha[ptr:end_ptr], ch_alphas=ch_alphas)
#                ptr = end_ptr
#            else: 
#                out = out + op_alpha[ptr] * op(x)
#                ptr += 1
#
#        if bn: out = bn(out)
#        if self.act: out = self.act(out)
#
#        return out
#
#    @classmethod
#    def genotype(self, cfg, op_alphas=None, ch_alphas=None, edge_alpha=None, num_reserved_op=None, num_reserved_ch=None, num_reserved_edge=None):
#        num_reserved_op = self.num_reserved_op if num_reserved_op is None else num_reserved_op
#        num_reserved_ch = self.num_reserved_ch if num_reserved_ch is None else num_reserved_ch
#        assert num_reserved_op==1
#        assert num_reserved_ch==1
#
#        new_cfg = deepcopy(cfg)
#        # del unused variables
#        need_key = inspect.signature(self.outOp.__init__).parameters.keys()
#        for k in cfg.keys():
#            if k not in need_key: del new_cfg['module_args'][k]
#
#        op_alphas = op_alphas if op_alphas is not None else (self.get_op_arch_param() if hasattr(self, 'op_alphas') else None)
#        op_alphas_idx = self.get_reserved_idx(num_reserved_op, op_alphas)
#        seen_num = 0
#        for op_idx, num in enumerate(self.num_alphas_each_op):
#            seen_num = seen_num + (num if num > 0 else 1)
#            if seen_num > op_alphas_idx:
#                break
#
#        new_cfg['module'] = self.outOp_name
#        if num > 0: # (Sep)ConvBNAct_search
#            select_op = self.candidate_op[op_idx]
#            layer_cfg = self.get_layer(select_op.Optype).genotype(select_op.args, op_alphas=op_alphas, ch_alphas=None, edge_alphas=None, num_reserved_op=num_reserved_op)
#            new_cfg['module_args']['op'] = OP(submodule_name=layer_cfg['module'], args=layer_cfg['module_args'])
#        else:
#            new_cfg['module_args']['op'] = self.candidate_op[op_idx]
#
#        ch_alphas = ch_alphas if ch_alphas is not None else (self.get_ch_arch_param() if hasattr(self, 'ch_alphas') else None)
#        if ch_alphas is not None:
#            ch_alphas_idx = self.get_reserved_idx(num_reserved_ch, ch_alphas)
#            new_cfg['module_args']['out_channel'] = cfg['module_args']['out_channel'] * cfg['module_args']['candidate_ch']
#
#        return new_cfg


