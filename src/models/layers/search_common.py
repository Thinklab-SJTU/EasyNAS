from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import autopad, gumbel_softmax, darts_candidate_op, eautodet_candidate_op, OP, get_layer, get_act
from .base import OpLayer

__all__ = ["SearchLayer", "ConvBNAct_search", "SepConvBNAct_search", "ParallelOpLayer", "AFF"]


class SearchLayer(nn.Module):
    def __init__(self):
        super(SearchLayer, self).__init__()
        self.set_outOp()
        self.num_reserved_op = 1
        self.num_reserved_ch = 1
        self.num_reserved_edge = 2

    def set_outOp(self, name=None):
        setattr(self, 'outOp_name', self.__class__.__name__.rstrip("_search") if name is None else name)
        setattr(self, 'outOp', get_layer(self.outOp_name))

    def forward(self, x):
        raise(NotImplementedError("No implementation"))

    def get_norm_layer(self, ch_alphas, bn, gumbel_channel=True):
        return bn[ch_alphas.argmax()] if gumbel_channel else bn

    def init_arch_param(self, arch_name, num_item):
        self.register_buffer(arch_name, torch.autograd.Variable(1e-3*torch.randn(len(self.kd)), requires_grad=True))

    def get_arch_param(self):
        out = []
        out.extend(self.get_op_arch_param())
        out.extend(self.get_ch_arch_param())
        out.extend(self.get_edge_arch_param())
        return out

    def get_op_arch_param(self):
        out = getattr(self, op_arch_param, None)
#        for n, m in self.named_modules():
#            if isinstance(m, SearchLayer):
#                out.extend(m.get_op_arch_param())
        return out

    def get_ch_arch_param(self):
        out = getattr(self, ch_arch_param, None)
#        for n, m in self.named_modules():
#            if isinstance(m, SearchLayer):
#                out.extend(m.get_ch_arch_param())
        return out

    def get_edge_arch_param(self):
        out = getattr(self, edge_arch_param, None)
#        for n, m in self.named_modules():
#            if isinstance(m, SearchLayer):
#                out.extend(m.get_edge_arch_param())
        return out

    def norm_arch_param(self, alphas, gumbel=False):
        return gumbel_softmax(F.log_softmax(alphas, dim=-1), hard=True) if gumbel else nn.functional.softmax(alphas, dim=-1)

    @classmethod
    def genotype(cls, cfg, op_alphas=None, ch_alphas=None, edge_alpha=None, num_reserved_op=1, num_reserved_ch=1, num_reserved_edge=2):
        raise(NotImplementedError(f"No implementation of function genotype for class {cls.__class__.__name__}"))

class ConvBNAct_search(SearchLayer):
    # Mixed Depthwise Conv https://arxiv.org/abs/1907.09595
    def __init__(self, in_channel, out_channel, candidate_op=[(1,1), (3,1), (5,1), (3,2)], candidate_ch=[1.], gumbel_op=False, gumbel_channel=True, stride=1, pad=None, group=1, act=True, bn=True, independent_ch_arch_param=True, independent_op_arch_param=True, bias=False, merge_kernel=True):
        # k=0 means zero op; d=0 means skip-connection
        super(Conv_search, self).__init__()
        self.merge_kernel = merge_kernel
        self.kd = candidate_op
        self.candidate_ch = candidate_ch
        self.stride = stride
        self.group = group
        self.gumbel_op = gumbel_op and len(candidate_op)>1
        self.gumbel_channel = gumbel_channel and len(candidate_ch)>1
        self.cout = out_channel
        cout_max = int(out_channel * max(candidate_ch))

        self.k_max = int(max([(k-1)*d+1 for k, d in candidate_op]))
        if merge_kernel:
            self.padding = (self.k_max - 1)//2
            self.weight = self.init_weight(cout_max, in_channel, self.k_max)
            self.bias = self.init_bias(cout_max, self.weight) if bias else None
        else:
            self.weight, self.bias = nn.ParameterList([]), nn.ParameterList([])
            for k, d in candidate_op:
                self.weight.append(self.init_weight(cout_max, in_channel, k))
                self.bias.append(self.init_bias(cout_max, self.weight[-1]))

        self.act = get_act(act)
        if self.gumbel_channel: self.bn = nn.ModuleList([nn.BatchNorm2d(int(self.cout*e)) for e in candidate_ch]) if bn else [None for _ in candidate_ch]
        else: self.bn = nn.BatchNorm2d(self.cout) if bn else None

        self.init_arch_param(independent_ch_arch_param, independent_op_arch_param)

    
    def init_weight(self, cout, cin, kernel):
        kernel = [kernel, kernel] if isinstance(kernel, int) else kernel
        tmp1 = torch.Tensor(cout, cin, *kernel)
        torch.nn.init.kaiming_normal_(tmp1, mode='fan_in')
        return nn.Parameter(tmp1)

    def init_bias(self, c, weight):
        b = torch.Tensor(c)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(b, -bound, bound)
        return nn.Parameter(b)

    def init_arch_param(self, ind_ch_alpha, ind_op_alpha):
        if len(self.kd) > 1 and ind_op_alpha:
            super().init_arch_param('op_alphas', len(self.kd))

        if len(self.candidate_ch) > 1 and ind_ch_alpha:
            super().init_arch_param('ch_alphas', len(self.candidate_ch))

    def get_merge_kernel(self, w_base, alphas, merge=True):
        merge_kernel = 0.
        if merge:
            for i, alpha in enumerate(alphas):
                k,d = self.kd[i]
                tmp_ks = (k-1)*d + 1
                start = int((self.k_max - tmp_ks) / 2)
                end = int(self.k_max - start)
                w = torch.zeros_like(w_base)
                w[:,:,start:end:d, start:end:d] = w_base[:,:,start:end:d, start:end:d]
                merge_kernel += w * alpha
        else:
            for i, alpha in enumerate(alphas):
                k,d = self.kd[i]
                tmp_ks = (k-1)*d + 1
                start = int((self.k_max - tmp_ks) / 2)
                end = int(self.k_max - start)
                c1, c2, _, _ = w_base[0].shape
                w = torch.zeros(c1,c2,self.k_max, self.k_max, dtype=w_base[0].dtype, device=w_base[0].device)
                w[:,:,start:end:d, start:end:d] = w_base[i][:,:,start:end:d, start:end:d]
                merge_kernel += w * alpha

        return merge_kernel

    def deal_merge_kernel_cout(self, merge_kernel, alphas, bias):
        Cout = merge_kernel.size(0)
        channel_mask = torch.zeros([Cout], dtype=merge_kernel.dtype, device=merge_kernel.device)
        if self.gumbel_channel:
            a_e, idx = alphas.max()
            merge_kernel = merge_kernel[:int(self.cout*self.candidate_e[idx]),:,:,:] * a_e
            if bias is not None: bias = bias[:int(self.cout*e)] 
        else:
            channel_idx = torch.arange(0, Cout, dtype=merge_kernel.dtype, device=merge_kernel.device).long()
#            channel_idx = torch.sort(merge_kernel.view(Cout,-1).sum(dim=-1), descending=True)[1]
            for e, a_e in zip(self.candidate_ch, alphas):
                channel_mask[channel_idx[:int(e*self.cout)]] += a_e
            merge_kernel = merge_kernel * channel_mask.view(-1,1,1,1)
        return merge_kernel, bias

        
    def forward(self, x, op_alphas=None, ch_alphas=None):
        Cin = x.size(1)
        bias = self.bias
        op_alphas = op_alphas if op_alphas is not None else (self.norm_arch_param(self.op_alphas, self.gumbel_op) if hasattr(self, 'op_alphas') else [1.])
        ch_alphas = ch_alphas if ch_alphas is not None else (self.norm_arch_param(self.ch_alphas, self.gumbel_channel) if hasattr(self, 'ch_alphas') else [1.])
        bn = self.get_norm_layer(ch_alphas, self.bn, self.gumbel_channel)
                                   
        merge_kernel = self.get_merge_kernel(self.weight, op_alphas, merge=self.merge_kernel) if len(self.kd>1) else (self.weight if self.merge_kernel else self.weight[0])

        if Cin != merge_kernel.size(1): merge_kernel = merge_kernel[:,:Cin,:,:]
        if len(self.candidate_ch) > 1:
            merge_kernel, bias = self.deal_merge_kernel_cout(merge_kernel, ch_alphas, self.bias)

        out = torch.nn.functional.conv2d(x, merge_kernel, stride=self.stride, padding=self.padding, dilation=1, groups=self.group)
        out = out + bias.view(1,-1,1,1) if bias is not None else out
        out = bn(out) if bn is not None else out
        out = act(out) if act is not None else out
        return out

    def get_reserved_idx(self, num_reserved, weight):
        return weight.argmax(dim=-1).item() if num_reserved==1 else [x.item() for x in torch.topk(weight, k=num_reserved, dim=-1)[1]]

    @classmethod
    def genotype(cls, cfg, op_alphas=None, ch_alphas=None, edge_alpha=None, num_reserved_op=None, num_reserved_ch=None, num_reserved_edge=None):
        num_reserved_op = cls.num_reserved_op if num_reserved_op is None else num_reserved_op
        num_reserved_ch = cls.num_reserved_ch if num_reserved_ch is None else num_reserved_ch
        assert num_reserved_op==1
        assert num_reserved_ch==1

        new_cfg = deepcopy(cfg)
        new_cfg['module'] = cls.outOp_name
        # del unused variables
        need_key = inspect.signature(cls.outOp.__init__).parameters.keys()
        for k in cfg.keys():
            if k not in need_key: del new_cfg['module_args'][k]

        ch_alphas = ch_alphas if ch_alphas is not None else (cls.get_ch_arch_param() if hasattr(cls, 'ch_alphas') else None)
        if ch_alphas is not None:
            ch_alphas_idx = cls.get_reserved_idx(num_reserved_ch, ch_alphas)
            new_cfg['module_args']['out_channel'] = cfg['module_args']['out_channel'] * cfg['module_args']['candidate_ch'][ch_alphas_idx]

        op_alphas = op_alphas if op_alphas is not None else (cls.get_op_arch_param() if hasattr(cls, 'op_alphas') else None)
        if op_alphas is not None:
            op_alphas_idx = cls.get_reserved_idx(num_reserved_op, op_alphas)
            new_cfg['module_args']['kernel'], new_cfg['module_args']['dilation'] = cfg['module_args']['candidate_op'][op_alphas_idx]
        return new_cfg


class SepConvBNAct_search(ConvBNAct_search):
    def init_weight(self, cout, cin, kernel):
        kernel = [kernel, kernel] if isinstance(kernel, int) else kernel
        point_w = torch.Tensor(cin, 1, ks, ks)
        torch.nn.init.kaiming_normal_(point_w, mode='fan_in')
        depth_w = torch.Tensor(cout, cin, *kernel)
        torch.nn.init.kaiming_normal_(depth_w, mode='fan_in')
        return nn.ParameterDict({
            'point_weight': nn.Parameter(point_w), 
            'depth_weight': nn.Parameter(depth_w)
        })

    def init_bias(self, c, weight):
        b = torch.Tensor(c)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight['point_weight'])
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(b, -bound, bound)
        return nn.Parameter(b)

    def get_merge_kernel(self, w_base, alphas, merge=True):
        merge_kernel = 0.
        if merge:
            for i, alpha in enumerate(alphas):
                k,d = self.kd[i]
                tmp_ks = (k-1)*d + 1
                start = int((self.k_max - tmp_ks) / 2)
                end = int(self.k_max - start)
                w = torch.zeros_like(w_base['depth_weight'])
                w[:,:,start:end:d, start:end:d] = w_base['depth_weight'][:,:,start:end:d, start:end:d]
                merge_kernel += w * alpha
        else:
            raise(ValueError("weight cannot be merged in SepConv if merge_kernel is False"))
        return merge_kernel

    def forward(self, x, op_alphas=None, ch_alphas=None):
        Cin = x.size(1)
        bias = self.bias
        op_alphas = op_alphas if op_alphas is not None else (self.norm_arch_param(self.op_alphas, self.gumbel_op) if hasattr(self, 'op_alphas') else [1.])
        ch_alphas = ch_alphas if ch_alphas is not None else (self.norm_arch_param(self.ch_alphas, self.gumbel_channel) if hasattr(self, 'ch_alphas') else [1.])
        bn = self.get_norm_layer(ch_alphas, self.bn, self.gumbel_channel)

        if self.merge_kernel:
            merge_kernel = self.get_merge_kernel(self.weight, op_alphas, merge=True) if len(self.kd>1) else self.weight['depth_weight']
            if Cin != merge_kernel.size(1): merge_kernel = merge_kernel[:Cin,:,:,:]
            out = torch.nn.functional.conv2d(x, merge_kernel, stride=self.stride, padding=self.padding, dilation=1, groups=Cin)
            # out channel for point-wise conv
            point_weight = self.weight['point_weight']
            if len(self.candidate_ch) > 1:
                merge_kernel, bias = self.deal_merge_kernel_cout(point_weight, ch_alphas, self.bias)
            if Cin != point_weight.size(1): point_weight = point_weight[:,:Cin,:,:]
            out = torch.nn.functional.conv2d(out, point_weight, stride=1, padding=0, dilation=1, groups=self.group)
            out = out + bias.view(1,-1,1,1) if bias is not None else out
        else:
            out = 0.
            op_alphas_norm = nn.functional.softmax(op_alphas, dim=-1)
            ch_alphas_norm = gumbel_softmax(F.log_softmax(ch_alphas, dim=-1), hard=True) if self.gumbel_channel else nn.functional.softmax(ch_alphas, dim=-1)
            for i, weight in enumerate(self.weight):
                depth_weight = weight['depth_weight']
                point_weight = weight['point_weight']
                if Cin != depth_weight.size(1): depth_weight = depth_weight[:Cin,:,:,:]
                tmp_out = torch.nn.functional.conv2d(x, depth_weight, stride=self.stride, padding=self.padding, dilation=1, groups=Cin)
                if len(self.candidate_ch) > 1:
                    point_weight, bias = self.deal_merge_kernel_cout(point_weight, ch_alphas_norm, self.bias)
                if Cin != point_weight.size(1): point_weight = point_weight[:,:Cin,:,:]
                tmp_out = torch.nn.functional.conv2d(tmp_out, point_weight, stride=1, padding=0, dilation=1, groups=self.group)
                out += op_alphas_norm[i] * (tmp_out + bias.view(1,-1,1,1) if bias is not None else tmp_out)

        out = bn(out) if bn is not None else out
        out = act(out) if act is not None else out
        return out



class ParallelOpLayer(SearchLayer, OpLayer):
    def __init__(self, in_channel, out_channel, candidate_op=darts_candidate_op, candidate_ch=[1.], gumbel_op=False, gumbel_channel=True, stride=1, act=nn.ReLU(), bn=True, independent_ch_arch_param=True, independent_op_arch_param=True):
        super(ParallelOpLayer, self).__init__()
        self.set_outOp("SingleOpLayer")
        self.candidate_op = candidate_op
        self.candidate_ch = candidate_ch
        self.gumbel_op = gumbel_op 
        self.gumbel_channel = gumbel_channel and len(candidate_ch)>1

        self.adjust_ch_op = OP(OPtype='ConvBNAct_search', args=dict(candidate_op=[(1,1)], candidate_ch=candidate_ch, gumbel_channel=gumbel_channel, stride=1, bn=False, act=None, independent_ch_arch_param=False))

        self.candidate_op = candidate_op
        self.refined_candidate_op = self.refine_C_stride(candidiate_op, in_channel=in_channel, out_channnel=out_channel, stride=stride)
        self.ops = self.build_op(self.refined_candidate_op)

        self.num_alphas_each_op = []
        for op in candidate_op:
            self.num_alphas_each_op.append(
                 len(op.args['candidate_op']) if hasattr(op.args, 'candidate_op') else -1)
        self.num_op_alphas = sum(abs(x) for x in self.num_alphas_each_op)

        self.act = get_act(act)
        if self.gumbel_channel: self.bn = nn.ModuleList([nn.BatchNorm2d(int(self.cout*e)) for e in candidate_ch]) if bn else [None for _ in candidate_ch]
        else: self.bn = nn.BatchNorm2d(self.cout) if bn else None

        self.init_arch_param(independent_ch_arch_param, independent_op_arch_param)

    def init_arch_param(self, ind_ch_alpha, ind_op_alpha):
        if self.num_op_alphas > 1 and ind_op_alpha:
            super().init_arch_param('op_alphas', len(self.num_op_alphas))

        if len(self.candidate_ch) > 1 and ind_ch_alpha:
            super().init_arch_param('ch_alphas', len(self.candidate_ch))

    def forward(self, x, op_alphas=None, ch_alphas=None):
        op_alphas = op_alphas if op_alphas is not None else (self.norm_arch_param(self.op_alphas, self.gumbel_op) if hasattr(self, 'op_alphas') else [1.])
        ch_alphas = ch_alphas if ch_alphas is not None else (self.norm_arch_param(self.ch_alphas, self.gumbel_channel) if hasattr(self, 'ch_alphas') else [1.])
        bn = self.get_norm_layer(ch_alphas, self.bn, self.gumbel_channel)

        out, ptr = 0., 0
        for idx, (op, num_alphas_each_op) in enumerate(zip(op_alphas, self.ops, self.num_alphas_each_op)):
            if num_alphas_each_op > 0: 
                end_ptr = ptr + num_alphas_each_op
                out = out + op(x, op_alphas=op_alpha[ptr:end_ptr], ch_alphas=ch_alphas)
                ptr = end_ptr
            else: 
                out = out + op_alpha[ptr] * op(x)
                ptr += 1

        if bn: out = bn(out)
        if self.act: out = self.act(out)

        return out

    @classmethod
    def genotype(self, cfg, op_alphas=None, ch_alphas=None, edge_alpha=None, num_reserved_op=None, num_reserved_ch=None, num_reserved_edge=None):
        num_reserved_op = self.num_reserved_op if num_reserved_op is None else num_reserved_op
        num_reserved_ch = self.num_reserved_ch if num_reserved_ch is None else num_reserved_ch
        assert num_reserved_op==1
        assert num_reserved_ch==1

        new_cfg = deepcopy(cfg)
        # del unused variables
        need_key = inspect.signature(self.outOp.__init__).parameters.keys()
        for k in cfg.keys():
            if k not in need_key: del new_cfg['module_args'][k]

        op_alphas = op_alphas if op_alphas is not None else (self.get_op_arch_param() if hasattr(self, 'op_alphas') else None)
        op_alphas_idx = self.get_reserved_idx(num_reserved_op, op_alphas)
        seen_num = 0
        for op_idx, num in enumerate(self.num_alphas_each_op):
            seen_num = seen_num + (num if num > 0 else 1)
            if seen_num > op_alphas_idx:
                break

        new_cfg['module'] = self.outOp_name
        if num > 0: # (Sep)ConvBNAct_search
            select_op = self.candidate_op[op_idx]
            layer_cfg = self.get_layer(select_op.Optype).genotype(select_op.args, op_alphas=op_alphas, ch_alphas=None, edge_alphas=None, num_reserved_op=num_reserved_op)
            new_cfg['module_args']['op'] = OP(OPtype=layer_cfg['module'], args=layer_cfg['module_args'])
        else:
            new_cfg['module_args']['op'] = self.candidate_op[op_idx]

        ch_alphas = ch_alphas if ch_alphas is not None else (self.get_ch_arch_param() if hasattr(self, 'ch_alphas') else None)
        if ch_alphas is not None:
            ch_alphas_idx = self.get_reserved_idx(num_reserved_ch, ch_alphas)
            new_cfg['module_args']['out_channel'] = cfg['module_args']['out_channel'] * cfg['module_args']['candidate_ch']

        return new_cfg


class AFF(SearchLayer, OpLayer):
    # Auto-Feature Fusion
    def __init__(self, in_channels, out_channel, strides, candidate_op=darts_candidate_op, candidate_ch=[1.], gumbel_op=False, gumbel_channel=True, gumbel_edge=False, act=nn.ReLU(), bn=True, independent_ch_arch_param=True, independent_op_arch_param=True, independent_edge_arch_param=True):
        """
        strides: a list indicating the scale for each edge. Whether to up-sampling or down-sampling, and how much the degree is
        """
        super(AFF, self).__init__()
        self.set_outOp("FuseLayer")

        self.check_valid(in_channels, strides)
        self.cin = in_channels
        self.cout = out_channel
        self.strides = strides
        self.candidate_ch = candidate_ch
        self.gumbel_op = gumbel_op
        self.gumbel_channel = gumbel_channel and len(candidate_ch)>1
        self.gumbel_edge = gumbel_edge
        self.candidate_op = candidate_op
        self.candidate_ch = candidate_ch

        self.m = nn.ModuleList([])
        for cin, s in zip(in_channels, strides):
            self.m.append(ParallelOpLayer(cin, out_channel, candidate_op, candidate_ch, 
                          gumbel_op, gumbel_channel,
                          stride=s, act=None, bn=False,
                          independent_ch_arch_param=False,
                          independent_op_arch_param=independent_op_arch_param))
        self.init_arch_param(independent_ch_arch_param, independent_edge_arch_param)
        self.act = get_act(act)
        if self.gumbel_channel: self.bn = nn.ModuleList([nn.BatchNorm2d(int(self.cout*e)) for e in candidate_ch]) if bn else [None for _ in candidate_ch]
        else: self.bn = nn.BatchNorm2d(self.cout) if bn else None

    def init_arch_param(self, ind_ch_alpha, ind_edge_alpha):
        if len(self.candidate_ch) > 1 and ind_ch_arch:
            super().init_arch_param('ch_alphas', len(self.candidate_ch))
        if ind_edge_alpha:
            super().init_arch_param('edge_alphas', len(self.cin))

    def check_valid(self, in_channels, strides):
        assert(len(in_channels)==len(strides))

    def forward(self, xs, op_alphas=None, ch_alphas=None, edge_alpha=None):
        ch_alphas = ch_alphas if ch_alphas is not None else (self.norm_arch_param(self.ch_alphas, self.gumbel_channel) if hasattr(self, 'ch_alphas') else [1.])
        edge_alphas = edge_alphas if edge_alphas is not None else (self.norm_arch_param(self.edge_alphas, self.gumbel_edge) if hasattr(self, 'edge_alphas') else [1.]*len(self.cin))
        bn = self.get_norm_layer(ch_alphas, self.bn, self.gumbel_channel)

        out = 0.
        for x, m, edge_alpha in zip(xs, self.m, edge_alphas):
            out = out + m(x, op_alphas=op_alphas, ch_alphas=ch_alphas) * edge_alpha

        if bn: out = bn(out)
        if self.act: out = self.act(out)

        return out

    @classmethod
    def genotype(self, cfg, op_alphas=None, ch_alphas=None, edge_alpha=None, num_reserved_op=None, num_reserved_ch=None, num_reserved_edge=None):
        num_reserved_op = self.num_reserved_op if num_reserved_op is None else num_reserved_op
        num_reserved_ch = self.num_reserved_ch if num_reserved_ch is None else num_reserved_ch
        assert num_reserved_op==1
        assert num_reserved_ch==1

        new_cfg = deepcopy(cfg)
        # del unused variables
        need_key = inspect.signature(self.outOp.__init__).parameters.keys()
        for k in cfg.keys():
            if k not in need_key: del new_cfg['module_args'][k]

        edge_alphas = edge_alphas if edge_alphas is not None else (self.get_edge_arch_param() if hasattr(self, 'edge_alphas') else None)
        new_cfg['module'] = self.outOp_name
        if edge_alphas is not None:
            edge_alphas_idx = self.get_reserved_idx(num_reserved_edge, edge_alphas)
            new_cfg['input_idx'] = [cfg['input_idx'][idx] for idx in edge_alphas_idx]
            new_cfg['module_args']['ops'], new_cfg['module_args']['strides'] = [], []
            for idx in edge_alphas_idx:
                edge_cfg = self.m[idx].genotype({'module_args': {'candidate_op': candidate_op}}, 
                                     ch_alphas=None)
                new_cfg['module_args']['ops'].append(OP(OPtype=edge_cfg['module'], args=edge_cfg['module_args']))
                new_cfg['module_args']['strides'].append(self.strides[idx])

        ch_alphas = ch_alphas if ch_alphas is not None else (self.get_ch_arch_param() if hasattr(self, 'ch_alphas') else None)
        if ch_alphas is not None:
            ch_alphas_idx = self.get_reserved_idx(num_reserved_ch, ch_alphas)
            new_cfg['module_args']['out_channel'] = cfg['module_args']['out_channel'] * cfg['module_args']['candidate_ch']

        return new_cfg

 
class SPP_search(SearchLayer):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, in_channel, out_channel, kernel=(5, 9, 13)):
        super(SPP_search, self).__init__()
        c_ = in_channel // 2  # hidden channels
        self.cv1 = ConvBNAct_search(in_channel, out_channel, candidate_op=[(1,1)], candidate_ch=[1.], stride=1, act=nn.SiLU(), bn=True, merge_kernel=True)
        self.cv2 = ConvBNAct(c_ * (len(k) + 1), out_channel, kernel=1, dilation=1, stride=1, act=nn.SiLU(), bn=True)

        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in kernel])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))
