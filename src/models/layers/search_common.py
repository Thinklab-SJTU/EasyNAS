from copy import deepcopy
import bisect
from functools import reduce
from itertools import accumulate
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import autopad, gumbel_softmax, get_layer, get_act
from .base import  OP_CFG, OpBuilder, SearchModule

__all__ = ["ConvBNAct_search", "SepConvBNAct_search", "AFF", "SPP_search"]


class SearchLayer(SearchModule):
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

    def init_arch_param(self, arch_name, *shape):
        self.register_buffer(arch_name, torch.autograd.Variable(1e-3*torch.randn(*shape), requires_grad=True))

    def set_arch_param(self, arch_name, arch_param):
        if hasattr(self, arch_name): delattr(self, arch_name)
        setattr(self, arch_name, arch_param)

    def get_arch_param(self):
        out = []
        out.extend(self.get_op_arch_param())
        out.extend(self.get_ch_arch_param())
        out.extend(self.get_edge_arch_param())
        return out

    def get_op_arch_param(self):
        out = getattr(self, op_arch_param, None)
        return out

    def get_ch_arch_param(self):
        out = getattr(self, ch_arch_param, None)
        return out

    def get_edge_arch_param(self):
        out = getattr(self, edge_arch_param, None)
        return out

    def norm_arch_param(self, alphas, gumbel=False):
        return gumbel_softmax(F.log_softmax(alphas, dim=-1), hard=True) if gumbel else nn.functional.softmax(alphas, dim=-1)

    def init_output_yaml(cls, arch_yaml, outOp_name=None):
        new_arch = deepcopy(arch_yaml)
        if outOp_name is not None:
            outOp = get_layer(outOp_name)
        else:
            outOp_name, outOp = cls.outOp_name, cls.outOp
        new_arch['submodule_name'] = outOp_name
        # del unused variables
        need_key = inspect.signature(outOp.__init__).parameters.keys()
        for k in arch_yaml.keys():
            if k not in need_key: del new_arch['args'][k]
        return new_arch

    def get_reserved_idx(self, num_reserved, weight):
#        return weight.argmax(dim=-1).item() if num_reserved==1 else [x.item() for x in torch.topk(weight, k=num_reserved, dim=-1)[1]]
        return [x.item() for x in torch.topk(weight, k=num_reserved, dim=-1)[1]]

    @classmethod
    def genotype(cls, cfg, op_alphas, ch_alphas, edge_alpha, num_reserved_op=1, num_reserved_edge=2):
        raise(NotImplementedError(f"No Implementation of genotype func for {cls}"))


class ConvBNAct_search(SearchModule):
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

        self.init_arch_parameters(independent_ch_arch_param, independent_op_arch_param)

    
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

    def init_arch_parameters(self, ind_ch_alpha, ind_op_alpha):
        if len(self.kd) > 1 and ind_op_alpha:
            super().init_arch_parameters('op_alphas', len(self.kd))

        if len(self.candidate_ch) > 1 and ind_ch_alpha:
            super().init_arch_parameters('ch_alphas', len(self.candidate_ch))

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
        op_alphas = op_alphas if op_alphas is not None else (self.norm_arch_parameters(self.op_alphas, self.gumbel_op) if hasattr(self, 'op_alphas') else [1.])
        ch_alphas = ch_alphas if ch_alphas is not None else (self.norm_arch_parameters(self.ch_alphas, self.gumbel_channel) if hasattr(self, 'ch_alphas') else [1.])
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

    def discretize(self, cfg=None, op_alphas=None, ch_alphas=None, edge_alpha=None, num_reserved_op=1, num_reserved_edge=None):
        assert num_reserved_op==1

        new_cfg = self.init_output_yaml(cfg)

        if ch_alphas is None: ch_alphas = self.ch_alphas
        if ch_alphas is not None:
            ch_alphas_idx = self.get_reserved_idx(1, ch_alphas)[0]
            new_cfg['args']['out_channel'] = cfg['args']['out_channel'] * cfg['args']['candidate_ch'][ch_alphas_idx]

        if op_alphas is None: op_alphas = self.op_alphas
        if op_alphas is not None:
            op_alphas_idx = self.get_reserved_idx(num_reserved_op, op_alphas)[0]
            new_cfg['args']['kernel'], new_cfg['args']['dilation'] = cfg['args']['candidate_op'][op_alphas_idx]
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
        op_alphas = op_alphas if op_alphas is not None else (self.norm_arch_parameters(self.op_alphas, self.gumbel_op) if hasattr(self, 'op_alphas') else [1.])
        ch_alphas = ch_alphas if ch_alphas is not None else (self.norm_arch_parameters(self.ch_alphas, self.gumbel_channel) if hasattr(self, 'ch_alphas') else [1.])
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



class AFF(SearchModule):
    # Auto-Feature Fusion
    #self.adjust_ch_op = OP_CFG(submodule_name='ConvBNAct_search', args=dict(candidate_op=[(1,1)], candidate_ch=candidate_ch, gumbel_channel=gumbel_channel, stride=1, bn=False, act=None, independent_ch_arch_param=False))
    def __init__(self, in_channel, out_channel, strides, 
    candidate_op, gumbel_op=False, 
    auto_refine=False, adjust_ch_op=None, up_sample_op=None, 
    candidate_ch=[1.], gumbel_channel=True, 
    gumbel_edge=False, 
    act=nn.ReLU(), bn=True, 
    independent_ch_arch_param=True, independent_op_arch_param=True, independent_edge_arch_param=True):
        """
        strides: a list indicating the scale for each edge. Whether to up-sampling or down-sampling, and how much the degree is
        """
        super(AFF, self).__init__()

        self.check_valid(in_channel, strides)
        self.cin = in_channel
        self.cout = out_channel
        self.strides = strides
        self.candidate_ch = candidate_ch
        self.gumbel_op = gumbel_op
        self.gumbel_channel = gumbel_channel and len(candidate_ch)>1
        self.gumbel_edge = gumbel_edge
        self.candidate_op = candidate_op
        self.candidate_ch = candidate_ch

        op_builder = OpBuilder(
              auto_refine=auto_refine,
              adjust_ch_op=adjust_ch_op,
              upsample_op=up_sample_op
        )
        self.m = nn.ModuleList([])
        for cin, s in zip(in_channel, strides):
            self.m.append(op_builder.build_op(candidate_op, cin, out_channel, s))
#            self.m.append(ParallelOpLayer(cin, out_channel, candidate_op, candidate_ch, 
#                          gumbel_op, gumbel_channel,
#                          stride=s, act=None, bn=False,
#                          independent_ch_arch_param=False,
#                          independent_op_arch_param=independent_op_arch_param))
        self.init_arch_parameters(independent_op_arch_param, independent_ch_arch_param, independent_edge_arch_param)
        self.num_alphas_each_op = []
        for op in candidate_op:
            self.num_alphas_each_op.append(
                 len(op.args['candidate_op']) if hasattr(op.args, 'candidate_op') else -1)
        self.num_op_alphas = sum(abs(x) for x in self.num_alphas_each_op)

        self.act = get_act(act)
        if self.gumbel_channel: self.bn = nn.ModuleList([nn.BatchNorm2d(int(self.cout*e)) for e in candidate_ch]) if bn else [None for _ in candidate_ch]
        else: self.bn = nn.BatchNorm2d(self.cout) if bn else None

    def init_arch_parameters(self, ind_op_alpha, ind_ch_alpha, ind_edge_alpha):
        if len(self.candidate_op) > 1 and ind_op_alpha:
            super().init_arch_parameters('op_alphas', len(self.cin), len(self.candidate_op))
        if len(self.candidate_ch) > 1 and ind_ch_alpha:
            super().init_arch_parameters('ch_alphas', len(self.candidate_ch))
        if len(self.cin) > 1 and ind_edge_alpha:
            super().init_arch_parameters('edge_alphas', len(self.cin))

    def check_valid(self, in_channel, strides):
        assert(len(in_channel)==len(strides))

    def forward_edge(self, x, edge_module, op_alphas, ch_alphas):
        out, ptr = 0., 0
        for idx, (op, num_alphas_each_op) in enumerate(zip(edge_module, self.num_alphas_each_op)):
            if num_alphas_each_op > 0: 
                end_ptr = ptr + num_alphas_each_op
                out = out + op(x, op_alphas=op_alphas[ptr:end_ptr], ch_alphas=ch_alphas)
                ptr = end_ptr
            else: 
                out = out + op_alphas[ptr] * op(x)
                ptr += 1

        return out

    def forward(self, xs, op_alphas=None, ch_alphas=None, edge_alphas=None):
        op_alphas = op_alphas if op_alphas is not None else (self.norm_arch_parameters(self.op_alphas, self.gumbel_op) if hasattr(self, 'op_alphas') else [1.])
        ch_alphas = ch_alphas if ch_alphas is not None else (self.norm_arch_parameters(self.ch_alphas, self.gumbel_channel) if hasattr(self, 'ch_alphas') else [1.])
        edge_alphas = edge_alphas if edge_alphas is not None else (self.norm_arch_parameters(self.edge_alphas, self.gumbel_edge) if hasattr(self, 'edge_alphas') else [1.]*len(self.cin))
        bn = self.get_norm_layer(ch_alphas, self.bn, self.gumbel_channel)

        out = 0.
        for x, m, edge_alpha, edge_op_alphas in zip(xs, self.m, edge_alphas, op_alphas):
            out = out + self.forward_edge(x, m, edge_op_alphas, ch_alphas) * edge_alpha

        if bn: out = bn(out)
        if self.act: out = self.act(out)

        return out

    def discretize_edge(self, op_alphas, num_reserved_op=1):
        assert num_reserved_op == 1
        op_alphas_idx = self.get_reserved_idx(num_reserved_op, op_alphas)[0]
#        num_alphas_before = reduce(lambda x,y: x+[x[-1]+abs(y)] if isinstance(x, list) else [abs(x),abs(x)+abs(y)], self.num_alphas_each_op)
        num_alphas_before = list(accumulate(abs(x) for x in self.num_alphas_each_op))
        op_idx = bisect.bisect_right(num_alphas_before, op_alphas_idx)
        if self.num_alphas_each_op[op_idx] > 0: # (Sep)ConvBNAct_search
            select_op = self.candidate_op[op_idx]
            layer_cfg = self.get_layer(select_op.Optype).discretize(select_op.args, op_alphas=op_alphas, ch_alphas=None, edge_alphas=None, num_reserved_op=num_reserved_op)
            return OP_CFG(submodule_name=layer_cfg['submodule_name'], args=layer_cfg['args'])
        else:
            return self.candidate_op[op_idx]

    def discretize(self, cfg=None, op_alphas=None, ch_alphas=None, edge_alphas=None, num_reserved_op=1, num_reserved_edge=2):
        assert num_reserved_op==1

        args = {}
        ch_alphas = getattr(self, 'ch_alphas', None) if ch_alphas is None else ch_alphas
        if ch_alphas is not None:
            ch_alphas_idx = self.get_reserved_idx(1, ch_alphas)[0]
            if cfg is not None:
                args['out_channel'] = cfg['args']['out_channel'] * cfg['args']['candidate_ch'][ch_alphas_idx]
            else:
                args['out_channel'] = self.cout * self.candidate_ch[ch_alphas_idx]

        if op_alphas is None: op_alphas = self.op_alphas
        if edge_alphas is None: edge_alphas = self.edge_alphas
        if edge_alphas is not None:
            edge_alphas_idx = self.get_reserved_idx(num_reserved_edge, edge_alphas)
        else: edge_alphas_idx = list(range(len(op_alphas)))
        args['ops'], args['strides'] = [], []
        for idx in edge_alphas_idx:
            edge_op = self.discretize_edge(op_alphas[idx], num_reserved_op)
            args['ops'].append(edge_op)
            args['strides'].append(self.strides[idx])
        new_cfg = self.init_output_yaml(cfg, outOp_name='FuseLayer', input_idx=edge_alphas_idx, **args)
        return new_cfg

 
class SPP_search(SearchModule):
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
