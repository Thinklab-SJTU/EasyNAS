import math
from copy import deepcopy
import bisect
from functools import reduce
from itertools import accumulate
from easydict import EasyDict as edict
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import autopad, gumbel_softmax, get_layer, get_act, get_norm
from .base import  OpBuilder, SearchModule
from src.search_space.base import DiscreteSpace, IIDSpace

__all__ = ["ConvBNAct_search", "SepConvBNAct_search", "AFF", "SPP_search"]

def check_nesting(src, least_depth):
    tmp_src = src
    while least_depth > 0:
        if not isinstance(tmp_src, (list, tuple, DiscreteSpace)): break
        least_depth -= 1
        tmp_src = tmp_src[0]
    while least_depth > 0:
        src = [src]
        least_depth -= 1
    return src

class ConvBNAct_search(SearchModule):
    # Mixed Depthwise Conv https://arxiv.org/abs/1907.09595
    def __init__(self, in_channel, out_channel, candidate_op=[(1,1), (3,1), (5,1), (3,2)], candidate_ch=[1.], gumbel_op=False, 
            gumbel_channel=True, 
            stride=1, pad=None, group=1, act=True, act_first=False, bn=dict(submodule_name='torch.nn.BatchNorm2d', args=dict(affine=True)), independent_ch_arch_param=True, independent_op_arch_param=True, bias=False, merge_kernel=True):
        # k=0 means zero op; d=0 means skip-connection
        super(ConvBNAct_search, self).__init__()
        self.merge_kernel = merge_kernel
        self.kd = check_nesting(candidate_op, 2)
        self.candidate_ch = check_nesting(candidate_ch, 1)
        self.stride = stride
        self.group = group
        self.gumbel_op = gumbel_op and len(self.kd)>1
        self.gumbel_channel = gumbel_channel and len(self.candidate_ch)>1
        self.cout = out_channel
        cout_max = int(out_channel * max(self.candidate_ch))

        self.k_max = int(max([(k-1)*d+1 for k, d in self.kd]))
        if merge_kernel:
            self.padding = (self.k_max - 1)//2
            self.weight = self.init_weight(cout_max, in_channel, self.k_max)
            self.bias = self.init_bias(cout_max, self.weight) if bias else None
        else:
            self.weight, self.bias = nn.ParameterList([]), nn.ParameterList([])
            for k, d in self.kd:
                self.weight.append(self.init_weight(cout_max, in_channel, k))
                self.bias.append(self.init_bias(cout_max, self.weight[-1]))

        self.act = get_act(act)
        self.act_first = act_first

        if bn and self.gumbel_channel: self.bn = nn.ModuleList([get_norm(bn, int(self.cout*e)) for e in self.candidate_ch]) 
        else: self.bn = get_norm(bn, cout_max)

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
        else: self.op_alphas = torch.tensor([1.]*len(self.kd))

        if len(self.candidate_ch) > 1 and ind_ch_alpha:
            super().init_arch_parameters('ch_alphas', len(self.candidate_ch))
        else: self.ch_alphas = torch.tensor([1.]*len(self.candidate_ch))

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
            a_e, idx = alphas.max(dim=-1)
            merge_kernel = merge_kernel[:int(self.cout*self.candidate_ch[idx]),:,:,:] * a_e
            if bias is not None: bias = bias[:int(self.cout*e)] 
        else:
            channel_idx = torch.arange(0, Cout, dtype=merge_kernel.dtype, device=merge_kernel.device).long()
#            channel_idx = torch.sort(merge_kernel.view(Cout,-1).sum(dim=-1), descending=True)[1]
            for e, a_e in zip(self.candidate_ch, alphas):
                channel_mask[channel_idx[:int(e*self.cout)]] += a_e
            merge_kernel = merge_kernel * channel_mask.view(-1,1,1,1)
        return merge_kernel, bias

        
    def forward(self, x, op_alphas=None, ch_alphas=None):
        if self.act_first and self.act: x = self.act(x)

        Cin = x.size(1)
        bias = self.bias
        op_alphas = op_alphas if op_alphas is not None else (self.norm_arch_parameters(self.op_alphas, self.gumbel_op) if len(self.op_alphas)>1 else self.op_alphas)
        ch_alphas = ch_alphas if ch_alphas is not None else (self.norm_arch_parameters(self.ch_alphas, self.gumbel_channel) if len(self.ch_alphas)>1 else self.ch_alphas)
        bn = self.get_norm_layer(ch_alphas, self.bn, self.gumbel_channel)
                                   
        merge_kernel = self.get_merge_kernel(self.weight, op_alphas, merge=self.merge_kernel) if len(self.kd)>1 else (self.weight if self.merge_kernel else self.weight[0])

        if Cin != merge_kernel.size(1): merge_kernel = merge_kernel[:,:Cin,:,:]
        if len(self.candidate_ch) > 1:
            merge_kernel, bias = self.deal_merge_kernel_cout(merge_kernel, ch_alphas, self.bias)

        out = torch.nn.functional.conv2d(x, merge_kernel, stride=self.stride, padding=self.padding, dilation=1, groups=self.group)
        out = out + bias.view(1,-1,1,1) if bias is not None else out
        out = bn(out) if bn is not None else out
        if (not self.act_first) and self.act: 
            out = self.act(out)
        return out

    def discretize(self, cfg=None, op_alphas=None, ch_alphas=None, edge_alphas=None, num_reserved_op=1, num_reserved_edge=None):
        assert num_reserved_op==1

        new_cfg = self.init_output_yaml(cfg, outOp_name=None, input_idx=-1)
        if ch_alphas is None: ch_alphas = getattr(self, 'ch_alphas', None)
        if ch_alphas is not None:
            ch_alphas_idx = self.get_reserved_idx(1, ch_alphas)[0]
            cout = cfg['args'].get('out_channel', self.cout)
            new_cfg['args']['out_channel'] = cout * cfg['args']['candidate_ch'][ch_alphas_idx]

        if op_alphas is None: op_alphas = self.op_alphas
        if op_alphas is not None:
            op_alphas_idx = self.get_reserved_idx(num_reserved_op, op_alphas)[0]
            new_cfg['args']['kernel'], new_cfg['args']['dilation'] = cfg['args']['candidate_op'][op_alphas_idx]
        return new_cfg

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        weight_name = prefix + 'weight'
        bias_name = prefix + 'bias'
        search_space = state_dict.pop(prefix+'search_space', None)
        if search_space is None or 'kd' not in search_space:
            cout, cin, _, _ = self.weight.shape
            state_dict[weight_name] = state_dict[weight_name][:cout, :cin, :, :]
            if bias_name in state_dict:
                state_dict[bias_name] = state_dict[bias_name][:cout]
            return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

        ss_kd = [list(kd) for kd in search_space['kd']]
        op_idx = [ss_kd.index(list(kd)) for kd in self.kd]
        state_dict_op_alphas = state_dict.pop(prefix+'op_alphas', torch.tensor([1.]*len(ss_kd)))
#        for idx in set(range(len(ss_kd)))-set(op_idx): state_dict_op_alphas[idx].copy_(0.)
        state_dict_op_alphas = torch.gather(state_dict_op_alphas, dim=-1, index=torch.tensor(op_idx))
        state_dict_op_alphas = state_dict_op_alphas / state_dict_op_alphas.sum()
        state_dict.pop(prefix+'ch_alphas', None)
        if self.merge_kernel:
            assert weight_name in state_dict
            weight = state_dict[weight_name]
            start = int((weight.shape[-1] - self.k_max) / 2)
            end = int(weight.shape[-1] - start)
            cout, cin, _, _ = self.weight.shape
            weight = self.get_merge_kernel(weight[:cout,:cin,start:end, start:end], state_dict_op_alphas, self.merge_kernel)
            state_dict[weight_name] = weight[:cout,:cin,:, :]
            if self.bias is not None:
                state_dict[bias_name] = state_dict[bias_name][:cout]
        else:
            weights = [state_dict.pop(weight_name + '.%s'%str(i)) for i in range(len(ss_kd))]
            biases = [state_dict.pop(bias_name + '.%s'%str(i)) for i in range(len(ss_kd))] if self.bias is not None else None
            for i, op_i in enumerate(op_idx):
                cout, cin, _, _ = self.weight[i].shape
                state_dict[weight_name+'.%s'%i] = weights[op_i][:cout, :cin, :, :]
                if biases is not None:
                    state_dict[bias_name+'.%s'%i] = biases[op_i][:cout]
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        


class SepConvBNAct_search(ConvBNAct_search):
    def init_weight(self, cout, cin, kernel):
        kernel = [kernel, kernel] if isinstance(kernel, int) else kernel
        point_w = torch.Tensor(cout, cin, 1, 1)
        torch.nn.init.kaiming_normal_(point_w, mode='fan_in')
        depth_w = torch.Tensor(cin, 1, *kernel)
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

#    def get_merge_kernel(self, depth_weight, alphas, merge=True):
#        merge_kernel = 0.
#        if merge:
#            for i, alpha in enumerate(alphas):
#                k,d = self.kd[i]
#                tmp_ks = (k-1)*d + 1
#                start = int((self.k_max - tmp_ks) / 2)
#                end = int(self.k_max - start)
#                if d == 1:
#                    w_pad = torch.nn.functional.pad(depth_weight[:,:,start:end, start:end], (start,)*4, "constant", value=0)
#                    merge_kernel += w_pad * alpha
#                else:
#                    w = torch.zeros_like(depth_weight)
#                    w[:,:,start:end:d, start:end:d] = depth_weight[:,:,start:end:d, start:end:d]
#                    merge_kernel += w * alpha
#        else:
#            raise(ValueError("weight cannot be merged in SepConv if merge_kernel is False"))
#        return merge_kernel

    def forward(self, x, op_alphas=None, ch_alphas=None):
        x = self.act(x) if self.act_first and self.act is not None else x

        Cin = x.size(1)
        bias = self.bias
        op_alphas = op_alphas if op_alphas is not None else (self.norm_arch_parameters(self.op_alphas, self.gumbel_op) if len(self.op_alphas)>1 else self.op_alphas)
        ch_alphas = ch_alphas if ch_alphas is not None else (self.norm_arch_parameters(self.ch_alphas, self.gumbel_channel) if len(self.ch_alphas)>1 else self.ch_alphas)
        bn = self.get_norm_layer(ch_alphas, self.bn, self.gumbel_channel)

        if self.merge_kernel:
            merge_kernel = self.get_merge_kernel(self.weight['depth_weight'], op_alphas, merge=True) if len(self.kd)>1 else self.weight['depth_weight']
            if Cin != merge_kernel.size(1): merge_kernel = merge_kernel[:Cin,:,:,:]
            out = torch.nn.functional.conv2d(x, merge_kernel, stride=self.stride, padding=self.padding, dilation=1, groups=Cin)
            # out channel for point-wise conv
            point_weight = self.weight['point_weight']
            if len(self.candidate_ch) > 1:
                point_weight, bias = self.deal_merge_kernel_cout(point_weight, ch_alphas, self.bias)
            if Cin != point_weight.size(1): point_weight = point_weight[:,:Cin,:,:]
            out = torch.nn.functional.conv2d(out, point_weight, stride=1, padding=0, dilation=1, groups=self.group)
            out = out + bias.view(1,-1,1,1) if bias is not None else out
        else:
            out = 0.
            for i, weight in enumerate(self.weight):
                depth_weight = weight['depth_weight']
                point_weight = weight['point_weight']
                if Cin != depth_weight.size(1): depth_weight = depth_weight[:Cin,:,:,:]
                tmp_out = torch.nn.functional.conv2d(x, depth_weight, stride=self.stride, padding=self.padding, dilation=1, groups=Cin)
                if len(self.candidate_ch) > 1:
                    point_weight, bias = self.deal_merge_kernel_cout(point_weight, ch_alphas, self.bias)
                if Cin != point_weight.size(1): point_weight = point_weight[:,:Cin,:,:]
                tmp_out = torch.nn.functional.conv2d(tmp_out, point_weight, stride=1, padding=0, dilation=1, groups=self.group)
                out += op_alphas[i] * (tmp_out + bias.view(1,-1,1,1) if bias is not None else tmp_out)

        out = bn(out) if bn is not None else out
        out = self.act(out) if not self.act_first and self.act is not None else out
        return out

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        weight_name = prefix + 'weight'
        bias_name = prefix + 'bias'
        search_space = state_dict.pop(prefix+'search_space', None)
        if search_space is None or 'kd' not in search_space:
            cout, cin, _, _ = self.weight['point_weight'].shape
            state_dict[weight_name+'.depth_weight'] = state_dict[weight_name+'.depth_weight'][:cin,:,start:end, start:end]
            state_dict[weight_name+'.point_weight'] = state_dict[weight_name+'.point_weight'][:cout,:cin,:,:]
            if bias_name in state_dict:
                state_dict[bias_name] = state_dict[bias_name][:cout]
            return super(ConvBNAct_search, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

        ss_kd = [list(kd) for kd in search_space['kd']]
        op_idx = [ss_kd.index(list(kd)) for kd in self.kd]
        state_dict_op_alphas = state_dict.pop(prefix+'op_alphas', torch.tensor([1.]*len(ss_kd)))
#        for idx in set(range(len(ss_kd)))-set(op_idx): state_dict_op_alphas[idx].copy_(0.)
        state_dict_op_alphas = torch.gather(state_dict_op_alphas, dim=-1, index=torch.tensor(op_idx))
        state_dict_op_alphas = state_dict_op_alphas / state_dict_op_alphas.sum()
        state_dict.pop(prefix+'ch_alphas', None)
        if self.merge_kernel:
            depth_weight = state_dict[weight_name+'.depth_weight']
            start = int((depth_weight.shape[-1] - self.k_max) / 2)
            end = int(depth_weight.shape[-1] - start)
            cout, cin, _, _ = self.weight['point_weight'].shape
            depth_weight = self.get_merge_kernel(depth_weight[:cin,:,start:end,start:end], state_dict_op_alphas, self.merge_kernel)
            state_dict[weight_name+'.depth_weight'] = depth_weight[:cin,:,:,:]
            state_dict[weight_name+'.point_weight'] = state_dict[weight_name+'.point_weight'][:cout,:cin,:,:]
            if self.bias is not None:
                state_dict[bias_name] = state_dict[bias_name][:cout]
        else:
            depth_weights = [state_dict.pop(weight_name + '.%d.depth_weight'%i) for i in range(len(ss_kd))]
            point_weights = [state_dict.pop(weight_name + '.%d.point_weight'%i) for i in range(len(ss_kd))]
            biases = [state_dict.pop(bias_name + '.%d'%i) for i in range(len(ss_kd))] if self.bias is not None else None
            for i, op_i in enumerate(op_idx):
                cout, cin, _, _ = self.weight[i]['point_weight'].shape
                state_dict[weight_name+'.%d.depth_weight'%i] = depth_weights[op_i][:cin, :, :, :]
                state_dict[weight_name+'.%d.point_weight'%i] = point_weights[op_i][:cout, :cin, :, :]
                if biases is not None:
                    state_dict[bias_name+'.%s'%i] = biases[op_i][:cout]
        return super(ConvBNAct_search, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


class AFF(SearchModule):
    # Auto-Feature Fusion
    #self.adjust_ch_op = edict(submodule_name='ConvBNAct_search', args=dict(candidate_op=[(1,1)], candidate_ch=candidate_ch, gumbel_channel=gumbel_channel, stride=1, bn=False, act=None, independent_ch_arch_param=False))
    def __init__(self, in_channel, out_channel, strides, input_idx,
    candidate_op, gumbel_op=False, 
    auto_refine=False, adjust_ch_op=None, upsample_op=None, 
    candidate_ch=[1.], gumbel_channel=True, 
    gumbel_edge=False, 
    act=nn.ReLU(), bn=dict(submodule_name='torch.nn.BatchNorm2d', args=dict(affine=True)), 
    independent_ch_arch_param=True, independent_op_arch_param=True, independent_edge_arch_param=True):
        """
        strides: a list indicating the scale for each edge. Whether to up-sampling or down-sampling, and how much the degree is
        """
        super(AFF, self).__init__()
        in_channel = check_nesting(in_channel, 1)
        strides = check_nesting(strides, 1)

        self.check_valid(in_channel, strides)
        self.cin = in_channel
        self.cout = out_channel
        self.strides = strides
        self.input_idx = input_idx
        self.candidate_op = check_nesting(candidate_op, 2)
        if len(self.candidate_op) == 1:
            self.candidate_op = [self.candidate_op[0]] * len(in_channel)
        assert len(self.candidate_op) == len(in_channel)
        self.candidate_ch = check_nesting(candidate_ch, 1)
        self.gumbel_op = gumbel_op
        self.gumbel_channel = gumbel_channel and len(self.candidate_ch)>1
        self.gumbel_edge = gumbel_edge

        op_builder = OpBuilder(
              auto_refine=auto_refine,
              adjust_ch_op=adjust_ch_op,
              upsample_op=upsample_op
        )
        self.m = nn.ModuleList([])
        for ei, (cin, s) in enumerate(zip(in_channel, strides)):
            self.m.append(op_builder.build_parallel_op(self.candidate_op[ei], cin, out_channel, s))
        #TODO: compute num_alphas_each_op for each edge
        self.num_alphas_each_op = []
        for op in self.candidate_op[0]:
            self.num_alphas_each_op.append(
                 len(op.args['candidate_op']) if hasattr(op, 'args') and hasattr(op.args, 'candidate_op') else -1)
        self.num_op_alphas = sum(abs(x) for x in self.num_alphas_each_op)
        self.init_arch_parameters(independent_op_arch_param, independent_ch_arch_param, independent_edge_arch_param)

        self.act = get_act(act)
        if self.gumbel_channel: self.bn = nn.ModuleList([get_norm(bn, int(self.cout*e)) for e in self.candidate_ch]) 
        else: self.bn = get_norm(bn, self.cout)

    def init_arch_parameters(self, ind_op_alpha, ind_ch_alpha, ind_edge_alpha):
        if self.num_op_alphas > 1 and ind_op_alpha:
            super().init_arch_parameters('op_alphas', len(self.cin), self.num_op_alphas)
        else: self.op_alphas = torch.tensor([[1.]*len(cand_edge_op) for cand_edge_op in self.candidate_op])
        if len(self.candidate_ch) > 1 and ind_ch_alpha:
            super().init_arch_parameters('ch_alphas', len(self.candidate_ch))
        else: self.ch_alphas = torch.tensor([1.]*len(self.candidate_ch))
        if len(self.cin) > 1 and ind_edge_alpha:
            super().init_arch_parameters('edge_alphas', len(self.cin))
        else: self.edge_alphas = torch.tensor([1.]*len(self.cin))

    def check_valid(self, in_channel, strides):
        assert(len(in_channel)==len(strides))

    def forward_edge(self, x, edge_module, op_alphas, ch_alphas):
        out, ptr = 0., 0
        for idx, (op, num_alphas_each_op) in enumerate(zip(edge_module, self.num_alphas_each_op)):
            if num_alphas_each_op > 0: 
                end_ptr = ptr + num_alphas_each_op
                if isinstance(op, nn.Sequential):
                    tmp = x
                    for sub_op in op:
                        if isinstance(sub_op, SearchModule):
                            tmp = sub_op(tmp, op_alphas=op_alphas[ptr:end_ptr], ch_alphas=ch_alphas)
                        else: tmp = sub_op(tmp)
                    out = out + tmp
                else:
                    out = out + op(x, op_alphas=op_alphas[ptr:end_ptr], ch_alphas=ch_alphas)
                ptr = end_ptr
            else: 
                if op_alphas[ptr] > 0:
                    out = out + op_alphas[ptr] * op(x)
                ptr += 1

        return out

    def forward(self, xs, op_alphas=None, ch_alphas=None, edge_alphas=None):
        op_alphas = op_alphas if op_alphas is not None else (self.norm_arch_parameters(self.op_alphas, self.gumbel_op) if self.op_alphas.shape[-1]>1 else self.op_alphas)
        ch_alphas = ch_alphas if ch_alphas is not None else (self.norm_arch_parameters(self.ch_alphas, self.gumbel_channel) if len(self.ch_alphas)>1 else self.ch_alphas)
        edge_alphas = edge_alphas if edge_alphas is not None else (self.norm_arch_parameters(self.edge_alphas, self.gumbel_edge) if len(self.edge_alphas)>1 else self.edge_alphas)
        bn = self.get_norm_layer(ch_alphas, self.bn, self.gumbel_channel)

        out = sum(self.forward_edge(x, m, edge_op_alphas, ch_alphas) * edge_alpha 
                for x, m, edge_alpha, edge_op_alphas in zip(xs, self.m, edge_alphas, op_alphas))

        if bn: out = bn(out)
        if self.act: out = self.act(out)

        return out
    
    def discretize_edge(self, edge_module, op_alphas, candidate_op, num_reserved_op=1, exclude_alpha_idx=[]):
        assert num_reserved_op == 1
        op_alphas_idx = self.get_reserved_idx(min(num_reserved_op+len(exclude_alpha_idx), len(op_alphas)), op_alphas)
        op_alphas_idx = [idx for idx in op_alphas_idx if idx not in exclude_alpha_idx]
#        num_alphas_before = reduce(lambda x,y: x+[x[-1]+abs(y)] if isinstance(x, list) else [abs(x),abs(x)+abs(y)], self.num_alphas_each_op)
        num_alphas_before = list(accumulate(abs(x) for x in self.num_alphas_each_op))
        op_idx = [bisect.bisect_right(num_alphas_before, x) for x in op_alphas_idx]
        op_idx, op_alphas_idx = op_idx[0], op_alphas_idx[0]
        if self.num_alphas_each_op[op_idx] > 0: # (Sep)ConvBNAct_search
            select_op = deepcopy(candidate_op[op_idx])
            if isinstance(edge_module[op_idx], SearchModule): 
                return edge_module[op_idx].discretize(select_op, op_alphas=op_alphas[(0 if op_idx==0 else num_alphas_before[op_idx-1]):num_alphas_before[op_idx]], ch_alphas=None, edge_alphas=None, num_reserved_op=num_reserved_op)
            else:
                layer_cfg = []
                select_op_iter = iter([select_op] if isinstance(select_op, (dict, IIDSpace)) else select_op)
                tmp_cfg = next(select_op_iter)
                tmp_module = get_layer(tmp_cfg['submodule_name']) 
                for sub_m_idx, sub_m in enumerate(edge_module[op_idx]):
                    if isinstance(sub_m, tmp_module):
                        layer_cfg.append(
                                sub_m.discretize(select_op, op_alphas=op_alphas[(0 if op_idx==0 else num_alphas_before[op_idx-1]):num_alphas_before[op_idx]], ch_alphas=None, edge_alphas=None, num_reserved_op=num_reserved_op) if isinstance(sub_m, SearchModule) else tmp_cfg)
                        try:
                            tmp_cfg = next(select_op_iter)
                        except StopIteration as e:
                            break
                        tmp_module = get_layer(tmp_cfg['submodule_name']) 
                return layer_cfg
        else:
            return deepcopy(candidate_op[op_idx])

    def get_exclude_op_idx(self, parallel_op, exclude_ops):
        num_alphas_before = list(accumulate(abs(x) for x in self.num_alphas_each_op))
        exclude_op_idx, exclude_alpha_idx = [], []
        for i in range(len(parallel_op)-1, -1, -1):
            sequence_op = parallel_op[i] 
            if isinstance(sequence_op, (dict, IIDSpace)): sequence_op = [sequence_op]
            for op in sequence_op:
                if op['submodule_name'] in exclude_ops: 
                    exclude_op_idx.append(i)
                    exclude_alpha_idx += list(range((0 if i==0 else num_alphas_before[i-1]), num_alphas_before[i]))
                    break
        return exclude_op_idx, exclude_alpha_idx

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
        op_alphas = F.softmax(op_alphas, dim=-1).detach()
#        exclude_op_idx, exclude_alpha_idx = self.get_exclude_op_idx(self.candidate_op, ['Zero']) 
        exclude_alpha_idx = []
        for c_op in self.candidate_op:
            exclude_alpha_idx.append(self.get_exclude_op_idx(c_op, ['Zero'])[1])

        edge_alphas = getattr(self, 'edge_alphas', None) if edge_alphas is None else edge_alphas
        if edge_alphas is None: 
            edge_alphas_idx = sorted(range(op_alphas.shape[0]), key=lambda x: -max(op_alphas[x][k] for k in range(len(op_alphas[x])) if k not in exclude_alpha_idx[x]))[:num_reserved_edge]
        else:
            edge_alphas_idx = self.get_reserved_idx(num_reserved_edge, edge_alphas)

        args['ops'], args['strides'] = [], []
        for idx in edge_alphas_idx:
            edge_op = self.discretize_edge(self.m[idx], op_alphas[idx], self.candidate_op[idx], num_reserved_op, exclude_alpha_idx[idx])
            args['ops'].append(edge_op)
            args['strides'].append(self.strides[idx])

        input_idx = getattr(self, 'input_idx', None)
        if cfg is not None:
            input_idx = cfg.get('input_idx', input_idx)
        input_idx = edge_alphas_idx if input_idx is None else [input_idx[ei] for ei in edge_alphas_idx]
        new_cfg = self.init_output_yaml(cfg, outOp_name='FuseLayer', input_idx=input_idx, **args)
        return new_cfg

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        search_space = state_dict.pop(prefix+'search_space')
        ss_op = [list(edge_op) for edge_op in search_space['candidate_op']]
        if len(ss_op) == 1: ss_op = ss_op * len(search_space['strides'])
        assert len(ss_op) == len(search_space['strides'])
        # keep op weights in the selected edges
        edge_idx = [search_space['input_idx'].index(idx) for idx in self.input_idx]
        #TODO: How to get the selected op, since the item of candidate_op can also be ConvBNAct_search?
#        op_idx = [[ss_op[ori_ei].index(self.candidate_op[ei])] for ei, ori_ei in enumerate(edge_idx)]
        edge_prefix = prefix + 'm'
        tmp_state_dict = {}
        keys = list(state_dict.keys())
        for key in keys:
            weight = state_dict.pop(key)
            for ei, ori_ei in enumerate(edge_idx):
                if key.startswith(edge_prefix+'.%d'%ori_ei): 
                    tmp_state_dict[key.replace(edge_prefix+'.%d'%ori_ei, edge_prefix+'.%d'%ei, 1)] = weight
            #TODO: How to deal with op_alphas?
#            if key.startswith(prefix+'op_alphas'):
#                op_alphas = torch.gather(weight, dim=0, index=torch.tensor(edge_idx))
#                tmp_state_dict[key] = op_alphas / op_alphas.sum(dim=-1, keepdim=True)
            if key.startswith(prefix+'edge_alphas'):
                edge_alphas = torch.gather(weight, dim=-1, index=torch.tensor(edge_idx))
                tmp_state_dict[key] = edge_alphas / edge_alphas.sum(dim=-1, keepdim=True)

        state_dict.update(tmp_state_dict)
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

 
class SPP_search(SearchModule):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, in_channel, out_channel, kernels=(5, 9, 13), bn=torch.nn.BatchNorm2d, act=nn.SiLU):
        super(SPP_search, self).__init__()
        c_ = in_channel // 2  # hidden channels
        self.cv1 = ConvBNAct_search(in_channel, c_, candidate_op=[(1,1)], candidate_ch=[1.], stride=1, act=act, bn=bn, merge_kernel=True)
        self.cv2 = ConvBNAct_search(c_ * (len(kernels) + 1), out_channel, candidate_op=[(1,1)], candidate_ch=[1.], stride=1, act=act, bn=bn, merge_kernel=True)

        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in kernels])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))

    def discretize(self, cfg=None):
        new_cfg = self.init_output_yaml(cfg, outOp_name='SPP')
        return new_cfg

