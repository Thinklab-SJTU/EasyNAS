from abc import ABC,abstractmethod
import math
import inspect
from copy import deepcopy
import bisect
from functools import reduce
from itertools import accumulate
from easydict import EasyDict as edict
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import autopad, gumbel_softmax, get_layer, get_act, get_norm, get_layer
from .base import  OpBuilder
from src.search_space.base import DiscreteSpace, IIDSpace, _SearchSpace, RepeatSpace
from .common import DropPath

__all__ = ["ConvBNAct_search", "SepConvBNAct_search", "AFF", "SPP_search"]

def check_nesting(src, least_depth, nest_type=(list, tuple, DiscreteSpace, RepeatSpace)):
    tmp_src = src
    while least_depth > 0:
        if not isinstance(tmp_src, nest_type): break
        least_depth -= 1
        tmp_src = tmp_src[0]
    while least_depth > 0:
        src = [src]
        least_depth -= 1
    return src

class SearchModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SearchModule, self).__init__()

    def norm_arch_parameters(self, search_space, arch_param=None, device='cuda'):
        if arch_param is not None: return arch_param
        if isinstance(search_space, DiscreteSpace):
            arch_param = list(search_space.sampler_weights())
            if len(arch_param) == 0:
                return torch.tensor([1./search_space.size for _ in range(search_space.size)], device=device)
            assert len(arch_param) == 1
            return search_space.sampler.norm_fn(arch_param[0])
        else: return torch.tensor([1./len(search_space) for _ in range(len(search_space))], device=device)

    def get_norm_layer(self, ch_alphas, bn, bn_per_ch=True):
        return bn[ch_alphas.argmax()] if bn_per_ch and isinstance(bn, nn.ModuleList) else bn

    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        destination = super(SearchModule, self).state_dict(*args, destination, prefix, keep_vars)
        search_space, sampler = {}, {}
        for k, v in vars(self).items():
            if isinstance(v, _SearchSpace):
                search_space[k] = v
                sampler[v.label] = v.sampler
            #TODO: we should reserve candidate_ch to make sure load multiple bn weight from nn.ModuleList. Maybe after we build multiple bn by SearchModule, there will be no needs to reserve candidate_ch 
            elif k == 'candidate_ch' and len(v) > 1:
                search_space[k] = v
        if len(search_space) > 0:
            destination[prefix+'search_space'] = search_space
        if len(sampler) > 0:
            destination[prefix+'search_space_sampler'] = sampler

#        destination[prefix+'search_space'] = {k:v for k, v in vars(self).items() if isinstance(v, _SearchSpace)}
        return destination

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        # deal with search space
        search_space = state_dict.pop(prefix+'search_space', {})
        for k, v in search_space.items():
            this_v = getattr(self, k)
            if isinstance(this_v, _SearchSpace) and this_v.size == v.size:
                setattr(self, k, v)

        # deal with sampler in search space
        #TODO: when the current search space differs from the search space in the state_dict
        sampler = state_dict.pop(prefix+'search_space_sampler', {})
        for label, s in sampler.items():
            if label not in _SearchSpace._samplers:
                _SearchSpace._samplers[label] = s

        #TODO: Maybe we should build bn as SearchModule, then we donot have to specifically deal with bn weights.
        # deal with bn
        def _bn_weight(ss_bn_prefix, bn_prefix, bn_state_dict, cout):
            for key in bn_state_dict.keys():
                if key.startswith(ss_bn_prefix):
                    state_dict[key.replace(ss_bn_prefix, bn_prefix)] = bn_state_dict[key][:cout] if bn_state_dict[key].dim() == 1 else bn_state_dict[key]

        ss_candidate_ch = search_space.get('candidate_ch', None)
        for k, v in self.named_children():
            if isinstance(v, nn.BatchNorm2d):
                bn_prefix = prefix + k
                keys = [k for k in state_dict.keys() if k.startswith(bn_prefix)]
                bn_state_dict = {k: state_dict.pop(k) for k in keys}
                if ss_candidate_ch is None or len(ss_candidate_ch)==1:
                    _bn_weight(bn_prefix, bn_prefix, bn_state_dict, v.num_features)
                else:
                    try:
                        ch_idx = ss_candidate_ch.index(v.num_features)
                    except:
                        ch_idx = bisect.bisect_right(ss_candidate_ch, v.num_features)
                    _bn_weight(bn_prefix+'.%d'%ch_idx, bn_prefix, bn_state_dict, v.num_features)
#                state_dict.update(bn_state_dict)
            elif isinstance(v, nn.ModuleList) and isinstance(v[0], nn.BatchNorm2d):
                bn_prefix = prefix + k
                keys = [k for k in state_dict.keys() if k.startswith(bn_prefix)]
                bn_state_dict = {k: state_dict.pop(k) for k in keys}
                for sub_idx, sub_bn in enumerate(v):
                    if ss_candidate_ch is None or len(ss_candidate_ch)==1:
                        _bn_weight(bn_prefix, bn_prefix+'.%d'%sub_idx, bn_state_dict, v.num_features)
                    else:
                        try:
                            ch_idx = ss_candidate_ch.index(sub_bn.num_features)
                        except:
                            ch_idx = bisect.bisect_right(ss_candidate_ch, v.num_features)
                        _bn_weight(bn_prefix+'.%d'%ch_idx, bn_prefix+'.%d'%sub_idx, bn_state_dict, v.num_features)
#                state_dict.update(bn_state_dict)
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
    

class AtomSearchModule(SearchModule):
    def __init__(self, in_channel, out_channel, strides, 
        input_idx, # candidate_edge
        candidate_op,
        auto_refine=False, adjust_ch_op=None, upsample_op=None, 
        act=nn.ReLU(), bn=dict(submodule_name='torch.nn.BatchNorm2d', args=dict(affine=True)), bn_per_ch=True, drop_path_prob=0.): 
        """
        strides: a list indicating the scale for each edge. Whether to up-sampling or down-sampling, and how much the degree is
        """
        super(AtomSearchModule, self).__init__()
        # candidate_edge
        self.input_idx = check_nesting(input_idx, 1)
        self.cin = check_nesting(in_channel, 1)
        self.strides = check_nesting(strides, 1)
        assert len(self.input_idx) == len(self.cin) and len(self.cin) == len(self.strides)

        # candidate_op
        self.candidate_op = check_nesting(candidate_op, 2)
        if len(self.candidate_op) == 1:
            self.candidate_op = [self.candidate_op[0]] * len(self.cin)
        assert len(self.candidate_op) == len(self.cin)
        #TODO: Note that cfg file has to make sure that the outlayer of self.candidate_op should has the same sampling property as self.input_idx

        # candidate_ch
        self.candidate_ch = check_nesting(out_channel, 1)
        self.cout = max(self.candidate_ch)

        def assign_ch(op_cfg):
            if isinstance(op_cfg, (list, tuple)): # only assign the cout of the last parametric operation
                for cfg in op_cfg[-1:-1:-1]:
                    if assign_ch(cfg):
                        break
            elif isinstance(op_cfg, (dict, IIDSpace)):
                arg_names = inspect.getfullargspec(get_layer(op_cfg['submodule_name']).__init__).args
                if 'out_channel' in arg_names:
                    if isinstance(self.candidate_ch, _SearchSpace): 
#                        assert(op_cfg, IIDSpace)
                        op_cfg['args']['out_channel'] = self.candidate_ch.space
                        op_cfg['args']['bn_per_ch'] = bn_per_ch
                    else:
                        op_cfg['args']['out_channel'] = self.candidate_ch
                    return True
                return False
        # build operations
        op_builder = OpBuilder(
              auto_refine=auto_refine,
              adjust_ch_op=adjust_ch_op,
              upsample_op=upsample_op
        )
        self.m = nn.ModuleList([])
        for ei, (cin, s, parallel_op) in enumerate(zip(self.cin, self.strides, self.candidate_op)):
            # assign candidate_ch to each candidate_op
            parallel_op = deepcopy(parallel_op)
            for seq_op in parallel_op:
                assign_ch(seq_op)
            # build operations on each edge
            self.m.append(op_builder.build_parallel_op(parallel_op, cin, [out_channel.space if isinstance(out_channel, _SearchSpace) else out_channel for _ in range(len(parallel_op))], s))

        self.act = get_act(act)
        self.bn_per_ch = bn_per_ch and len(self.candidate_ch)>1
        if self.bn_per_ch:
            self.bn = nn.ModuleList([get_norm(bn, int(ch)) for ch in self.candidate_ch]) 
        else: self.bn = get_norm(bn, self.cout)
        if drop_path_prob > 0:
            self.drop_path = DropPath(drop_path_prob)
        else: self.drop_path = None

    def forward_edge(self, x, edge_module, op_alphas, op_space, ch_alphas):
        def _forward_op(x, op, op_in_space, ptr):
            if isinstance(op, SearchModule) and isinstance(op_in_space, _SearchSpace):
                end_ptr = ptr + op_in_space.size
                return op(x, op_alphas=op_alphas[ptr:end_ptr], ch_alphas=ch_alphas), end_ptr
            else: 
                return op(x), ptr

        out, ptr = 0., 0
        for idx, (op, op_in_space) in enumerate(zip(edge_module, op_space)):
#            print(op)
            if isinstance(op, nn.Sequential):
                tmp, end_ptr = x, ptr
                for sub_op in op:
                    tmp, end_ptr = _forward_op(tmp, sub_op, op_in_space, end_ptr)
            else:
                tmp, end_ptr = _forward_op(x, op, op_in_space, ptr)
            if end_ptr == ptr:
                out = out + op_alphas[ptr] * tmp 
                ptr = ptr+1
            else:
                out = out + tmp
                ptr = end_ptr
        if self.drop_path: # and (len(edge_module)>1 or not torch.equal(x, out)): 
            out = self.drop_path(out)
        return out

#        out, ptr = 0., 0
#        for idx, (op, op_in_space) in enumerate(zip(edge_module, op_space)):
#            if isinstance(op, nn.Sequential):
#                tmp = x
#                end_ptr = ptr + 1
#                for sub_op in op:
#                    if isinstance(sub_op, SearchModule):
#                        end_ptr = ptr + op_in_space.size
#                        tmp = sub_op(tmp, op_alphas=op_alphas[ptr:end_ptr], ch_alphas=ch_alphas)
#                    else: tmp = sub_op(tmp)
#                out = out + tmp
#            elif isinstance(op, SearchModule):
#                out = out + op(x, op_alphas=op_alphas[ptr:end_ptr], ch_alphas=ch_alphas)
#            else:
#                if op_alphas[ptr] > 0:
#                    out = out + op_alphas[ptr] * op(x)
#                ptr += 1
#            ptr = end_ptr


    def forward(self, xs, op_alphas=None, ch_alphas=None, edge_alphas=None):
        edge_alphas = self.norm_arch_parameters(self.input_idx, edge_alphas, device=xs[0].device)
        ch_alphas = self.norm_arch_parameters(self.candidate_ch, ch_alphas, device=xs[0].device)
        if op_alphas is None: op_alphas = [None for _ in range(len(self.input_idx))]
        op_alphas = [self.norm_arch_parameters(cand_op, op_alpha, device=xs[0].device) for cand_op, op_alpha in zip(self.candidate_op, op_alphas)]

        out = sum(self.forward_edge(x, m, edge_op_alphas, edge_op_space, ch_alphas) * edge_alpha 
                for x, m, edge_alpha, edge_op_alphas, edge_op_space in zip(xs, self.m, edge_alphas, op_alphas, self.candidate_op))

        bn = self.get_norm_layer(ch_alphas, self.bn, self.bn_per_ch)

        if bn: out = bn(out)
        if self.act: out = self.act(out)

        return out

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        search_space = state_dict.get(prefix+'search_space')
        # get the selected edge
        if search_space is None or len(search_space)==0:
            return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        if 'input_idx' in search_space:
            edge_idx = [search_space['input_idx'].index(idx) for idx in self.input_idx]
        else: edge_idx = self.input_idx
        # get the selected op on each selected edge
        #TODO: How to get the selected op, since the item of candidate_op can also be ConvBNAct_search? 
        #Solved: We adopt FlattenSampledDiscreteSpace to flatten the search space, so we can directly index from the space
        if 'candidate_op' in search_space:
            op_idx = []
            for ei, ss_ei in enumerate(edge_idx):
                space_size = [cand.size if isinstance(cand, _SearchSpace) else 1 for cand in search_space['candidate_op'][ss_ei]]
                flattened_idx = [search_space['candidate_op'][ss_ei].index(tmp) for tmp in self.candidate_op[ei]]
                op_idx.append([bisect.bisect_right(space_size, tmp) for tmp in flattened_idx])
            # rename the op weights in the state dict
            edge_prefix = prefix + 'm'
            prefix_mapping = {}
            for ei, ss_ei in enumerate(edge_idx):
                for opi, ss_opi in enumerate(op_idx[ei]):
                    prefix_mapping[edge_prefix+'.%d.%d'%(ss_ei, ss_opi)] = edge_prefix+'.%d.%d'%(ei, opi)
    
            tmp_state_dict = {}
            keys = list(state_dict.keys())
            for key in keys:
                weight = state_dict.pop(key)
                for ss_prefix, new_prefix in prefix_mapping.items():
                    if key.startswith(ss_prefix):
                        tmp_state_dict[key.replace(ss_prefix, new_prefix, 1)] = weight
            state_dict.update(tmp_state_dict)

        # load search space
        #TODO: How to deal with sampler_weights?
        for k, v in search_space.items():
            this_v = getattr(self, k)
            if isinstance(this_v, _SearchSpace) and this_v.size == v.size:
                setattr(self, k, v)

        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

class ConvBNAct_search(SearchModule):
    def __init__(self, in_channel, out_channel, 
            candidate_op=[(1,1), (3,1), (5,1), (3,2)], 
            stride=1, pad=None, group=1, act=True, act_first=False, bn=dict(submodule_name='torch.nn.BatchNorm2d', args=dict(affine=True)), bn_per_ch=True, bias=False, merge_kernel=True):
        super(ConvBNAct_search, self).__init__()
        self.merge_kernel = merge_kernel
        self.kd = check_nesting(candidate_op, 2)
        self.stride = stride
        self.group = group
        # candidate_ch
        self.candidate_ch = check_nesting(out_channel, 1)
        self.cout = int(max(self.candidate_ch))

        self.k_max = int(max([(k-1)*d+1 for k, d in self.kd]))
        if merge_kernel:
            self.padding = (self.k_max - 1)//2
            self.weight = self.init_weight(self.cout, in_channel, self.k_max)
            self.bias = self.init_bias(self.cout, self.weight) if bias else None
        else:
            self.weight, self.bias = nn.ParameterList([]), nn.ParameterList([])
            for k, d in self.kd:
                self.weight.append(self.init_weight(self.cout, in_channel, k))
                self.bias.append(self.init_bias(self.cout, self.weight[-1]))

        self.act = get_act(act)
        self.act_first = act_first

        self.bn_per_ch = bn_per_ch and len(self.candidate_ch)>1
        if self.bn_per_ch: self.bn = nn.ModuleList([get_norm(bn, int(ch)) for ch in self.candidate_ch]) 
        else: self.bn = get_norm(bn, self.cout)

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
        valid_cout = 0
        for ch, a_e in zip(self.candidate_ch, alphas):
            if a_e > 0:
                channel_mask[:int(ch)] += a_e
                valid_cout = max(valid_cout, int(ch))
        merge_kernel = merge_kernel[:valid_cout,:,:,:] * channel_mask[:valid_cout].view(-1,1,1,1)
        if bias is not None: bias = bias[:valid_cout] 
        return merge_kernel, bias

        
    def forward(self, x, op_alphas=None, ch_alphas=None):
        if self.act_first and self.act: x = self.act(x)

        Cin = x.size(1)
        bias = self.bias
        ch_alphas = self.norm_arch_parameters(self.candidate_ch, ch_alphas, device=x.device)
        op_alphas = self.norm_arch_parameters(self.kd, op_alphas, device=x.device)
        bn = self.get_norm_layer(ch_alphas, self.bn, self.bn_per_ch)
                                   
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

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        weight_name = prefix + 'weight'
        bias_name = prefix + 'bias'
        search_space = state_dict.get(prefix+'search_space', None)
        if search_space is None or 'kd' not in search_space:
            cout, cin, _, _ = self.weight.shape
            state_dict[weight_name] = state_dict[weight_name][:cout, :cin, :, :]
            if bias_name in state_dict:
                state_dict[bias_name] = state_dict[bias_name][:cout]
            return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

        ss_kd = [list(kd) for kd in search_space['kd']]
        op_idx = [ss_kd.index(list(kd)) for kd in self.kd]
        if self.merge_kernel:
            assert weight_name in state_dict
            weight = state_dict[weight_name]
            start = int((weight.shape[-1] - self.k_max) / 2)
            end = int(weight.shape[-1] - start)
            cout, cin, _, _ = self.weight.shape
            with torch.no_grad():
                state_dict_op_alphas = self.norm_arch_parameters(search_space['kd'], device=weight.device)
            state_dict_op_alphas = torch.gather(state_dict_op_alphas, dim=-1, index=torch.tensor(op_idx, device=state_dict_op_alphas.device))
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

        for k, v in search_space.items():
            this_v = getattr(self, k)
            if isinstance(this_v, _SearchSpace) and this_v.size == v.size:
                setattr(self, k, v)


        #TODO: How to deal with arch parameters?
#        state_dict_op_alphas = state_dict.pop(prefix+'op_alphas', torch.tensor([1.]*len(ss_kd)))
##        for idx in set(range(len(ss_kd)))-set(op_idx): state_dict_op_alphas[idx].copy_(0.)
#        state_dict_op_alphas = torch.gather(state_dict_op_alphas, dim=-1, index=torch.tensor(op_idx))
#        state_dict_op_alphas = state_dict_op_alphas / state_dict_op_alphas.sum()
#        state_dict.pop(prefix+'ch_alphas', None)

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

    def forward(self, x, op_alphas=None, ch_alphas=None):
        x = self.act(x) if self.act_first and self.act is not None else x

        Cin = x.size(1)
        bias = self.bias
        ch_alphas = self.norm_arch_parameters(self.candidate_ch, ch_alphas, device=x.device)
        op_alphas = self.norm_arch_parameters(self.kd, op_alphas, device=x.device)
        bn = self.get_norm_layer(ch_alphas, self.bn, self.bn_per_ch)

        if self.merge_kernel:
            merge_kernel = self.get_merge_kernel(self.weight['depth_weight'], op_alphas, merge=True) if len(self.kd)>1 else self.weight['depth_weight']
            if Cin != merge_kernel.size(0): merge_kernel = merge_kernel[:Cin,:,:,:]
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
        search_space = state_dict.get(prefix+'search_space', None)
        if search_space is None or 'kd' not in search_space:
            cout, cin, _, _ = self.weight['point_weight'].shape
            depth_weight = state_dict[weight_name+'.depth_weight']
            start = int((depth_weight.shape[-1] - self.k_max) / 2)
            end = int(depth_weight.shape[-1] - start)
            state_dict[weight_name+'.depth_weight'] = state_dict[weight_name+'.depth_weight'][:cin,:,start:end, start:end]
            state_dict[weight_name+'.point_weight'] = state_dict[weight_name+'.point_weight'][:cout,:cin,:,:]
            if bias_name in state_dict:
                state_dict[bias_name] = state_dict[bias_name][:cout]
            return super(ConvBNAct_search, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

        ss_kd = [list(kd) for kd in search_space['kd']]
        op_idx = [ss_kd.index(list(kd)) for kd in self.kd]
        if self.merge_kernel:
            depth_weight = state_dict[weight_name+'.depth_weight']
            start = int((depth_weight.shape[-1] - self.k_max) / 2)
            end = int(depth_weight.shape[-1] - start)
            cout, cin, _, _ = self.weight['point_weight'].shape
            with torch.no_grad():
                state_dict_op_alphas = self.norm_arch_parameters(search_space['kd'], device=depth_weight.device)
            state_dict_op_alphas = torch.gather(state_dict_op_alphas, dim=-1, index=torch.tensor(op_idx, device=state_dict_op_alphas.device))
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

        for k, v in search_space.items():
            this_v = getattr(self, k)
            if isinstance(this_v, _SearchSpace) and this_v.size == v.size:
                setattr(self, k, v)
        #TODO: How to deal with arch parameters?
#        state_dict_op_alphas = state_dict.pop(prefix+'op_alphas', torch.tensor([1.]*len(ss_kd)))
##        for idx in set(range(len(ss_kd)))-set(op_idx): state_dict_op_alphas[idx].copy_(0.)
#        state_dict_op_alphas = torch.gather(state_dict_op_alphas, dim=-1, index=torch.tensor(op_idx))
#        state_dict_op_alphas = state_dict_op_alphas / state_dict_op_alphas.sum()
#        state_dict.pop(prefix+'ch_alphas', None)
        return super(ConvBNAct_search, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


class SPP_search(SearchModule):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, in_channel, out_channel, kernels=(5, 9, 13), expansion=0.5, bn=torch.nn.BatchNorm2d, act=nn.SiLU):
        super(SPP_search, self).__init__()
        if isinstance(expansion, _SearchSpace):
            c_ = expansion.new_space(space=[int(in_channel*e) for e in expansion])
        else:
            c_ = int(in_channel * expansion)  # hidden channels
        self.cv1 = ConvBNAct_search(in_channel, c_, candidate_op=[(1,1)], stride=1, act=act, bn=bn, merge_kernel=True)
        self.cv2 = ConvBNAct_search(c_ * (len(kernels) + 1), out_channel, candidate_op=[(1,1)], stride=1, act=act, bn=bn, merge_kernel=True)

        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in kernels])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))

