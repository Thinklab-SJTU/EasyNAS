import torch
import torch.nn as nn

class OpLayer(nn.Module):
    def __init__(self): 
        super(OpLayer, self).__init__()
        self.adjust_ch_op = OP(OPtype='ConvBNAct', args=dict(kernel=1, dilation=1, bn=False, act=None))
        self.upsample_op = OP(OPtype=nn.Upsampling, args=dict(size=None, scale_factor=None, mode='nearest', align_corners=None))

    def refine_op(self, op_config, in_channel, out_channel, stride=1, **update_args):
        refined_op_config = []
        if isinstance(in_channel, int): in_channel = (in_channel,)*len(op_config)
        if isinstance(out_channel, int): out_channel = (out_channel,)*len(op_config)
        if isinstance(stride, int): stride = (stride,) + (1,)*len(op_config)
        for idx, (cin, cout, s, op) in enumerate(zip(in_channel, out_channel, stride, op_config)):
            if isinstance(op, [tuple, list]):
                refined_op = list(deepcopy(op))
                refined_op[0].args.update(in_channel=cin)
                assert update_args is None
            elif isinstance(op, OP):
                refined_op = deepcopy(op)
                up_s, s = int(1./s), max(1, s)
                adjust_ch = False
                tmp_update_args = {}
                for k, v in update_args:
                    tmp_update_args[k] = v[idx] if isinstance(v, [list, tuple]) else v
                if refined_op.OPtype in NEED_INOUTC_OPs: 
                    refined_op.args.update(in_channel=cin, out_channel=cout, stride=s, **tmp_update_args)
                else:
                    refined_op.args.update(stride=s, **tmp_update_args)
                    if in_channel != out_channel:
                        print("Warning: input channel should be the same as output channel")
                        adjust_ch = True
                refined_op = [refined_op]
                if up_s > 1: 
                    upsample_op = deepcopy(self.upsample_op)
                    upsample_op.args.update(scale_factor=up_s)
                    refined_op.append(upsample_op)
                if adjust_ch: 
                    adjust_ch_op = deepcopy(self.adjust_ch_op)
                    adjust_ch_op.args.update(in_channel=cin, out_channel=cout)
                    refined_op.append(adjust_ch_op)

            refined_op_config.append(tuple(refined_op))
        return tuple(refined_op_config)

    def refine_C_stride(self, op_config, in_channel, out_channel, stride):
        return self.refine_op(op_config, in_channel, out_channel, stride)

    def build_op(self, op_config):
        ops = nn.ModuleList([])
        for config in op_config:
            if isinstance(config, [tuple, list]):
                op = nn.Sequential()
                for idx, sub_config in enumerate(config):
                    module = get_submodule(sub_config.OPtype) 
                    op.add_module(idx, module(**sub_config.args))
            elif isinstance(config, OP):
                module = get_submodule(config.OPtype) 
                op = module(**config.args)
            else: 
                raise(TypeError("op_config should be either OP or sequence"))
            ops.append(op)
        return ops
