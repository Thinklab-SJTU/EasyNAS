from .search_common import SearchLayer, ConvBNAct_search, SepConvBNAct_search

class YOLOBottleneck_search(SearchLayer):
    # Standard bottleneck
    def __init__(self, in_channel, out_channel, candidate_kernel_dilation=[(3,1),(5,1),(3,2)], candidate_ch=[1.], shortcut=True, group=1, expansion=0.5, gumbel_channel=False, separable=False, merge_kernel=True):  # ch_in, ch_out, shortcut, groups, expansion
        super(YOLOBottleneck_search, self).__init__()
        self.gumbel_channel = gumbel_channel

        c_ = int(out_channel * e)  # hidden channels
        c_max = int(c_ * max(candidate_ch))
        self.cv1 = ConvBNAct_search(in_channel, c_max, candidate_kernel_dilation=[(1,1)], candidate_ch=candidate_ch, stride=1, gumbel_channel=gumbel_channel, act=nn.SiLU, bn=True, merge_kernel=merge_kernel)
        if separable: my_conv = SepConvBNAct_search
        else: my_conv = ConvBNAct_search
        self.cv2 = my_conv(c_max, out_channel, candidate_kernel_dialtion, candidate_ch=[1.], stride=1, group=group, gumbel_channel=gumbel_channel, act=nn.SiLU, bn=True, merge_kernel=merge_kernel)
        self.add = shortcut and in_channel == out_channel

    def forward(self, x, op_alphas=None, ch_alphas=None):
        if self.gumbel_channel:
          cout = x.size(1)
          out = self.cv2(self.cv1(x), op_alphas, ch_alphas)
          return x + out[:,:cout,:,:] if self.add else out
        else:
          return x + self.cv2(self.cv1(x), op_alphas, ch_alphas) if self.add else self.cv2(self.cv1(x), op_alphas, ch_alphas)

class YOLOC3_search(SearchLayer):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, in_channel, out_channel, num_repeat=1, candidate_kernel_dilation=[(3,1),(5,1),(3,2)], candidate_ch=[1.], shortcut=True, group=1, expansion=0.5, search_out_channel=None, gumbel_channel=False, separable=False, merge_kernel):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(YOLOC3_search, self).__init__()
        if search_out_channel==True:
            self.search_out_channel = candidate_ch
        elif search_out_channel in [False, None]:
            self.search_out_channel = [1.]
        elif isinstance(search_out_channel, list):
            self.search_out_channel = search_out_channel
        else:
            raise(ValueError("search_out_channel has to be bool or None or a list of float"))
        self.gumbel_channel = gumbel_channel

        out_channel = out_channel * max(self.search_out_channel)
        c_ = int(out_channel * expansion)  # hidden channels
        self.cv1 = ConvBNAct_search(in_channel, c_, candidate_kernel_dilation=[(1,1)], candidate_ch=self.search_out_channel, stride=1, gumbel_channel=gumbel_channel, independent_ch_arch_param=False, merge_kernel=merge_kernel)
        self.cv2 = ConvBNAct_search(in_channel, c_, candidate_kernel_dilation=[(1,1)], candidate_ch=self.search_out_channel, stride=1, gumbel_channel=gumbel_channel, independent_ch_arch_param=False, merge_kernel=merge_kernel)
        if gumbel_channel:
            self.cv3 = nn.ModuleList([ConvBNAct_search(c_, out_channel, candidate_kernel_dilation=[(1,1)], candidate_ch=self.search_out_channel, stride=1, gumbel_channel=gumbel_channel, act=False, bn=False, independent_ch_arch_param=False) for _ in range(2)])  
            self.cv3_bn = nn.ModuleList([nn.BatchNorm2d(int(out_channel*e)) for e in self.search_out_channel])
            self.cv3_act = nn.SiLU()
        else:
            self.cv3 = ConvBNAct_search(2 * c_, out_channel, candidate_kernel_dilation=[(1,1)], candidate_ch=self.search_out_channel, stride=1, act=nn.SiLU(), bn=True, gumbel_channel=gumbel_channel, independent_ch_arch_param=False, merge_kernel=merge_kernel)  

        if len(self.search_out_channel) > 1:
            self.register_buffer('ch_alphas', torch.autograd.Variable(1e-3*torch.randn(len(self.search_out_channel)), requires_grad=True))

        self.m = nn.Sequential(*[YOLOBottleneck_search(c_, c_, candidate_kernel_dilation, candidate_ch, shortcut, group, expansion=1.0, gumbel_channel=gumbel_channel, separable=separable, merge_kernel=merge_kernel) for _ in range(num_repeat)])

    def forward(self, x):
        if self.gumbel_channel:
            ch_alphas = gumbel_softmax(F.log_softmax(self.ch_alphas, dim=-1), hard=True) if hasattr(self, 'ch_alphas') else None 
            out = self.cv3[0](self.m(self.cv1.forward_withAlpha(x, alphas_channel)), ch_alphas=ch_alphas) + self.cv3[1](self.cv2.forward_withAlpha(x, alphas_channel), ch_alphas=ch_alphas)
            a_e, idx = ch_alphas.max()
            return self.cv3_act(self.cv3_bn[idx](out))
        else:
            ch_alphas = nn.functional.softmax(self.ch_alphas, dim=-1) if hasattr(self, 'ch_alphas') else None
            return self.cv3(torch.cat((self.m(self.cv1(x, ch_alphas=ch_alphas)), self.cv2(x, ch_alphas=ch_alphas)), dim=1), ch_alphas=ch_alphas)
