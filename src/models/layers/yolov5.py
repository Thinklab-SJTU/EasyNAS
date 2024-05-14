import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ConvBNAct, SepConvBNAct

__all__ = ["YOLOBottleneck", "YOLOBottleneckCSP"
          ,"YOLOC3", "NMS", "autoShape", "Detections", "Classify", "YOLODetect"]

class YOLOBottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, in_channel, out_channel, kernel, dilation=1, shortcut=True, group=1, expansion=0.5, separable=False):  # ch_in, ch_out, shortcut, groups, expansion
        super(YOLOBottleneck, self).__init__()
        hidden_channel = int(out_channel * expansion)  # hidden channels
        if separable: my_conv = SepConvBNAct
        else: my_conv = ConvBNAct
        self.cv1 = ConvBNAct(in_channel, hidden_channel, kernel=1, dilation=1, stride=1, bn=nn.BatchNorm2d, act=nn.SiLU())
        self.cv2 = my_conv(hidden_channel, out_channel, kernel, dilation=dilation, stride=1, group=group, bn=nn.BatchNorm2d, act=nn.SiLU())
        self.add = shortcut and in_channel == out_channel

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class YOLOBottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, in_channel, out_channel, num_repeat=1, shortcut=True, group=1, expansion=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(YOLOBottleneckCSP, self).__init__()
        hidden_channel = int(out_channel * expansion)  # hidden channels
        self.cv1 = ConvBNAct(in_channel, hidden_channel, kernel=1, dilation=1, stride=1, bn=nn.BatchNorm2d, act=nn.SiLU())
        self.cv2 = nn.Conv2d(in_channel, hidden_channel, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(hidden_channel, hidden_channel, 1, 1, bias=False)
        self.cv4 = ConvBNAct(2 * hidden_channel, out_channel, kernel=1, dilation=1, stride=1)
        self.bn = nn.BatchNorm2d(2 * hidden_channel)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[YOLOBottleneck(hidden_channel, hidden_channel, shortcut, group, expansion=1.0) for _ in range(num_repeat)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class YOLOC3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, in_channel, out_channel, num_repeat=1, kernel=3, dilation=1, shortcut=True, group=1, expansion=0.5, e_bottleneck=1., separable=False):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(YOLOC3, self).__init__()
        if isinstance(kernel, int): ks = [kernel for _ in range(num_repeat)]
        else: ks = kernel
        if isinstance(dilation, int): ds = [dilation for _ in range(num_repeat)]
        else: ds = dilation
        if isinstance(e_bottleneck, float): es = [e_bottleneck for _ in range(num_repeat)]
        else: es = e_bottleneck
        assert len(ks) >= num_repeat
        assert len(ds) >= num_repeat
        assert (len(es) >= num_repeat) or (len(es) >= num_repeat+1)

        if isinstance(out_channel, int): out_channel = [out_channel for _ in range(num_repeat)]  
        c1out = int(out_channel[0]*expansion); c2out = int(out_channel[-1]*expansion)  # hidden channels

        self.cv1 = ConvBNAct(in_channel, c1out, kernel=1, dilation=1, stride=1, bn=nn.BatchNorm2d, act=nn.SiLU())
        m_list = []; cin = c1out
        for i in range(num_repeat):
          m_list.append(YOLOBottleneck(cin, int(out_channel[i]*expansion), ks[i], ds[i], shortcut, group, expansion=es[i], separable=separable))
          cin = int(out_channel[i]*expansion)
        self.m = nn.Sequential(*m_list)
        self.cv2 = ConvBNAct(in_channel, c2out, kernel=1, dilation=1, stride=1, bn=nn.BatchNorm2d, act=nn.SiLU())
        self.cv3 = ConvBNAct(2 * c2out, out_channel[-1], kernel=1, dilation=1, stride=1, bn=nn.BatchNorm2d, act=nn.SiLU())  # act=FReLU(c2)

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class NMS(nn.Module):
    # Non-Maximum Suppression (NMS) module
    conf = 0.25  # confidence threshold
    iou = 0.45  # IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self):
        super(NMS, self).__init__()

    def forward(self, x):
        return non_max_suppression(x[0], conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)


class autoShape(nn.Module):
    # input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    img_size = 640  # inference size (pixels)
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self, model):
        super(autoShape, self).__init__()
        self.model = model.eval()

    def autoshape(self):
        print('autoShape already enabled, skipping... ')  # model already converted to model.autoshape()
        return self

    def forward(self, imgs, size=640, augment=False, profile=False):
        # Inference from various sources. For height=720, width=1280, RGB images example inputs are:
        #   filename:   imgs = 'data/samples/zidane.jpg'
        #   URI:             = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(720,1280,3)
        #   PIL:             = Image.open('image.jpg')  # HWC x(720,1280,3)
        #   numpy:           = np.zeros((720,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,720,1280)  # BCHW
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        p = next(self.model.parameters())  # for device and type
        if isinstance(imgs, torch.Tensor):  # torch
            return self.model(imgs.to(p.device).type_as(p), augment, profile)  # inference

        # Pre-process
        n, imgs = (len(imgs), imgs) if isinstance(imgs, list) else (1, [imgs])  # number of images, list of images
        shape0, shape1, files = [], [], []  # image and inference shapes, filenames
        for i, im in enumerate(imgs):
            if isinstance(im, str):  # filename or uri
                im, f = Image.open(requests.get(im, stream=True).raw if im.startswith('http') else im), im  # open
                im.filename = f  # for uri
            files.append(Path(im.filename).with_suffix('.jpg').name if isinstance(im, Image.Image) else f'image{i}.jpg')
            im = np.array(im)  # to numpy
            if im.shape[0] < 5:  # image in CHW
                im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
            im = im[:, :, :3] if im.ndim == 3 else np.tile(im[:, :, None], 3)  # enforce 3ch input
            s = im.shape[:2]  # HWC
            shape0.append(s)  # image shape
            g = (size / max(s))  # gain
            shape1.append([y * g for y in s])
            imgs[i] = im  # update
        shape1 = [make_divisible(x, int(self.stride.max())) for x in np.stack(shape1, 0).max(0)]  # inference shape
        x = [letterbox(im, new_shape=shape1, auto=False)[0] for im in imgs]  # pad
        x = np.stack(x, 0) if n > 1 else x[0][None]  # stack
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))  # BHWC to BCHW
        x = torch.from_numpy(x).to(p.device).type_as(p) / 255.  # uint8 to fp16/32

        # Inference
        with torch.no_grad():
            y = self.model(x, augment, profile)[0]  # forward
        y = non_max_suppression(y, conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)  # NMS

        # Post-process
        for i in range(n):
            scale_coords(shape1, y[i][:, :4], shape0[i])

        return Detections(imgs, y, files, self.names)


class Detections:
    # detections class for YOLOv5 inference results
    def __init__(self, imgs, pred, files, names=None):
        super(Detections, self).__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*[im.shape[i] for i in [1, 0, 1, 0]], 1., 1.], device=d) for im in imgs]  # normalizations
        self.imgs = imgs  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.files = files  # image filenames
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)

    def display(self, pprint=False, show=False, save=False, render=False, save_dir=''):
        colors = color_list()
        for i, (img, pred) in enumerate(zip(self.imgs, self.pred)):
            str = f'image {i + 1}/{len(self.pred)}: {img.shape[0]}x{img.shape[1]} '
            if pred is not None:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    str += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                if show or save or render:
                    for *box, conf, cls in pred:  # xyxy, confidence, class
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        plot_one_box(box, img, label=label, color=colors[int(cls) % 10])
            img = Image.fromarray(img.astype(np.uint8)) if isinstance(img, np.ndarray) else img  # from np
            if pprint:
                print(str.rstrip(', '))
            if show:
                img.show(self.files[i])  # show
            if save:
                f = Path(save_dir) / self.files[i]
                img.save(f)  # save
                print(f"{'Saving' * (i == 0)} {f},", end='' if i < self.n - 1 else ' done.\n')
            if render:
                self.imgs[i] = np.asarray(img)

    def print(self):
        self.display(pprint=True)  # print results

    def show(self):
        self.display(show=True)  # show results

    def save(self, save_dir='results/'):
        Path(save_dir).mkdir(exist_ok=True)
        self.display(save=True, save_dir=save_dir)  # save results

    def render(self):
        self.display(render=True)  # render results
        return self.imgs

    def __len__(self):
        return self.n

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        x = [Detections([self.imgs[i]], [self.pred[i]], self.names) for i in range(self.n)]
        for d in x:
            for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
                setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x


class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Classify, self).__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g)  # to x(b,c2,1,1)
        self.flat = nn.Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)

class YOLODetect(nn.Module):
    export = False  # onnx export

    def __init__(self, in_channel, strides, num_classes=80, anchors=()):  # detection layer
        super(YOLODetect, self).__init__()
        self.strides = strides
        self.nc = num_classes
        self.no = num_classes + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
#        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.grid = [None for _ in range(self.nl)]# init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.anchors /= torch.tensor(strides).view(-1, 1, 1)
        self.strides = strides
        self._initialize_modules(in_channel)
        self._initialize_biases()

    def _initialize_modules(self, in_channel):
#        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in in_channel)  # output conv
        self.m = nn.ModuleList(ConvBNAct(x, self.no * self.na, kernel=1, dilation=1, stride=1, bn=nn.BatchNorm2d, act=nn.SiLU) for x in in_channel)  # output conv

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
#        for mi, s in zip(self.m, self.strides):  # from
#            b = mi.bias.view(self.na, -1)  # conv.bias(255) to (3,85)
#            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
#            b.data[:, 5:] += math.log(0.6 / (self.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
#            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        self.bias = torch.nn.Parameter(torch.zeros(len(self.strides), self.na, self.no), requires_grad=True)
        for i, s in enumerate(self.strides):  # from
            self.bias.data[i, :, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
        self.bias.data[:, :, 5:] += math.log(0.6 / (self.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls

    def forward(self, x):
        # x = x.copy()  # for profiling
        logits = []
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            logits.append(self.m[i](x[i]))  # conv
            if hasattr(self, 'bias'): 
                logits[-1] += self.bias[i].view(1,-1,1,1)
            bs, _, ny, nx = logits[-1].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            if not self.export:
                logits[-1] = logits[-1].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
#            logits.append(tmp)

            if not self.training:
                grid = self._make_grid(nx, ny).to(logits[-1].device)
#                if self.grid[i] is None or self.grid[i].shape[2:4] != x[i].shape[2:4]:
#                    self.grid[i] = self._make_grid(nx, ny).to(logits[-1].device)
    
                y = logits[-1].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid) * self.strides[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

#        if self.export:
#            for idx in range(len(logits)):
#                logits[idx] = logits[idx].view(bs, -1, self.no)
#            logits = torch.cat(logits, 1)
        return logits if self.training else torch.cat(z, 1)


#    @staticmethod
    def _make_grid(self, nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

