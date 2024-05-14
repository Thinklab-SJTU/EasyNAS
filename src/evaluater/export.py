import argparse
import sys
import time
import warnings
from collections.abc import Iterable

import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile


# SiLU https://arxiv.org/pdf/1606.08415.pdf ----------------------------------------------------------------------------
class SiLU(nn.Module):  # export-friendly version of nn.SiLU()
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

def export_onnx(model, onnx_path=None, dynamic=False, dynamic_batch=False):
    import onnx
    print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
    input_size = model.input_size
    if input_size is None:
        dynamic = True
        input_size = [640, 640]
    if onnx_path is None:
        onnx_path = 'model.onnx'
    model.eval()
    for m in model.modules():
        if hasattr(m, 'export'):
            m.export = True
    # Input
    input_size = [input_size, input_size] if isinstance(input_size, int) else input_size
    img = torch.zeros(1, 3, *input_size).to(model.device)  # image size(1,3,320,192) iDetection
    with torch.no_grad():
        model_out = model(img)

    if isinstance(model_out, torch.Tensor):
        output_names = ['output']
    else:
        assert isinstance(model_out, Iterable)
        output_names = [f'output#{idx}' for idx in range(len(model_out))]

    dynamic_axes = None
    if dynamic:
        dynamic_axes = {'images': {0: 'batch', 2: 'height', 3: 'width'}}  # size(1,3,640,640)
        for name in output_names:
            dynamic_axes[name] = {0: 'batch', 2: 'y', 3: 'x'}
    elif dynamic_batch:
        dynamic_axes = {
            'images': {
                0: 'batch',
            }, }
        for name in output_names:
            dynamic_axes[name] = {0: 'batch'}

    for k, m in model.named_modules():
        for child_k, child_m in m.named_children():
            if isinstance(child_m, nn.SiLU):
                setattr(m, child_k, SiLU())

    model.to('cpu')
    img = img.to('cpu')

    torch.onnx.export(model, img, onnx_path, verbose=False, opset_version=12, input_names=['images'],
                      output_names=output_names,
                      dynamic_axes=dynamic_axes)

    # Checks
    onnx_model = onnx.load(onnx_path)  # load onnx model
    onnx.checker.check_model(onnx_model)  # check onnx model
    onnx.save(onnx_model, onnx_path)
    print('ONNX export success, saved as %s' % onnx_path)

#    # runtime test
#    import onnxruntime
#    import numpy as np
#    
#    def _to_numpy(var):
#        if isinstance(var, torch.Tensor):
#            var = var.numpy()
#        elif isinstance(var, (list, tuple)):
#            for idx in range(len(var)):
#                var[idx] = _to_numpy(var[idx])
#        elif isinstance(var, dict):
#            for key in var.keys():
#                var[key] = _to_numpy(var[key])
#        return var
#    model_out = _to_numpy(model_out)
#    sess = onnxruntime.InferenceSession(onnx_path)
#    output = sess.run(output_names, {'images': img.numpy()})
#    if isinstance(model_out, (list, tuple)):
#        for tmp1, tmp2 in zip(output, model_out):
#            assert np.allclose(tmp1, tmp2)
#    elif isinstance(model_out, dict):
#        for tmp1, tmp2 in zip(output, model_out.values()):
#            assert np.allclose(tmp1, tmp2)
#    else:
#        assert np.allclose(output, model_out)

    # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model

    # # Metadata
    # d = {'stride': int(max(model.stride))}
    # for k, v in d.items():
    #     meta = onnx_model.metadata_props.add()
    #     meta.key, meta.value = k, str(v)
    # onnx.save(onnx_model, onnx_path)

    # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
    return onnx_model

