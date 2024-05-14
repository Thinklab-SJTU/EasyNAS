import os
from easydict import EasyDict
from typing import Union, List
import bisect
import math
from itertools import chain
import torch
import torch.nn as nn

from .base import BaseEngine

from builder import create_model
from src.evaluater.connect import build_sshclient, fetch_info_rk3588
from src.evaluater.export import export_onnx
from src.hook import CkptHOOK

class EdgeDeviceEngine(BaseEngine):
    def __init__(self, input_size, ckpt_root, remote_onnx_path, remote_cmd, host, username, password=None, pkey=None, port=22, fetch_info_fn=fetch_info_rk3588, eval_names=('neg-latency',)):
        self.input_size = input_size
        self.eval_names = eval_names
        self.fetch_info_fn = fetch_info_fn 
        self.ckpt_root = ckpt_root
        self.onnx_path = os.path.join(ckpt_root, 'model.onnx')
        self.ssh = build_sshclient(host, username, password, pkey, port)
        self.sftp = self.ssh.open_sftp()
        self.remote_onnx_path = remote_onnx_path
        self.cmd = remote_cmd.format(onnx_path=remote_onnx_path)

        self.info = EasyDict({
            'results': {},
            })

    def _build_model(self, model, input_size=None):
        if isinstance(model, dict):
            print("Building model")
            self.model = create_model(model, input_size=input_size)
        else: 
            assert(isinstance(model, nn.Module))
            self.model = model

        ckpt = CkptHOOK.get_pretrain_model(device=self.model.device, pretrain=self.ckpt_root)
        self.model.load_state_dict(ckpt['state_dict'], strict=False)

    def run(self):
        print("Export onnx model")
        onnx = export_onnx(self.model, self.onnx_path)
        print(f"Transport onnx model from {self.onnx_path} to the host computer {self.remote_onnx_path}")
        self.sftp.put(self.onnx_path, self.remote_onnx_path)
        print("Fetch information from the host computer")
        self.info.results = self.fetch_info_fn(self.ssh, self.cmd)

    def update(self, sample):
        data = sample.get('data', None)
        input_size = data.get('input_size', None) if isinstance(data, dict) else self.input_size
        self._build_model(sample['model'], input_size)

    def extract_performance(self, eval_names=None):
        if eval_names is None: eval_names = self.eval_names
        performance = []
        for i, eval_name in enumerate(eval_names):
            eval_name = eval_name.split('-')
            sign = 1 if len(eval_name) == 1 else -1
            eval_name = eval_name[-1]
            performance = sign * self.info.results[eval_name]
        return performance
        

