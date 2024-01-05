import os
import torch
import numpy as np

from builder import parse_cfg, get_submodule_by_name, create_model
from src.hook import CkptHOOK
from src.search_space.base import SampleNode

def get_random(task):
    return np.random.randn(1)[0]

def get_performance(task):
    task_cfg = task.config
    print(task_cfg)
    engine_cfg = task_cfg['engine']
    engine = get_submodule_by_name(engine_cfg['submodule_name'], search_path='engines')(
                      **engine_cfg['args'],
                      )
    print("Engine is running...")
    engine.run(**engine_cfg.get('run_args', {}))
    reward = engine.extract_performance() #engine.info.results.val.best
    del engine
    torch.cuda.empty_cache()
    return reward

def get_num_parameters(task):
    task_cfg = task.config
    engine_cfg = task_cfg['engine']
    engine = get_submodule_by_name(engine_cfg['submodule_name'], search_path='engines')(
                      **engine_cfg['args'],
                      )
    return count_parameters_in_MB(engine.model_without_ddp)

def get_edgeDevice_latency(task, remote_cmd, host, username, password, port=22):
    if isinstance(task, SampleNode):
        task_cfg = task.config
    elif isinstance(task, dict):
        task_cfg = task
    else:
        raise(TypeError(f"No implementation for task typed as {type(task)}"))
    # get model
    engine_cfg = task_cfg['engine']
    engine = get_submodule_by_name(engine_cfg['submodule_name'], search_path='engines')(
                      **engine_cfg['args'],
                      )
    model = engine.model_without_ddp
    model.to('cpu')
    model.device = 'cpu'
    model.eval()
    # get checkpoint
    onnx_path = None
    for hook in engine._hooks:
        if isinstance(hook, CkptHOOK):
            onnx_path = os.path.join(hook.save_root, 'model.onnx')
            checkpoint = hook.get_pretrain_model(engine.device, pretrain=hook.save_root)
#            if checkpoint is not None:
#                model.load_state_dict(checkpoint['state_dict'], strict=False)

    # convert to onnx
    from src.evaluater.export import export_onnx
    onnx = export_onnx(model, onnx_path)

    # scp the model to the device that connected to the edge device.
    from src.evaluater.connect import build_sshclient, fetch_info_rk3588
    ssh = build_sshclient(host, username, password, port)
    sftp = ssh.open_sftp()
    remote_path = '~/' + onnx_path.split('/')[-1]
    sftp.put(onnx_path, remote_path)
#    scpclient = SCPClient(ssh_client.get_transport(), socket_timeout=15.0)
#    local_path = file_path + "/" + file_name
#    try:
#        scpclient.put(local_path, remote_path, True)
#    except FileNotFoundError :
#        print "上传失败:" + local_path
#    else:
#        print "上传成功:" + local_path

    # ssh the command to the device and test.
#    remote_cmd = f"docker run -t -i --privileged -v /dev/bus/usb:/dev/bus/usb -v /home/wangxiaoxing:/home/wangxiaoxing rknn-toolkit2:ubuntu20.04-cp38 sh -c 'adb devices && cd /home/wangxiaoxing/rknn_model_zoo/examples/yolov7/python && python test_onnx.py --onnx_path ../model/yolov7-tiny.onnx --target rk3588 --quant i8'"
    cmd = remote_cmd + f' --onnx_path {remote_path}'
    info = fetch_info_rk3588(ssh, remote_cmd)

#def get_mip_reward(task):
#    task_cfg = task.config
#    print(task_cfg)
#    engine_cfg = task_cfg['engine']
#    engine = get_submodule_by_name(engine_cfg['submodule_name'], search_path='engines')(
#                      **engine_cfg['args'],
#                      )
#    print("Engine is running...")
#    engine.run(**engine_cfg.get('run_args', {}))
#    reward = -abs(engine.info.results['sum_primal_bound'])
#    del engine
#    torch.cuda.empty_cache()
#    return reward

if __name__ == '__main__':
    cfg = 'cfg/EdgeDevice/test.yaml'
    cfg = parse_cfg(cfg)
    get_edgeDevice_latency(cfg)

