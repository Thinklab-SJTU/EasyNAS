import os
import argparse

from builder import parse_cfg, create_dataloader, create_model, create_criterion, create_submodule_from_dict, create_hook
from pipeline.trainer import Trainer

parser = argparse.ArgumentParser("train")
parser.add_argument('--cfg', type=str, help='location of the config file')

args = parser.parse_args()

def main():
    # read the config file
    cfg = parse_cfg(args.cfg)
    for k, v in cfg.items():
        print(k, v)
    if cfg.get('root_path', None):
        os.makedirs(cfg.get('root_path'), exist_ok=True)

    # build submodules
    # build data
    print("Building dataloader")
    datasets, dataloaders = create_dataloader(cfg['data'])
    # parse model
    print("Building model")
    if cfg.get('root_path') and cfg['model'].get('log_path', None):
        cfg['model']['log_path'] = os.path.join(cfg['root_path'], cfg['model']['log_path'])
    model = create_model(cfg['model'], num_classes=cfg['data']['num_classes'], input_size=cfg['data'].get('input_size', None), log_path=cfg['model'].get('log_path', None))
    # parse criterion
    print("Building criterion")
    criterion = create_criterion(cfg['criterion'])
    # parse optimizer
    print("Building optimizer")
    param = model.parameters()
    cfg['optimizer']['args']['params'] = param
    optimizer = create_submodule_from_dict(cfg['optimizer'])
    opt_hook = cfg.get('opt_hook', None)
    if opt_hook:
        opt_hook['hook_args']['optimizer'] = optimizer
        opt_hook = create_hook(opt_hook)
    # parse scheduler
    print("Building lr scheduler")
    cfg['lr_scheduler']['args']['optimizer'] = optimizer
    scheduler = create_submodule_from_dict(cfg['lr_scheduler'])
    scheduler_hook = cfg.get('lr_scheduler_hook', None)
    if scheduler_hook:
        scheduler_hook['hook_args']['lr_scheduler'] = scheduler
        scheduler_hook = create_hook(scheduler_hook)
    # parse other hooks
    print("Building hooks")
    hooks = []
    for k, v in cfg.items():
        if 'hook' in k:
            hooks.append(create_hook(v))

    # build trainer
    trainer = Trainer(dataloaders=dataloaders, 
                      model=model, 
                      criterion=criterion, 
                      optimizer=opt_hook,
                      lr_scheduler=scheduler_hook,
                      hooks=hooks
                      )
    print("Traning...")
    trainer.run(cfg['epoch'])
    

if __name__ == '__main__':
    main()
