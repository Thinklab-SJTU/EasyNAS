import argparse
from builder import parse_cfg, create_dataloder, create_model, create_cirterion, create_submodule_from_dict, create_hook

parser = argparse.ArgumentParser("train")
parser.add_argument('--cfg', type=str, help='location of the config file')

args = parser.parse_args()

def main():
    # read the config file
    cfg = parse_cfg(args.cfg)
    for k, v in cfg.items():
        print(k, v)

    # build submodules
    # build data
    dataloader = create_dataloder(cfg['data'])
    # parse model
    model = create_model(cfg['model'])
    # parse criterion
    criterion = create_criterion(cfg['criterion'])
    # parse optimizer
    param = model.parameters()
    cfg['optimizer']['args']['params'] = param
    optimizer = create_submodule_from_dict(cfg['optimizer'])
    opt_hook = cfg.get('opt_hook', None)
    if opt_hook:
        opt_hook['hook_args']['optimizer'] = optimizer
        optimizer = create_hook(opt_hook)
    # parse scheduler
    cfg['lr_scheduler']['args']['optimizer'] = optimizer
    scheduler = create_submodule_from_dict(cfg['lr_scheduler'])
    scheduler_hook = cfg.get('lr_scheduler_hook', None)
    if scheduler_hook:
        scheduler_hook['hook_args']['lr_scheduler'] = scheduler
        scheduler = create_hook(scheduler_hook)
    # parse other hooks
    hooks = []
    for k, v in cfg.items():
        if 'hook' in k:
            hooks.append(create_hook(v))

    # build trainer
    trainer = Trainer(dataloders=dataloders, 
                      model=model, 
                      criterion=criterion, 
                      optimizer=optimizer,
                      lr_scheduler=scheduler,
                      hooks=hooks
                      )
    trainer.run(cfg['epoch'])
    

if __name__ == '__main__':
    main()
