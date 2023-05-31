import sys
import os
sys.path.append(os.getcwd())
import yaml

from builder import create_dataloader

def get_all_yml(path):
    return [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.split('.')[-1] in ['yaml', 'yml']]

def test(path):
    if os.path.isdir(path):
        yml_files = get_all_yml(path)
    elif os.path.isfile(path) and f.split('.')[-1] in ['yaml', 'yml']:
        yml_files = [path]
    else:
        raise(ValueError(f"The path should be a directory or a yaml file, but got {path}"))
    print(yml_files)

    for name in yml_files:
        print(f"Build dataloader according to {name}")
        with open(name, 'r') as f:
       	    cfg = yaml.safe_load(f.read())
            print(cfg)
            datasets, dataloaders = create_dataloader(cfg)
            for k, dataset in datasets.items():
                print(len(dataset))
            for k, dataloader in dataloaders.items():
                print(len(dataloader.dataset))
    
if __name__ == '__main__':
    test('dataset/cfg/')
