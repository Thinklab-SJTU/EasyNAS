from inspect import isfunction
import numpy as np
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms

from .utils import Cutout

def get_transforms(train, mean, std, cutout=True, cutout_length=16, cutout_prob=1.0):
  if train:
      transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
      ])
      if cutout:
        transform.transforms.append(Cutout(cutout_length, cutout_prob))
  else:
      transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        ])
  return transform

class CIFAR10(dset.CIFAR10):
    MEAN = [0.49139968, 0.48215827, 0.44653124]
    STD = [0.24703233, 0.24348505, 0.26158768]
    def __init__(self, root, train, download=True, transform='default'):
        if transform == 'default':
            transform = partial(get_transforms, cutout=True, cutout_length=16, cutout_prob=1.0)
        try:
            transform = transform(train, self.MEAN, self.STD)
            assert isinstance(transform, transforms.Compose), f"No implementation for transform as {transform}"
        except Exception as e:
            raise(e)
        super().__init__(root=root, train=train, download=download, transform=transform)


class CIFAR100(dset.CIFAR100):
    MEAN = [0.5071, 0.4865, 0.4409]
    STD = [0.2673, 0.2564, 0.2762]
    def __init__(self, root, train, download=True, transform='default'):
        if transform == 'default':
            transform = partial(get_transforms, cutout=True, cutout_length=16, cutout_prob=1.0)
        try:
            transform = transform(train, self.MEAN, self.STD)
            assert isinstance(transform, transforms.Compose), f"No implementation for transform as {transform}"
        except Exception as e:
            raise(e)
        super().__init__(root=root, train=train, download=download, transform=transform)
	

class SVHN(dset.SVHN):
    MEAN = [0.4377, 0.4438, 0.4728]
    STD = [0.1980, 0.2010, 0.1970]
    def __init__(self, root, train, download=True, transform='default'):
        if transform == 'default':
            transform = partial(get_transforms, cutout=True, cutout_length=16, cutout_prob=1.0)
        try:
            transform = transform(train, self.MEAN, self.STD)
            assert isinstance(transform, transforms.Compose), f"No implementation for transform as {transform}"
        except Exception as e:
            raise(e)
        super().__init__(root=root, split='train' if train else 'test', download=download, transform=transform)
