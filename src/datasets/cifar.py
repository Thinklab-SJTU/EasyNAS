import numpy as np
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms

from .utils import Cutout



def _data_transforms_cifar10(train, cutout=True, cutout_length=16, cutout_prob=1.0):
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

  if train:
      transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
      ])
      if cutout:
        transform.transforms.append(Cutout(cutout_length, cutout_prob))
  else:
      transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
  return transform

def _data_transforms_svhn(train, cutout=True, cutout_length=16, cutout_prob=1.0):
  SVHN_MEAN = [0.4377, 0.4438, 0.4728]
  SVHN_STD = [0.1980, 0.2010, 0.1970]

  if train:
      transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(SVHN_MEAN, SVHN_STD),
      ])
      if cutout:
        transform.transforms.append(Cutout(cutout_length,
                                          cutout_prob))
  else:
      transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(SVHN_MEAN, SVHN_STD),
        ])
  return transform


def _data_transforms_cifar100(train, cutout=True, cutout_length=16, cutout_prob=1.0):
  CIFAR_MEAN = [0.5071, 0.4865, 0.4409]
  CIFAR_STD = [0.2673, 0.2564, 0.2762]

  if train:
      transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
      ])
      if cutout:
        transform.transforms.append(Cutout(cutout_length,
                                          cutout_prob))
  else:	
      transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
  return transform


class CIFAR10(dset.CIFAR10):
    def __init__(self, root, train, download=True, transform='default'):
        if transform == 'default':
            transform = _data_transforms_cifar10(train, cutout=True, cutout_length=16, cutout_prob=1.0)
        assert isinstance(transform, transforms.Compose), f"No implementation for transform as {transform}"
        super().__init__(root=root, train=train, download=download, transform=transform)

class CIFAR100(dset.CIFAR100):
    def __init__(self, root, train, download=True, transform='default'):
        if transform == 'default':
            transform = _data_transforms_cifar100(train, cutout=True, cutout_length=16, cutout_prob=1.0)
        assert isinstance(transform, transforms.Compose), f"No implementation for transform as {transform}"
        super().__init__(root=root, train=train, download=download, transform=transform)
	

class SVHN(dset.SVHN):
    def __init__(self, root, train, download=True, transform='default'):
        if transform == 'default':
            transform = _data_transforms_svhn(train, cutout=True, cutout_length=16, cutout_prob=1.0)
        assert isinstance(transform, transforms.Compose), f"No implementation for transform as {transform}"
        super().__init__(root=root, split='train' if train else 'test', download=download, transform=transform)
