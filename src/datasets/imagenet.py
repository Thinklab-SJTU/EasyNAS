import numpy as np
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms

from .utils import Cutout

def _data_transforms_imagenet(train):
  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  if train:
    transform = transforms.Compose([
      transforms.RandomResizedCrop(224),
      transforms.RandomHorizontalFlip(),
      transforms.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.2),
      transforms.ToTensor(),
      normalize,
    ])
  else:
    transform = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      normalize,
    ])
  return transform

class IMAGENET(dset.ImageFolder):
    def __init__(self, root, train, transform='default'):
        if transform == 'default':
            transform = _data_transforms_imagenet(train)
        assert isinstance(transform, transforms.Compose), f"No implementation for transform as {transform}"
        super().__init__(root=root, transform=transform)
