from comet_ml import Experiment
from pytorch_lightning.loggers import CometLogger
import pytorch_lightning as pl
from pl_bolts.models.self_supervised import MocoV2
from pl_bolts.models.self_supervised.moco import Moco2TrainCIFAR10Transforms, Moco2EvalCIFAR10Transforms
from torchvision import transforms
from pl_bolts.transforms.self_supervised import RandomTranslateWithReflect, Patchify
from pl_bolts.datamodules import CIFAR10DataModule
import random
from PIL import ImageFilter
import torch.nn.functional as F
import torch.nn as nn
from pytorch_lightning.metrics import FBeta
import torch
import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from chest_xray_supervised import Chest_Xray_Supervised
import torch
from torchvision.models import resnet
from pytorch_lightning.callbacks import ModelCheckpoint
from resnet_new import resnet50
from resnet_new_bt import resnet50
from typing import Union
from pl_bolts.metrics import mean, precision_at_k
from pytorch_lightning.metrics import Metric
from sklearn.metrics import roc_auc_score
import argparse
import utils

class Chest_Xray_DM(LightningDataModule):

    def __init__(self, data_path,num_workers: int = 16,batch_size: int = 64, train_frac = 1):
        super().__init__()
        self.data_path = data_path
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.train_frac = train_frac
        

    def setup(self,stage):
        # called on every GPU
        self.train_dataset = Chest_Xray_Supervised(self.data_path, 'chest_xray_14_train_list.txt',\
                                                    train = True, train_frac = self.train_frac)
        self.val_dataset = Chest_Xray_Supervised(self.data_path, 'chest_xray_14_val_list.txt')
        self.test_dataset = Chest_Xray_Supervised(self.data_path, 'chest_xray_14_test_list.txt')

    def train_dataloader(self):
        if self.train_transforms is not None:
            transforms = self.train_transforms
            self.train_dataset.ssl_transforms = transforms
        return DataLoader(self.train_dataset, batch_size=self.batch_size,\
            shuffle=True,\
            num_workers=self.num_workers,\
            drop_last=True,\
            pin_memory=True)

    def val_dataloader(self):
        if self.val_transforms is not None:
            transforms = self.val_transforms 
            self.val_dataset.ssl_transforms = transforms
        return DataLoader(self.val_dataset, batch_size=self.batch_size,\
            shuffle=False,\
            num_workers=self.num_workers,\
            pin_memory=True,\
            drop_last=True)

    def test_dataloader(self):
        if self.test_transforms is not None:
            transforms = self.test_transforms 
            self.test_dataset.ssl_transforms = transforms
        return DataLoader(self.test_dataset, batch_size=self.batch_size,\
            shuffle=False,\
            num_workers=self.num_workers,\
            pin_memory=True,\
            drop_last=True)

    def num_classes(self):
        """
        Return:
            10
        """
        return 2
    
class Moco2TrainTransformsChestXray:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """

    def __init__(self, height=256):
        # image augmentation functions
        self.train_transform = transforms.Compose([
#             transforms.RandomResizedCrop(height, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.Resize(height),
            transforms.RandomRotation(30),
#             transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

    def __call__(self, inp):
        q = self.train_transform(inp)
        k = self.train_transform(inp)
        return q, k


class Moco2EvalTransformsChestXray:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """
    def __init__(self, height=256):
        self.test_transform = transforms.Compose([
            transforms.Resize(height),
#             transforms.CenterCrop(height),
            transforms.ToTensor(),
        ])

    def __call__(self, inp):
        q = self.test_transform(inp)
        k = self.test_transform(inp)
        return q, k

class FTTrainTransformsChestXray:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """

    def __init__(self, height=256):
        # image augmentation functions
        self.train_transform = transforms.Compose([
#             transforms.RandomResizedCrop(height, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
#             transforms.Resize(height),
            transforms.RandomRotation(30),
#             transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

    def __call__(self, inp):
        q = self.train_transform(inp)
        return q


class FTEvalTransformsChestXray:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """
    def __init__(self, height=256):
        self.test_transform = transforms.Compose([
#             transforms.Resize(height),
#             transforms.CenterCrop(height),
            transforms.ToTensor(),
        ])

    def __call__(self, inp):
        q = self.test_transform(inp)
        return q


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x