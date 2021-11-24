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
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from hist_cancer_supervised import Hist_Cancer_Supervised
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse
import utils
from torchvision.models import resnet
from collections import OrderedDict
from torch.utils.data import Subset
import numpy as np

class Breast_Cancer_DM(LightningDataModule):

    def __init__(self, data_path,num_workers: int = 16,batch_size: int = 64, data_size = 1):
        super().__init__()
        self.data_path = data_path
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.data_size = data_size
        

#     def prepare_data(self):
#         # called only on 1 GPU
#         download_dataset()
#         tokenize()
#         build_vocab()

    def setup(self,stage):
        # called on every GPU
#         transform=transforms.Compose([transforms.ToTensor()])
        self.train_dataset = Hist_Cancer_Supervised(self.data_path, 'train_split.csv',\
                                                    train = True,breast_hist=True)
        if self.data_size < 1:
            n = len(self.train_dataset)
            idx = torch.arange(n)
            np.random.shuffle(idx)
            selected_idx = idx[:int(n*self.data_size)]
            self.train_dataset = Subset(self.train_dataset,selected_idx)
            print("Using {} fraction of the data set for training".format(self.data_size))
        
        self.val_dataset = Hist_Cancer_Supervised(self.data_path, 'val_split.csv',\
                                                  breast_hist=True)
        self.test_dataset = Hist_Cancer_Supervised(self.data_path, 'test_split.csv',\
                                                   breast_hist=True)

#         self.train, self.val, self.test = load_datasets()
#         self.train_dims = self.train_dataset.next_batch.size()

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

class Moco2TrainTransformsBreastCancer:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """

    def __init__(self, height=128):
        # image augmentation functions
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(height, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

    def __call__(self, inp):
        q = self.train_transform(inp)
        k = self.train_transform(inp)
        return q, k


class Moco2EvalTransformsBreastCancer:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """
    def __init__(self, height=128):
        self.test_transform = transforms.Compose([
            transforms.Resize(height + 32),
            transforms.CenterCrop(height),
            transforms.ToTensor(),
        ])

    def __call__(self, inp):
        q = self.test_transform(inp)
        k = self.test_transform(inp)
        return q, k


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x