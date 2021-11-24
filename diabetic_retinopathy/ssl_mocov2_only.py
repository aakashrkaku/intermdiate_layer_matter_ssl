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
import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.sampler import WeightedRandomSampler, Sampler
import argparse
from typing import Union
from pl_bolts.metrics import mean, precision_at_k
import os
from collections import Counter
import torch.distributed as dist
import math
import random
from pytorch_lightning.callbacks import ModelCheckpoint

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

class BaselineDataModule(pl.LightningDataModule):
    def __init__(self, data_dir='../jama16-retina-replication/data/eyepacs/bin2/', batch_size=8, num_workers = 8):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_sizes = {}

#         num_train = 57146
#         indices = list(range(num_train))
#         valid_size = 0.2
#         split = int(np.floor(valid_size * num_train))
            
#         np.random.shuffle(indices)
#         self.train_idx, self.valid_idx = indices[split:], indices[:split]

    def setup(self, stage = None):
        if stage == 'fit' or stage is None:
            self.train_dataset  = datasets.ImageFolder(
                    os.path.join(self.data_dir, 'train'), 
                    transform=self.train_transforms)
#             self.train_dataset = custom_subset(t_dataset, self.train_idx)

            self.val_dataset = datasets.ImageFolder(
                    os.path.join(self.data_dir, 'validation'), 
                    transform=self.val_transforms)
            
#             self.val_dataset = custom_subset(v_dataset, self.valid_idx)
            self.dataset_sizes['train'] = len(self.train_dataset)
            self.dataset_sizes['val'] = len(self.val_dataset)
            
                        
        if stage == 'test' or stage is None:
            self.test_dataset= datasets.ImageFolder(os.path.join(self.data_dir, 'test'), 
                                                    transform=self.test_transforms)
            self.dataset_sizes['test'] = len(self.test_dataset)
        
#         print(self.dataset_sizes)
           
            
    def _sampler(self):
        targets = self.train_dataset.targets
        class_sample_counts = torch.tensor(
            [(targets == t).sum() for t in torch.unique(targets, sorted=True)])
#         print(class_sample_counts)
        weight = 1. / class_sample_counts.double()
        samples_weight = torch.tensor([weight[t] for t in targets])

        sampler = WeightedRandomSampler(
                        weights=samples_weight,
                        num_samples=len(samples_weight),
                        replacement=True)
        return sampler

    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          shuffle = True,
#                           sampler=weighted_sampler, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers,
#                           drop_last =True,
                          pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
                          shuffle=False,
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers,
#                           drop_last = True,
                          pin_memory = True)

    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, 
                  batch_size = self.batch_size, 
                  num_workers= self.num_workers,
                  shuffle = False,
#                   drop_last = True,
                  pin_memory = True)
    
    def num_classes(self):
        return 2

class Moco2TrainTransformsDR:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """

    def __init__(self, height=299):
        # image augmentation functions
        self.train_transform = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

    def __call__(self, inp):
        q = self.train_transform(inp)
        k = self.train_transform(inp)
        return q, k


class Moco2EvalTransformsDR:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """
    def __init__(self, height=299):
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __call__(self, inp):
        q = self.test_transform(inp)
        k = self.test_transform(inp)
        return q, k
    
class FTTrainTransforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """
    def __init__(self, height=256):
        # image augmentation functions
        self.train_transform = transforms.Compose([
            transforms.RandomApply([
                    transforms.ColorJitter(brightness=0, contrast=0.4, saturation=0, hue=0.1)  # not strengthened
                ], p=0.8),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ])

    def __call__(self, inp):
        q = self.train_transform(inp)
        return q


class FTEvalTransforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    
    """
    def __init__(self, height=256):
        self.test_transform = transforms.Compose([
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

class custom_subset(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        labels(sequence) : targets as required for the indices. will be the same length as indices
    """
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.targets = torch.tensor(dataset.targets)[torch.tensor(indices)]
        self.indices = indices
        
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.targets)
        
    
def main(args):
    dm = BaselineDataModule(data_path = args.data_path,num_workers=40, \
                   batch_size=args.batch_size) 
    
    data_module.train_transforms = Moco2TrainTransformsDR(height=299)
    data_module.val_transforms = Moco2EvalTransformsDR(height=299)
    data_module.test_transforms = Moco2EvalTransformsDR(height=299)
    
    model = MocoV2(base_encoder = 'resnet50', datamodule=dm)
    
    
    checkpoint_callback = ModelCheckpoint(
                                dirpath=args.save_path+'mocov2_all_augs_dr/',
                                filename='{epoch}-{val_loss:.3f}',
                                monitor="val_loss", 
                                mode="min",
                                save_top_k=3)
        
    if torch.cuda.is_available():
        trainer = pl.Trainer(gpus=1,checkpoint_callback = checkpoint_callback,)
    else:
        trainer = pl.Trainer(checkpoint_callback = checkpoint_callback,)
    trainer.fit(model, dm)
    

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)

    # Add data arguments
    parser.add_argument("--data-path", help="path to data directory", type=str)
    parser.add_argument("--batch-size", default=8, type=int, help="train batch size")
    
    # Add model arguments
    parser.add_argument("--save-path", help="path to saving model directory", type=str)
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    main(args)