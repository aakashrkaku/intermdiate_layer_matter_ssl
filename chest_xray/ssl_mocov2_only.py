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
from torch.utils.data import DataLoader, random_split
from chest_xray_supervised import Chest_Xray_Supervised
import argparse
from typing import Union
from pl_bolts.metrics import mean, precision_at_k
from pytorch_lightning.callbacks import ModelCheckpoint

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

class Chest_Xray_DM(LightningDataModule):

    def __init__(self, data_path,num_workers: int = 16,batch_size: int = 64):
        super().__init__()
        self.data_path = data_path
        self.num_workers = num_workers
        self.batch_size = batch_size
        

#     def prepare_data(self):
#         # called only on 1 GPU
#         download_dataset()
#         tokenize()
#         build_vocab()

    def setup(self,stage):
        # called on every GPU
#         transform=transforms.Compose([transforms.ToTensor()])
        self.train_dataset = Chest_Xray_Supervised(self.data_path, 'chest_xray_14_train_list.txt',\
                                                    train = True, return_idx = False)
        self.val_dataset = Chest_Xray_Supervised(self.data_path, 'chest_xray_14_val_list.txt',return_idx = False)
        self.test_dataset = Chest_Xray_Supervised(self.data_path, 'chest_xray_14_test_list.txt',return_idx = False)

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
#             transforms.Resize(height),
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
#             transforms.Resize(height + 32),
#             transforms.CenterCrop(height),
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
        
    
def main(args):
    dm = Chest_Xray_DM(data_path = args.data_path,num_workers=40, \
                   batch_size=args.batch_size) 
    dm.setup('a')
    dm.train_transforms = Moco2TrainTransformsChestXray()
    dm.val_transforms = Moco2EvalTransformsChestXray()
    dm.test_transforms = Moco2EvalTransformsChestXray()
    
    model = MocoV2(base_encoder = 'resnet50',datamodule=dm)
    
    checkpoint_callback = ModelCheckpoint(
                                dirpath=args.save_path+'mocov2_all_augs_full_res_chest_xray/',
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
    parser.add_argument("--batch-size", default=16, type=int, help="train batch size")
    parser.add_argument("--save-path", help="path to saving model directory", type=str)
    
    # Add model arguments
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    main(args)