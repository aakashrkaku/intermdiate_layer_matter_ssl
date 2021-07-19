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

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from chest_xray_supervised import Chest_Xray_Supervised
from resnet_new_bt import resnet50
from typing import Union
from pl_bolts.metrics import mean, precision_at_k
from chest_xray_dm import *
from breast_cancer_hist_dm import *

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
    
    
def main(args):
    if args.dataset == 'chestxray':
        dm = Chest_Xray_DM(args.data_path,num_workers=40,\
                       batch_size=16, train_frac = args.data_size) 
        dm.setup('a')
        dm.train_transforms = Moco2TrainTransformsChestXray()
        dm.val_transforms = Moco2EvalTransformsChestXray()
        dm.test_transforms = Moco2EvalTransformsChestXray()
    elif args.dataset == 'histopathology':
        dm = Breast_Cancer_DM(args.data_path,num_workers=40, batch_size=32) 
        dm.setup('a')
        dm.train_transforms = Moco2TrainTransformsBreastCancer()
        dm.val_transforms = Moco2EvalTransformsBreastCancer()
        dm.test_transforms = Moco2EvalTransformsBreastCancer()
    elif args.dataset == "retinopathy":
        raise ValueError('Please build a dataloader and a datamodule for this dataset and make changes here.')
    
    model = MocoV2(datamodule=dm)
    
    trainer = pl.Trainer(gpus=1,default_root_dir=args.save_path)
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
    parser.add_argument("--dataset", type=str, help = "one of 'chestxray', 'histopathology' or 'retinopathy'")
    parser.add_argument("--data_path", type=str, help="path to data")
    # Add model arguments
    parser.add_argument("--save_path",type=str, help="path to save the checkpoint")
    
     parser = utils.add_logging_arguments(parser)
    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    args = get_args()
    print(args)
    main(args)