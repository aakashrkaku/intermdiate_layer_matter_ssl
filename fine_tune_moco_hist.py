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
from breast_cancer_hist_dm import *

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

    
class New_model(pl.LightningModule):

    def __init__(self,from_scratch = True, ll_only=False, ckpt_path = None, model_type = 'None'):
        super().__init__()
        self.from_scratch = from_scratch
        self.conv = resnet.resnet50(num_classes=128)
        if not self.from_scratch:
            if model_type == 'moco-mse' or model_type == 'moco-btwins':
                wts = self.get_moco_mse_btwins_wts(ckpt_path)
            else:
                backbone = MocoV2.load_from_checkpoint(ckpt_path)
                wts = backbone.encoder_q.state_dict()
            self.conv.load_state_dict(wts)
        self.conv.fc = nn.Identity()
        self.final_fc = nn.Linear(2048,2)
        self.train_F = FBeta(num_classes = 2)
        self.valid_F = FBeta(num_classes = 2)
        self.ll_only = ll_only
    
    def get_moco_mse_btwins_wts(self,ckpt_path):
        data_dict = torch.load(ckpt_path)
        state_dict_q = OrderedDict()
        for k,v in data_dict['state_dict'].items():
            if 'encoder_q' in k:
                state_dict_q[k[10:]] = v.cpu()
        return state_dict_q

    def forward(self, x):
        if self.ll_only:
            with torch.no_grad():
                x = self.conv(x)
        else:
            x = self.conv(x)
        x = self.final_fc(x)
        x = torch.log_softmax(x, dim=1)
        return x

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        self.log('train_loss', loss)
        self.train_F(logits, y)
        self.log('train_fscore', self.train_F, on_step=False, on_epoch=True)
        
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        self.log('val_loss', loss)
        self.valid_F(logits, y)
        self.log('val_fscore', self.valid_F, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
class Moco_v2():
    pass
class Projection_mse():
    pass
class Projection():
    pass

class Probing_model(pl.LightningModule):

    def __init__(self,wts_path,layer):
        super().__init__()
#         backbone = MocoV2.load_from_checkpoint('./lightning_logs/version_10225641/checkpoints/epoch=42.ckpt')
        self.wts = torch.load(wts_path)
        self.conv = resnet.resnet18(num_classes=128)
        self.load_wts()
        self.layer = layer
        self.model_subset()
        if layer == 'layer1':
            fc_dim = 262144
        elif layer == 'layer2':
            fc_dim = 131072
        elif layer == 'layer3':
            fc_dim = 65536
        elif layer == 'layer4':
            fc_dim = 32768
        
        for p in self.conv.parameters():
            p.requires_grad = False
        
        self.fc = nn.Linear(fc_dim,2)
        self.train_F = FBeta(num_classes = 2)
        self.valid_F = FBeta(num_classes = 2)
        
    def load_wts(self):
        new_model_state_dict = self.conv.state_dict()
        loading_state_dict = {}
        for k,v in new_model_state_dict.items():
            for k2,v2 in self.wts.items():
                if k in k2:
                    loading_state_dict[k]=v2
                else:
                    loading_state_dict[k]=v
        self.conv.load_state_dict(loading_state_dict)
        
    def model_subset(self):
        model_list = []
        flag = 0
        for k,i in enumerate(self.conv.named_children()):
            if flag ==0:
                model_list.append(i)
            else:
                break
            if i[0] == self.layer:
                flag = 1
        self.conv = nn.Sequential(OrderedDict(model_list))

    def forward(self, x):
        x = self.conv(x)
        B, F, H, W = x.shape
        x = x.view(B,-1)
        x = self.fc(x)
        x = torch.log_softmax(x, dim=1)
        return x

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        self.log('train_loss', loss)
        self.train_F(logits, y)
        self.log('train_fscore', self.train_F, on_step=False, on_epoch=True)
        
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        self.log('val_loss', loss)
        self.valid_F(logits, y)
        self.log('val_fscore', self.valid_F, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.fc.parameters(), lr=1e-3)
        return optimizer
    


def main(args):
    dm = Breast_Cancer_DM(data_path=args.data_path,num_workers=40, batch_size=args.batch_size, data_size = args.data_size) 
    dm.setup('a')
    
    if args.model_type == 'moco-mse':
        ckpt_path = args.ckpt_path
    elif args.model_type == 'moco-btwins':
        ckpt_path = args.ckpt_path
    elif args.model_type == 'moco-only':
        ckpt_path = args.ckpt_path
    else:
        ckpt_path = None
    
    finetune_model = New_model(args.from_scratch,args.ll_only,ckpt_path,args.model_type)
    
    if args.from_scratch:
        filename = 'supervised-'
    else:
        filename = args.model_type + '-'
    
    if args.ll_only:
        filename += 'll-'
        
    filename += '{epoch:02d}-{val_fscore:.3f}'    
    
    checkpoint_callback = ModelCheckpoint(
    monitor='val_fscore',
    dirpath=args.save_path,
    filename=filename,
    save_top_k=3,
    mode='max')
    
    trainer = pl.Trainer(gpus=1,default_root_dir=args.save_path,\
                     callbacks=[checkpoint_callback])
    trainer.fit(finetune_model, dm.train_dataloader(),dm.val_dataloader())
    

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
    parser.add_argument("--data-path", help="path to data directory")
    parser.add_argument("--batch-size", default=128, type=int, help="train batch size")
    parser.add_argument("--data-size", default=1, type=float, help="fraction of data")
    
    # Add model arguments
    parser.add_argument("--save_path",type=str)
#     parser.add_argument("--filename", default='finetuning_model-{epoch:02d}-{val_fscore:.2f}', type=str)
    parser.add_argument('--from_scratch', type=str2bool, nargs='?',const=True, default=False)
    parser.add_argument('--ll_only', type=str2bool, nargs='?',const=True, default=False)
    parser.add_argument('--model_type',type=str, help = "one of 'moco-mse', 'moco-btwin', or 'moco-only'")
    parser.add_argument('--ckpt_path',type=str, help = "Path to check point of pretrained ssl model")
    
    # Parse twice as model arguments are not known the first time
    parser = utils.add_logging_arguments(parser)
    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    args = get_args()
    print(args)
    main(args)