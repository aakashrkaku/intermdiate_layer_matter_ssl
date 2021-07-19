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
from hist_cancer_supervised import Hist_Cancer_Supervised
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
from collections import OrderedDict
import sys
from resnet_new_bt import resnet50
from pytorch_lightning.metrics import FBeta
from breast_cancer_hist_dm import *

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class Projection(nn.Module):
    def __init__(self, in_dim,lambd=5e-5,scale_loss=1/32):
        super().__init__()
        # projector
        sizes = [in_dim, 2048, 2048,2048]
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)
        self.lambd=lambd
        self.scale_loss=scale_loss

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)
        

    def forward(self, y1, y2):
        z1 = self.projector(y1)
        z2 = self.projector(y2)
        batch_size = z1.shape[0]

        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(batch_size)
#         torch.distributed.all_reduce(c)

        # use --scale-loss to multiply the loss by a constant factor
        # see the Issues section of the readme
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum().mul(self.scale_loss)
        off_diag = off_diagonal(c).pow_(2).sum().mul(self.scale_loss)
        loss = self.lambd *on_diag + self.lambd * off_diag
#         print(on_diag)
#         print(off_diag)
        return loss
    
class Projection_mse(nn.Module):
    def __init__(self, lambd=0.25):
        super().__init__()
        # projector
        self.mse = nn.MSELoss()
        self.lambd = lambd

    def forward(self, y1, y2):
        loss = self.lambd * self.mse(y1,y2)
#         print(loss)
        return loss

class Moco_v2(pl.LightningModule):
    """
    PyTorch Lightning implementation of `Moco <https://arxiv.org/abs/2003.04297>`_
    Paper authors: Xinlei Chen, Haoqi Fan, Ross Girshick, Kaiming He.
    Code adapted from `facebookresearch/moco <https://github.com/facebookresearch/moco>`_ to Lightning by:
        - `William Falcon <https://github.com/williamFalcon>`_
    Example::
        from pl_bolts.models.self_supervised import Moco_v2
        model = Moco_v2()
        trainer = Trainer()
        trainer.fit(model)
    CLI command::
        # cifar10
        python moco2_module.py --gpus 1
        # imagenet
        python moco2_module.py
            --gpus 8
            --dataset imagenet2012
            --data_dir /path/to/imagenet/
            --meta_dir /path/to/folder/with/meta.bin/
            --batch_size 32
    """

    def __init__(
        self,
        base_encoder: Union[str, torch.nn.Module] = 'resnet18',
        emb_dim: int = 128,
        num_negatives: int = 65536,
        encoder_momentum: float = 0.999,
        softmax_temperature: float = 0.07,
        learning_rate: float = 0.03,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        data_dir: str = './',
        batch_size: int = 256,
        use_mlp: bool = False,
        num_workers: int = 8,
        *args,
        **kwargs
    ):
        """
        Args:
            base_encoder: torchvision model name or torch.nn.Module
            emb_dim: feature dimension (default: 128)
            num_negatives: queue size; number of negative keys (default: 65536)
            encoder_momentum: moco momentum of updating key encoder (default: 0.999)
            softmax_temperature: softmax temperature (default: 0.07)
            learning_rate: the learning rate
            momentum: optimizer momentum
            weight_decay: optimizer weight decay
            datamodule: the DataModule (train, val, test dataloaders)
            data_dir: the directory to store data
            batch_size: batch size
            use_mlp: add an mlp to the encoders
            num_workers: workers for the loaders
        """

        super().__init__()
        self.save_hyperparameters()

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q, self.encoder_k = self.init_encoders(base_encoder)

        if use_mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(emb_dim, num_negatives))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.p1 = Projection_mse()
        self.p2 = Projection_mse()
        self.p3 = Projection_mse()
        self.p4 = Projection_mse()
        

    def init_encoders(self, base_encoder):
        """
        Override to add your own encoders
        """
        encoder_q = resnet50(num_classes=self.hparams.emb_dim,zero_init_residual=True)
        encoder_k = resnet50(num_classes=self.hparams.emb_dim,zero_init_residual=True)

        return encoder_q, encoder_k

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            em = self.hparams.encoder_momentum
            param_k.data = param_k.data * em + param_q.data * (1. - em)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        if self.trainer.use_ddp or self.trainer.use_ddp2:
            keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.hparams.num_negatives % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.hparams.num_negatives  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):  # pragma: no cover
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):  # pragma: no cover
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, img_q, img_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q,q1,q2,q3,q4 = self.encoder_q(img_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            if self.trainer.use_ddp or self.trainer.use_ddp2:
                img_k, idx_unshuffle = self._batch_shuffle_ddp(img_k)

            k,k1,k2,k3,k4 = self.encoder_k(img_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            if self.trainer.use_ddp or self.trainer.use_ddp2:
                k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.hparams.softmax_temperature

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long)
        labels = labels.type_as(logits)

#         dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels, q1,q2,q3,q4,k1,k2,k3,k4

    def training_step(self, batch, batch_idx):
        # in STL10 we pass in both lab+unl for online ft
        if self.trainer.datamodule.name == 'stl10':
            # labeled_batch = batch[1]
            unlabeled_batch = batch[0]
            batch = unlabeled_batch

        (img_1, img_2), _ = batch

        output, target,q1,q2,q3,q4,k1,k2,k3,k4 = self(img_q=img_1, img_k=img_2)
        loss = F.cross_entropy(output.float(), target.long())
#         print('Main loss = {}'.format(loss))
        loss+= self.p1(q1,k1)
        loss+= self.p2(q2,k2)
        loss+= self.p3(q3,k3)
        loss+= self.p4(q4,k4)

        acc1, acc5 = precision_at_k(output, target, top_k=(1, 5))

        log = {'train_loss': loss, 'train_acc1': acc1, 'train_acc5': acc5}
        self.log_dict(log)
        return loss

    def validation_step(self, batch, batch_idx):
        # in STL10 we pass in both lab+unl for online ft
        if self.trainer.datamodule.name == 'stl10':
            # labeled_batch = batch[1]
            unlabeled_batch = batch[0]
            batch = unlabeled_batch

        (img_1, img_2), labels = batch

        output, target,q1,q2,q3,q4,k1,k2,k3,k4 = self(img_q=img_1, img_k=img_2)
        loss = F.cross_entropy(output.float(), target.long())
#         print('Main loss = {}'.format(loss))
        loss+= self.p1(q1,k1)
        loss+= self.p2(q2,k2)
        loss+= self.p3(q3,k3)
        loss+= self.p4(q4,k4)

        acc1, acc5 = precision_at_k(output, target, top_k=(1, 5))

        results = {'val_loss': loss, 'val_acc1': acc1, 'val_acc5': acc5}
        return results

    def validation_epoch_end(self, outputs):
        val_loss = mean(outputs, 'val_loss')
        val_acc1 = mean(outputs, 'val_acc1')
        val_acc5 = mean(outputs, 'val_acc5')

        log = {'val_loss': val_loss, 'val_acc1': val_acc1, 'val_acc5': val_acc5}
        self.log_dict(log)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            self.hparams.learning_rate,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay
        )
        return optimizer


class Probing_model(pl.LightningModule):

    def __init__(self,layer,from_scratch = True,mse_btwin = False, ckpt_path = None):
        super().__init__()
        self.from_scratch = from_scratch
        self.conv = resnet50(num_classes=128)
        if not self.from_scratch:
            if mse_btwin == 'moco-mse' or mse_btwin == 'moco-btwins':
                print("Loading {}".format(mse_btwin))
                backbone = Moco_v2.load_from_checkpoint(ckpt_path,strict=False)
            else:
                print("Loading normal mocov2")
                backbone = MocoV2.load_from_checkpoint(ckpt_path)
            wts = backbone.encoder_q.state_dict()
            self.conv.load_state_dict(wts)
        self.conv.fc = nn.Identity()
        self.layer = layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model_subset()
        
        if layer == 'layer1':
            fc_dim = 256
        elif layer == 'layer2':
            fc_dim = 512
        elif layer == 'layer3':
            fc_dim = 1024
        elif layer == 'layer4':
            fc_dim = 2048
        elif layer == 'all':
            fc_dim = 2048
        
        self.final_fc = nn.Linear(fc_dim,2)
        self.train_auc = FBeta(num_classes = 2)
        self.valid_auc = FBeta(num_classes = 2)

    def forward(self, x):
        with torch.no_grad():
            feat = self.conv(x)
        if self.layer != 'all':
            feat = self.avgpool(feat)
            feat = torch.flatten(feat,1)
        x = self.final_fc(feat)
        x = torch.log_softmax(x, dim=1)
        return x, feat

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits,_ = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        self.log('train_loss', loss)
        self.train_auc(logits, y)
        self.log('train_fscore', self.train_auc, on_step=False, on_epoch=True)
        
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits, _ = self.forward(x)
#         print(logits)
#         print(y)
        loss = self.cross_entropy_loss(logits, y)
        self.log('val_loss', loss)
        self.valid_auc(logits, y)
        self.log('val_fscore', self.valid_auc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
        
    def model_subset(self):
        if self.layer == 'all':
            pass
        else:
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
    


def main(args):
    dm = Breast_Cancer_DM(data_path = args.data_path,num_workers=40,\
                   batch_size=args.batch_size, data_size = args.train_frac) 
    dm.setup('a')
    
    if args.model_type == 'moco-mse':
        ckpt_path = args.ckpt_path
    elif args.model_type == 'moco-btwins':
        ckpt_path = args.ckpt_path
    elif args.model_type == 'moco-only':
        ckpt_path = args.ckpt_path
    else:
        ckpt_path = None
        
    prob_model = Probing_model(args.probe_layer,args.from_scratch,args.model_type, ckpt_path)
    filename = 'probing-new-'
    if args.model_type == 'moco-mse' or args.model_type == 'moco-btwins':
        filename += args.model_type
    else:
        filename += 'mocov-only'
    filename += '-'+args.probe_layer+'-'+str(args.from_scratch)+'-{epoch:02d}-{val_fscore:.3f}'
    checkpoint_callback = ModelCheckpoint(
    monitor='val_fscore',
    dirpath=args.save_path,
    filename=filename,
    save_top_k=3,
    mode='max')
    
    
    trainer = pl.Trainer(gpus=1,default_root_dir=args.save_path,\
                     callbacks=[checkpoint_callback])
    trainer.fit(prob_model, dm.train_dataloader(),dm.val_dataloader())
    

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
    parser.add_argument("--batch-size", default=64, type=int, help="train batch size")
    parser.add_argument("--train_frac",default = 1,type=float)
    
    # Add model arguments
    parser.add_argument("--save_path",type=str)
    parser.add_argument("--probe_layer", type=str, help = "one of 'layer1', 'layer2', 'layer3' or 'layer4'")
    parser.add_argument('--from_scratch', type=str2bool, nargs='?',const=True, default=False)
    parser.add_argument('--model_type',type=str, help = "one of 'moco-mse', 'moco-btwin', or 'moco-only'")
    parser.add_argument('--ckpt_path',type=str, help = "Path to check point of pretrained ssl model")
    
    # models.MODEL_REGISTRY[args.model]
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    main(args)