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
from sklearn.metrics import f1_score, roc_auc_score
import argparse
import utils

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

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
        self.train_dataset = Hist_Cancer_Supervised(self.data_path, './../datasets/breast-hist/train_split.csv',\
                                                    train = True,breast_hist=True)
        if self.data_size < 1:
            n = len(self.train_dataset)
            idx = torch.arange(n)
            np.random.shuffle(idx)
            selected_idx = idx[:int(n*self.data_size)]
            self.train_dataset = Subset(self.train_dataset,selected_idx)
            print("Using {} fraction of the data set for training".format(self.data_size))
        
        self.val_dataset = Hist_Cancer_Supervised(self.data_path, './../datasets/breast-hist/val_split.csv',\
                                                  breast_hist=True)
        self.test_dataset = Hist_Cancer_Supervised(self.data_path, './../datasets/breast-hist/test_split.csv',\
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
    
class FTTrainTransformsBreastCancer:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """

    def __init__(self, height=255):
        # image augmentation functions
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(height, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
#             transforms.Resize(height),
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

    def __call__(self, inp):
        q = self.train_transform(inp)
        return q


class FTEvalTransformsBreastCancer:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """
    def __init__(self, height=255):
        self.test_transform = transforms.Compose([
            transforms.Resize(height),
            transforms.CenterCrop(height),
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


class New_model(pl.LightningModule):

    def __init__(self,from_scratch = True,mse_btwin = False, ckpt_path = None, only_ll = False):
        super().__init__()
        self.from_scratch = from_scratch
        self.conv = resnet_new_bt.resnet50(num_classes=128)
        self.only_ll = only_ll
        if not self.from_scratch:
            if mse_btwin == 'moco-mse' or mse_btwin == 'moco-btwin':
                backbone = Moco_v2.load_from_checkpoint(ckpt_path, strict=False)
            else:
                backbone = MocoV2.load_from_checkpoint(ckpt_path)
            wts = backbone.encoder_q.state_dict()
            self.conv.load_state_dict(wts)
        self.conv.fc = nn.Identity()
        self.final_fc = nn.Linear(2048,14)
        self.train_F = FBeta(num_classes = 2)
        self.valid_F = FBeta(num_classes = 2)

    def forward(self, x):
        if self.only_ll:
            with torch.no_grad():
                feat, (x1,x2,x3,x4) = self.conv(x)
        else:
            feat, (x1,x2,x3,x4) = self.conv(x)
        x = self.final_fc(feat)
        x = torch.log_softmax(x, dim=1)
        return x, (x1,x2,x3,x4)
    
    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits,_ = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        self.log('train_loss', loss)
        self.train_F(logits, y)
        self.log('train_fscore', self.train_F, on_step=False, on_epoch=True)
        
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits,_ = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        self.log('val_loss', loss)
        self.valid_F(logits, y)
        self.log('val_fscore', self.valid_F, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
def bootstrap_performance(targets,preds,average_type = 'macro',num_sets = 500):
    all_scores = []
    size = targets.shape[0]
    idx = np.arange(size) 
    for i in range(num_sets):
        sel_idx = np.random.choice(idx,size=size)
        all_scores.append(f1_score(targets[sel_idx],preds[sel_idx],average = average_type))
    mean_score = np.mean(all_scores)
    std_score = np.std(all_scores)
    return mean_score, mean_score - 1.96*std_score/num_sets**0.5, mean_score + 1.96*std_score/num_sets**0.5
    
def main(args):
    dm = Chest_Xray_DM(data_path = args.data_path,num_workers=40,\
                   batch_size=args.batch_size, train_frac = args.data_size) 
    dm.setup('a')
    dm.train_transforms = FTTrainTransformsBreastCancer()
    dm.val_transforms = FTEvalTransformsBreastCancer()
    dm.test_transforms = FTEvalTransformsBreastCancer()
    
    save_path = args.save_path + str(args.data_size) + '/'
    
    finetune_model = New_model(args.from_scratch, args.mse_btwin, args.ckpt_path, args.only_ll)
    
    
    if args.from_scratch:
        filename = 'supervised-{epoch:02d}-{val_fscore:.3f}'
    else:
        if args.mse_btwin == 'moco-mse' or args.mse_btwin == 'moco-btwin':
            filename = args.mse_btwin
        else:
            filename = 'mocov2only'
    if args.file_extension:
        filename += args.file_extension
        
    if args.only_ll:
        filename += '-ll'
    filename += '-{epoch:02d}-{val_fscore:.3f}'
    checkpoint_callback = ModelCheckpoint(
    monitor='val_fscore',
    dirpath=save_path,
    filename=filename,
    save_top_k=3,
    mode='max')
    
    
    if torch.cuda.is_available():
        trainer = pl.Trainer(gpus=1,default_root_dir=args.save_path,\
                         callbacks=[checkpoint_callback])
    else:
        trainer = pl.Trainer(default_root_dir=args.save_path,\
                         callbacks=[checkpoint_callback])
        
    if not args.no_training:
        trainer.fit(finetune_model, dm.train_dataloader(),dm.val_dataloader())
    
    # Inference
    all_feats_l1 = []
    all_feats_l2 = []
    all_feats_l3 = []
    all_feats_l4 = []
    all_ys=[]
    all_logits = []
    for i in dm.test_dataloader():
        logit,feats = model(i[0].to(device))
        all_ys.append(i[1].numpy())
        all_logits.append(logit.data.cpu().numpy())
        all_feats_l1.append(feats[0].data.cpu().numpy())
        all_feats_l2.append(feats[1].data.cpu().numpy())
        all_feats_l3.append(feats[2].data.cpu().numpy())
        all_feats_l4.append(feats[3].data.cpu().numpy())
        
    all_feats_l1 = np.concatenate(all_feats_l1)
    all_feats_l2 = np.concatenate(all_feats_l2)
    all_feats_l3 = np.concatenate(all_feats_l3)
    all_feats_l4 = np.concatenate(all_feats_l4)
    all_ys = np.concatenate(all_ys)
    all_logits = np.concatenate(all_logits)
    
    mean_auc, ll_auc, ul_auc = bootstrap_performance(all_ys, all_logits)
    
    if args.no_training:
        dict_name = save_path+filename+'_not_trained_preds.p'
    else:
        dict_name = save_path+filename+'_preds.p'
    
    pickling({'l1':all_feats_l1,'l2':all_feats_l2,'l3':all_feats_l3,'l4':all_feats_l4,\
          'logits':all_logits,'ys':all_ys, 'mean_auc': mean_auc, 'll_auc': ll_auc, 'ul_auc':ul_auc },dict_name)
    

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
    parser.add_argument("--data_size", default=1, type=float, help="fraction of data, 1 means 100% and 0.01 means 1%")
    parser.add_argument("--batch-size", default=16, type=int, help="train batch size")
    
    # Add model arguments
    parser.add_argument("--save_path", default='./lightning_logs/finetuning_model/',type=str, help = "path to directory for saving fine-tuned models")
    parser.add_argument("--file_extension", default=None, type=str, help = "any file extensions you want to add to the model saving file name")
    
    parser.add_argument("--mse_btwin", default=None, type=str, help = "could be one of 'moco-mse', 'moco-btwin' or 'moco-only'")
    parser.add_argument('--from_scratch', type=str2bool, nargs='?',const=True, default=False, help = "if true then a fully supervised model with random intialization will be trained")
    parser.add_argument('--only_ll', type=str2bool, nargs='?',const=True, default=False, help = "if true, then it will only train the last linear layer (rest of the network would be freezed)")
    parser.add_argument("--ckpt_path", type=str, help = "path to the ssl check point model used of intializing wts. of the model")
    parser.add_argument('--no_training', type=str2bool, nargs='?',const=True, default=False, help = "if true, then models would not be fine-tuned")
    
    # Parse twice as model arguments are not known the first time
    parser = utils.add_logging_arguments(parser)
    args, _ = parser.parse_known_args()
    # models.MODEL_REGISTRY[args.model]
    # args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    print(args)
    main(args)

