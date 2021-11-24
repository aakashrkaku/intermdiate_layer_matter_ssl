# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 11:58:07 2017
@author: Biagio Brattoli
"""
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
from torch.utils.data.dataloader import default_collate



class Hist_Cancer_Supervised(data.Dataset):
    def __init__(self, data_path, txt_list, train = False,breast_hist=False,ssl_transforms=None):
        self.data_path = data_path
        self.names, self.sup_labels = self.__dataset_info(txt_list,breast_hist)
        self.N = len(self.names)
#         self.breast_hist=breast_hist

        self.__image_transformer = transforms.Compose([
            transforms.Resize(256, Image.BILINEAR),
            transforms.CenterCrop(255)])
        self.__augment_tile = transforms.Compose([
#             transforms.RandomCrop(64),
#             transforms.Resize((75, 75), Image.BILINEAR),
            transforms.RandomHorizontalFlip(p=0.5),
#             transforms.ColorJitter(brightness=[0.5, 1.5], contrast=0.4, saturation=0.4, hue=0.2),
#             transforms.Lambda(rgb_jittering),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            # std =[0.229, 0.224, 0.225])
        ])
        self.__no_augment_tile = transforms.Compose([
#             transforms.RandomCrop(64),
#             transforms.Resize((75, 75), Image.BILINEAR),
#             transforms.RandomHorizontalFlip(p=0.5),
#             transforms.ColorJitter(brightness=[0.5, 1.5], contrast=0.4, saturation=0.4, hue=0.2),
#             transforms.Lambda(rgb_jittering),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            # std =[0.229, 0.224, 0.225])
        ])
        self.train = train
        self.ssl_transforms=ssl_transforms

    def __getitem__(self, index):
        framename = self.data_path + '/' + self.names[index]
        s_label = self.sup_labels[index]
        img = Image.open(framename).convert('RGB')
#         if np.random.rand() < 0.30:
#             img = img.convert('LA').convert('RGB')

        if img.size[0] != 255:
            img = self.__image_transformer(img)
        if self.ssl_transforms:
            data = self.ssl_transforms(img)
        else:
            if self.train:
                data = self.__augment_tile(img)
            else:
                data = self.__no_augment_tile(img)
                    
        labels = torch.tensor(s_label)

        return (data, labels)

    def __len__(self):
        return len(self.names)

    def __dataset_info(self, txt_labels,breast_hist):
#         with open(txt_labels, 'r') as f:
#             images_list = f.readlines()

#         file_names = []
#         labels = []
#         for row in images_list:
#             row = row.split(' ')
#             print(row)
#             file_names.append(row[0])
#             labels.append(int(row[1]))
        labels_file = pd.read_csv(txt_labels)
        if breast_hist:
            file_names = np.array(labels_file.iloc[:,0])
        else:
            file_names = np.array(labels_file.iloc[:,0]+'.tif')
        labels = np.array(labels_file.iloc[:,1])

        return file_names, labels
    
    def _collate_fun(self,batch):
        batch = default_collate(batch)
        return batch

def rgb_jittering(im):
    im = np.array(im, 'int32')
    for ch in range(3):
        im[:, :, ch] += np.random.randint(-2, 2)
    im[im > 255] = 255
    im[im < 0] = 0
    return im.astype('uint8')