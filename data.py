import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cv2
from tqdm import tqdm
from glob import glob
from PIL import Image
from skimage.transform import resize
from sklearn.model_selection import train_test_split, KFold

import shutil

import torch
from torch.utils.data import DataLoader,Dataset
import torch.nn as nn
import albumentations
import torchvision
from torchvision import transforms, models

import random
from PIL import Image

ROOT = "ultrasound-nerve-segmentation"
trainpath = "./data/train/"
#testpath = "ultrasound-nerve-segmentation/test/"

masks = [os.path.join(trainpath,i) for i in os.listdir(trainpath) if "mask" in i]
imgs = [i.replace("_mask","") for i in masks]

df = pd.DataFrame({"Image":imgs,"Mask":masks})

df_train, df_val = train_test_split(df,test_size = 0.2)


def trainaugs():
    transform =  [
                albumentations.Resize(height=128,width=128,interpolation=Image.BILINEAR),
                albumentations.HorizontalFlip(),
                albumentations.VerticalFlip()
            ]
    return albumentations.Compose(transform)

def valaugs():
    transform = [
                albumentations.Resize(height=128,width=128,interpolation=Image.BILINEAR)
            ]
    return albumentations.Compose(transform)

class Custom_dataset(Dataset):
    def __init__(self, imagespath, maskspath, augment=None):
        self.imagespath = imagespath
        self.maskspath = maskspath
        self.augment = augment

    def __len__(self):
        return len(self.imagespath)

    def __getitem__(self, idx):
        image = cv2.cvtColor(cv2.imread(self.imagespath[idx]), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(self.maskspath[idx]), cv2.COLOR_BGR2RGB)

        if self.augment:
            sample = self.augment(image=image, mask=mask)
            image, mask = sample['image']/255, sample['mask']/255

            image = image.transpose(2,0,1)
            mask = mask.transpose(2, 0, 1)
        return image, mask


traindata = Custom_dataset(imagespath = df_train['Image'].tolist(),
                            maskspath = df_train['Mask'].tolist(),
                            augment = trainaugs())


validationdata = Custom_dataset(imagespath = df_val['Image'].tolist(),
                            maskspath = df_val['Mask'].tolist(),
                            augment = valaugs())

trainloader = DataLoader(traindata,batch_size = 16,shuffle=True)
valloader = DataLoader(validationdata,batch_size=8,shuffle=False)