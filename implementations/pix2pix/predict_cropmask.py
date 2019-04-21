import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys
import cv2

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch

model_epoch = 35

parser = argparse.ArgumentParser()
parser.add_argument('--img_height', type=int, default=256, help='size of image height')
parser.add_argument('--img_width', type=int, default=256, help='size of image width')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False

# Initialize generator and discriminator
generator = GeneratorUNet(in_channels=3, out_channels=1)
discriminator = Discriminator(in_channels=4)

if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()

generator.load_state_dict(torch.load('/media/arg_ws3/5E703E3A703E18EB/research/pix2pix_cropmask/saved_models/rgb/generator_10.pth'))
#discriminator.load_state_dict(torch.load('saved_models/sparse2dense/discriminator_%d.pth'%(model_epoch)))

# Configure dataloaders
transforms_A = [ transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
transforms_B = [ #transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
                transforms.ToTensor() ]

data_transform = transforms.Compose(transforms_B)

# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def sample_images():
    """Saves a generated sample from the validation set"""
    prev_time = time.time()
    image = cv2.imread("/media/arg_ws3/5E703E3A703E18EB/data/subt_all/image/drill/scene000002/12.jpg")
    # image = cv2.imread("/media/arg_ws3/5E703E3A703E18EB/data/d435_mm/empty/0_depth.png", cv2.IMREAD_ANYDEPTH)
    image = cv2.resize(image, (opt.img_height, opt.img_width))
    '''image = cv2.resize(image, (opt.img_height, opt.img_width))
    image = torch.tensor(image)
    image = image.permute(2, 0, 1).unsqueeze(dim=0)
    print(image.shape)
    my_img = Variable(image.type(Tensor))'''
    '''
    image = image/1000.
    depth_max = image.max()
    depth_min = image.min()
    image = (image - depth_min)/(depth_max - depth_min)
    pil_im = torch.tensor(image).unsqueeze(dim=0).unsqueeze(dim=0)
    '''
    pil_im = data_transform(image).unsqueeze(dim=0)
    # pil_im = Image.fromarray(image)
    # pil_im = data_transform(pil_im)
    # pil_im = pil_im.unsqueeze(0)

    my_img = Variable(pil_im.type(Tensor))
    my_img_fake = generator(my_img)
    my_img_fake = my_img_fake.squeeze(0).detach().cpu()

    pil_ = my_img_fake.mul(255).clamp(0, 255).byte().permute(1, 2, 0)
    #print(pil_)
    pil_ = np.array(pil_)
    pil_ = pil_[...,::-1]
    pil_ = cv2.resize(pil_, (640, 480))
    '''for i in range(640):
        for j in range(480):
            if pil_[j][i]!=0 and pil_[j][i]>5:
                print(pil_[j][i])'''
    print(pil_.dtype)
    #print(pil_)
    cv2.imwrite("crop_mask.jpg", pil_)
    print("Hz: ", 1./(time.time() - prev_time))
    save_image(my_img_fake.data, 'crop_mask.png', nrow=1, normalize=False)

sample_images()