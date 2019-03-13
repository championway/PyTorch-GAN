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

model_epoch = 190

parser = argparse.ArgumentParser()
parser.add_argument('--img_height', type=int, default=256, help='size of image height')
parser.add_argument('--img_width', type=int, default=256, help='size of image width')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False

# Initialize generator and discriminator
generator = GeneratorUNet()
discriminator = Discriminator()

if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()

generator.load_state_dict(torch.load('saved_models/subt/generator_%d.pth'%(model_epoch)))
discriminator.load_state_dict(torch.load('saved_models/subt/discriminator_%d.pth'%(model_epoch)))

# Configure dataloaders
transforms_A = [ transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
transforms_B = [ transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
                transforms.ToTensor() ]

data_transform = transforms.Compose(transforms_B)

# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def sample_images():
    """Saves a generated sample from the validation set"""
    prev_time = time.time()
    image = cv2.imread("out1.png",-1)
    #image = np.transpose(image, (1, 0))
    print(image.shape)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image/1000.
    #image = np.asarray(image, dtype=np.uint16)

    #print(image)
    pil_im = Image.fromarray(image).convert('RGB')
    pil_im = data_transform(pil_im)
    pil_im = pil_im.unsqueeze(0)

    my_img = Variable(pil_im.type(Tensor))
    my_img_fake = generator(my_img)
    my_img_fake = my_img_fake.squeeze(0).detach().cpu()
    print(my_img_fake.shape)
    pil_ = my_img_fake.mul(255).byte()
    pil_ = np.asarray(pil_, dtype=np.uint8)
    #pil_ = cv2.cvtColor(pil_, cv2.COLOR_RGB2BGR)
    #pil_ = np.array(pil_)
    pil_ = np.transpose(pil_, (1, 2, 0))
    print(pil_)
    print(pil_.shape)
    cv2.imwrite("ss.jpg", pil_)

    save_image(my_img_fake, 'sss.jpg', normalize=True)
    print("Hz: ", 1./(time.time() - prev_time))

sample_images()