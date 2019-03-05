import argparse
import os
import numpy as np
import math
import itertools
import sys
import datetime
import time
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

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--dataset_name', type=str, default="edges2shoes", help='name of the dataset')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_height', type=int, default=320, help='size of image height')
parser.add_argument('--img_width', type=int, default=320, help='size of image width')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=100, help='interval between sampling of images from generators')
parser.add_argument('--checkpoint_interval', type=int, default=-1, help='interval between model checkpoints')
opt = parser.parse_args()
print(opt)

# Create sample and checkpoint directories
os.makedirs('images/%s' % opt.dataset_name, exist_ok=True)
os.makedirs('saved_models/%s' % opt.dataset_name, exist_ok=True)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

# Losses
adversarial_loss = torch.nn.MSELoss()
cycle_loss = torch.nn.L1Loss()
pixelwise_loss = torch.nn.L1Loss()

cuda = True if torch.cuda.is_available() else False

# Calculate output of image discriminator (PatchGAN)
patch = (1, opt.img_height//2**3, opt.img_width//2**3)

# Initialize generator and discriminator
G_AB = GeneratorUNet()
G_BA = GeneratorUNet()
D_A = Discriminator()
D_B = Discriminator()

if cuda:
    G_AB = G_AB.cuda()
    G_BA = G_BA.cuda()
    D_A = D_A.cuda()
    D_B = D_B.cuda()
    adversarial_loss.cuda()
    cycle_loss.cuda()
    pixelwise_loss.cuda()

pth_num = 13
if True:
    # Load pretrained models
    G_AB.load_state_dict(torch.load('saved_models/%s/G_AB_%d.pth' % (opt.dataset_name, pth_num)))
    G_BA.load_state_dict(torch.load('saved_models/%s/G_BA_%d.pth' % (opt.dataset_name, pth_num)))
    D_A.load_state_dict(torch.load('saved_models/%s/D_A_%d.pth' % (opt.dataset_name, pth_num)))
    D_B.load_state_dict(torch.load('saved_models/%s/D_B_%d.pth' % (opt.dataset_name, pth_num)))
else:
    # Initialize weights
    G_AB.apply(weights_init_normal)
    G_BA.apply(weights_init_normal)
    D_A.apply(weights_init_normal)
    D_B.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()),
                                lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Input tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

data_transform = transforms.Compose(
                [ transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
    )

def sample_images(batches_done):
    """Saves a generated sample from the validation set"""
    image = cv2.imread("0000021.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(image)
    pil_im = data_transform(pil_im)
    pil_im = pil_im.unsqueeze(0)

    real_A = Variable(pil_im.type(Tensor))
    fake_B = G_AB(real_A)
    real_B = Variable(pil_im.type(Tensor))
    fake_A = G_BA(real_B)
    my_img_fake = fake_B.squeeze(0).detach().cpu()
    save_image(my_img_fake, 'test.png', nrow=8, normalize=True)

sample_images(1)