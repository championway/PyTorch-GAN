import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from datasets_cropmask import *

import torch.nn as nn
import torch.nn.functional as F
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--dataset_name', type=str, default="depth_rgb", help='name of the dataset')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch from which to start lr decay')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_height', type=int, default=256, help='size of image height')
parser.add_argument('--img_width', type=int, default=256, help='size of image width')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=1000, help='interval between sampling of images from generators')
parser.add_argument('--checkpoint_interval', type=int, default=10, help='interval between model checkpoints')
opt = parser.parse_args()
print(opt)
root = '/media/arg_ws3/5E703E3A703E18EB/research/pix2pix_cropmask/'
os.makedirs(root + 'images/%s' % opt.dataset_name, exist_ok=True)
os.makedirs(root + 'saved_models/%s' % opt.dataset_name, exist_ok=True)

cuda = True if torch.cuda.is_available() else False

# Loss functions
criterion_GAN = torch.nn.MSELoss()
criterion_pixelwise = torch.nn.L1Loss()

# Loss weight of L1 pixel-wise loss between translated image and real image
lambda_pixel = 100

# Calculate output of image discriminator (PatchGAN)
patch = (1, opt.img_height//2**4, opt.img_width//2**4)

# Initialize generator and discriminator
generator = GeneratorUNet(in_channels=4, out_channels=1)
discriminator = Discriminator(in_channels=5)

if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    criterion_GAN.cuda()
    criterion_pixelwise.cuda()

if opt.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load(root + 'saved_models/%s/generator_%d.pth' % (opt.dataset_name, opt.epoch)))
    discriminator.load_state_dict(torch.load(root + 'saved_models/%s/discriminator_%d.pth' % (opt.dataset_name, opt.epoch)))
else:
    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Configure dataloaders
transforms_A = [ transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
transforms_B = [ #transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
                # transforms.RandomHorizontalFlip(p=0.5),
                # transforms.RandomVerticalFlip(p=0.5),
                # transforms.RandomResizedCrop(256, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2),
                transforms.ToTensor() 
                ]

dataloader = DataLoader(ImageDataset("/media/arg_ws3/5E703E3A703E18EB/data/subt_all/", transforms_A=transforms_B, transforms_B=transforms_B),
                        batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)

val_dataloader = DataLoader(ImageDataset("/media/arg_ws3/5E703E3A703E18EB/data/subt_all/", transforms_A=transforms_B, transforms_B=transforms_B, mode='val'),
                            batch_size=10, shuffle=True, num_workers=1)

# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def sample_images(batches_done):
    """Saves a generated sample from the validation set"""
    imgs = next(iter(val_dataloader))
    real_D = Variable(imgs['B'].type(Tensor)) # depth
    real_B = Variable(imgs['A'].type(Tensor)) # mask
    real_C = Variable(imgs['C'].type(Tensor)) # rgb
    real_A = torch.cat((real_D, real_C), 1)
    # real_A = real_D
    fake_B = generator(real_A)
    img_sample = torch.cat((real_C[:,0,:,:].unsqueeze(1).data, fake_B.data, real_B.data), -2)
    save_image(img_sample, root + 'images/%s/%s.png' % (opt.dataset_name, batches_done), nrow=5, normalize=False)

# ----------
#  Training
# ----------

prev_time = time.time()

for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):

        # Model inputs
        real_D = Variable(batch['B'].type(Tensor)) # depth
        real_B = Variable(batch['A'].type(Tensor)) # mask
        real_C = Variable(batch['C'].type(Tensor)) # rgb
        real_A = torch.cat((real_D, real_C), 1)
        # real_A = real_D
        # print('===========')
        # print(real_D.shape, real_B.shape, real_C.shape, real_A.shape)

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((real_B.size(0), *patch))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_B.size(0), *patch))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # GAN loss
        fake_B = generator(real_A)
        pred_fake = discriminator(fake_B, real_A)
        loss_GAN = criterion_GAN(pred_fake, valid)
        # Pixel-wise loss
        loss_pixel = criterion_pixelwise(fake_B, real_B)

        # Total loss
        loss_G = loss_GAN + lambda_pixel * loss_pixel
        #loss_G = loss_GAN

        loss_G.backward()

        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Real loss
        pred_real = discriminator(real_B, real_A)
        loss_real = criterion_GAN(pred_real, valid)

        # Fake loss
        pred_fake = discriminator(fake_B.detach(), real_A)
        loss_fake = criterion_GAN(pred_fake, fake)

        # Total loss
        loss_D = 0.5 * (loss_real + loss_fake)

        loss_D.backward()
        optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write("\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f] ETA: %s" %
                                                        (epoch, opt.n_epochs,
                                                        i, len(dataloader),
                                                        loss_D.item(), loss_G.item(),
                                                        loss_pixel.item(), loss_GAN.item(),
                                                        time_left))

        # If at sample interval save image
        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done)


    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), root + 'saved_models/%s/generator_%d.pth' % (opt.dataset_name, epoch))
        torch.save(discriminator.state_dict(), root + 'saved_models/%s/discriminator_%d.pth' % (opt.dataset_name, epoch))
torch.save(generator.state_dict(), root + 'saved_models/%s/generator.pth' % (opt.dataset_name))
torch.save(discriminator.state_dict(), root + 'saved_models/%s/discriminator.pth' % (opt.dataset_name))