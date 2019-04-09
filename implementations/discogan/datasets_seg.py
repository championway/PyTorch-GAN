import glob
import os
import torch

import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root = "/media/arg_ws3/5E703E3A703E18EB/data/unity_obj/", transforms_=None, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.files = []
        self.root = root
        for line in open(os.path.join(root, mode + '.txt')):
            self.files.append(line.strip())

    def __getitem__(self, index):
        idx = index % len(self.files)
        A_path = self.root + self.files[idx] + '_seg.png'
        split_name = A_path.replace('image', '.').split('.')[1]
        B_path = self.root + self.files[idx] + '_cnt.png'
        img_A = Image.open(A_path)
        img_B = Image.open(B_path)
        #img_A = Image.open(A_path).convert('L')
        #img_B = Image.open(B_path).convert('L')
        #img_B = np.array(img_B)
        #img_B = Image.fromarray(img_B).convert('RGB')

        #if np.random.random() < 0.5:
        #    img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], 'RGB')
        #    img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], 'RGB')

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return {'A': img_A, 'B': img_B}

    def __len__(self):
        return len(self.files)
