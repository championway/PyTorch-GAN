import glob
import random
import os
import numpy as np

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root = "/media/arg_ws3/5E703E3A703E18EB/data/unity_obj/", transforms_A=None, transforms_B=None, mode='train'):
        self.transform_A = transforms.Compose(transforms_A)
        self.transform_B = transforms.Compose(transforms_B)
        self.root = root
        self.files = []
        for line in open(os.path.join(root, mode + '.txt')):
            self.files.append(line.strip())

    def __getitem__(self, index):
        idx = index % len(self.files)
        A_path = self.root + self.files[idx] + '_cnt.png'
        split_name = A_path.replace('image', '.').split('.')[1]
        B_path = self.root + self.files[idx] + '_depth.png'
        C_path = self.root + self.files[idx] + '_original.png'
        img_A = Image.open(A_path).resize((256, 256), Image.ANTIALIAS)
        img_B = Image.open(B_path).resize((256, 256), Image.ANTIALIAS)
        img_C = Image.open(C_path).resize((256, 256), Image.ANTIALIAS)
        img_A = img_A.convert('L')
        img_B = img_B.convert('L')

        img_A = np.array(img_A)
        img_B = np.array(img_B)/100.
        img_C = np.array(img_C)

        depth_max = float(img_B.max())
        depth_min = float(img_B.min())
        # print('=========================')
        # print(depth_min, depth_max)

        img_B = (img_B - depth_min)/(depth_max - depth_min)
        # print(img_B.min(), img_B.max())

        # print()
        # print(np.array(img_A).max(), np.array(img_A).min())
        # print(np.array(img_B).max(), np.array(img_B).min())
        #img_B = np.array(img_B)
        #img_B = Image.fromarray(img_B).convert('RGB')
        if np.random.random() < 0.5:
            # img_A = Image.fromarray(np.array(img_A)[::-1, :], 'L')
            # img_B = Image.fromarray(np.array(img_B)[::-1, :], 'L')
            # img_C = Image.fromarray(np.array(img_C)[::-1, :])
            img_A = np.array(img_A)[::-1, :]
            img_B = np.array(img_B)[::-1, :]
            img_C = np.array(img_C)[::-1, :]
            # print(np.array(img_A).shape)

        # img_A = torch.tensor(np.array(img_A)).unsqueeze(dim=0)
        img_A = self.transform_A(Image.fromarray(img_A, 'L'))
        img_B = torch.tensor(np.array(img_B)).unsqueeze(dim=0)
        img_C = torch.tensor(np.array(img_C)).unsqueeze(dim=0)
        # print(img_B.min(), img_B.max())
        # print(img_B)

        # img_A = self.transform_A(img_A)
        # img_B = self.transform_B(img_B)
        # img_C = self.transform_B(img_C)
        # print(img_A.shape, img_B.shape, img_C.shape)

        return {'A': img_A, 'B': img_B, 'C': img_C}

    '''def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        w, h = img.size
        img_A = img.crop((0, 0, w/2, h))
        img_B = img.crop((w/2, 0, w, h))

        if np.random.random() < 0.5:
            img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], 'RGB')
            img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], 'RGB')

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return {'A': img_A, 'B': img_B}'''

    def __len__(self):
        return len(self.files)
