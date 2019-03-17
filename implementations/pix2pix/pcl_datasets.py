import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root = "/media/arg_ws3/5E703E3A703E18EB/data/sparse2dense/", transforms_=None, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.root = root
        self.files = []
        for dirPath, dirNames, fileNames in os.walk(self.root + "depth/"):
            for f in fileNames:
                self.files.append(os.path.join(dirPath, f))
        print(len(self.files))
        #self.files = sorted(glob.glob(os.path.join(root, mode) + '/*.*'))
        #if mode == 'train':
        #    self.files.extend(sorted(glob.glob(os.path.join(root, 'test') + '/*.*')))

    def __getitem__(self, index):
        idx = index % len(self.files)
        A_path = self.files[idx]
        split_name = A_path.replace('depth', '/').split('/')[-1]
        B_path = self.root + "pcl/" + split_name
        img_A = Image.open(A_path)
        img_B = Image.open(B_path)
        img_A = np.array(img_A)/10000.
        img_B = np.array(img_B)/10000.
        img_A = Image.fromarray(img_A)
        img_B = Image.fromarray(img_B)

        #if np.random.random() < 0.5:
        #    img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], 'RGB')
        #    img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], 'RGB')

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return {'A': img_A, 'B': img_B}

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
