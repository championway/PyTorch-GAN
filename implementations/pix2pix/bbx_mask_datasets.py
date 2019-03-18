import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import sys
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

class ImageDataset(Dataset):
    def __init__(self, root = "/media/arg_ws3/5E703E3A703E18EB/data/subt_real_ssd/", transforms_A=None, transforms_B=None, mode='train'):
        self.transform_A = transforms.Compose(transforms_A)
        self.transform_B = transforms.Compose(transforms_B)
        self.ANNOTATION_ROOT = '/media/arg_ws3/5E703E3A703E18EB/data/subt_real_ssd/Annotations'
        self.DEPTH_ROOT = '/media/arg_ws3/5E703E3A703E18EB/data/subt_real_ssd/JPEGImages'
        self.MASK_ROOT = '/media/arg_ws3/5E703E3A703E18EB/data/subt_real_ssd/MASKImages'
        self.root = root
        self.files = []
        for line in open('/media/arg_ws3/5E703E3A703E18EB/data/subt_real_ssd/ImageSets/Main/train.txt'):
            self.files.append((line.strip().split(' ')[0]))
        print(len(self.files))
        #self.files = sorted(glob.glob(os.path.join(root, mode) + '/*.*'))
        #if mode == 'train':
        #    self.files.extend(sorted(glob.glob(os.path.join(root, 'test') + '/*.*')))
    def get_crop_image(self, file, img_path):
        target = ET.parse(self.ANNOTATION_ROOT + '/' + file + '.xml').getroot()
        bndbox = []
        for obj in target.iter('object'):
            name = obj.find('name').text.lower().strip()
            if name != 'bb_extinguisher':
                continue
            polygons = obj.find('polygon')
            x = []
            y = []
            for polygon in polygons.iter('pt'):
                x.append(int(polygon.find('x').text))
                y.append(int(polygon.find('y').text))
            bndbox.append(min(x))
            bndbox.append(min(y))
            bndbox.append(max(x))
            bndbox.append(max(y))
        img = Image.open(img_path + '/' + file + '.png')
        crop_box = (bndbox[0], bndbox[1], bndbox[2], bndbox[3])
        crop_img = img.crop(crop_box)
        return crop_img

    def __getitem__(self, index):
        idx = index % len(self.files)
        '''A_path = self.files[idx]
        split_name = A_path.replace('mask', '*').split('*')[-1]
        B_path = self.root + "depth/" + split_name
        img_A = Image.open(A_path)
        img_B = Image.open(B_path)'''
        img_A = self.get_crop_image(file = self.files[idx], img_path = self.MASK_ROOT)
        img_B = self.get_crop_image(file = self.files[idx], img_path = self.DEPTH_ROOT)
        img_A = np.array(img_A)
        img_B = np.array(img_B)/1000.

        img_A = Image.fromarray(img_A)
        img_B = Image.fromarray(img_B)

        if np.random.random() < 0.5:
            img_A = Image.fromarray(np.array(img_A)[:, ::-1])
            img_B = Image.fromarray(np.array(img_B)[:, ::-1])

        img_A = self.transform_A(img_A)
        img_B = self.transform_B(img_B)

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
