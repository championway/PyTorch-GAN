import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))

    def __getitem__(self, index):
        A_correct = False
        B_correct = False

        A_name = self.files_A[index % len(self.files_A)]
        img_A = Image.open(A_name)
        try:
            item_A = self.transform(img_A)
            A_correct = True
        except:
            print("Wrong transform A: ", A_name)

        if self.unaligned:
            B_name = self.files_B[random.randint(0, len(self.files_B) - 1)]
            img_B = Image.open(B_name)
            try:
                item_B = self.transform(img_B)
                B_correct = True
            except:
                print("Wrong transform B: ", B_name)
        else:
            B_name = self.files_B[index % len(self.files_B)]
            img_B = Image.open(B_name)
            try:
                item_B = self.transform(img_B)
                B_correct = True
            except:
                print("Wrong transform B: ", B_name)

        if not A_correct or not B_correct: # if data encounter error
            return []

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
