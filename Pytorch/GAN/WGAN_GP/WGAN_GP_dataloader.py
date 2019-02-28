import os
import glob
import torch.utils.data
import numpy as np
from PIL import Image
import LJY_utils

class Dataloader(torch.utils.data.Dataset):
    def __init__(self, path, transform, type='train'):
        super().__init__()
        self.type = type
        self.transform = transform

        assert os.path.exists(path)
        self.base_path = path

        #self.mean_image = self.get_mean_image()

        cur_file_paths = glob.glob(self.base_path + '/*.png')
        cur_file_paths.sort()

        self.file_paths, self.val_paths, self.test_paths = LJY_utils.tvt_divider(cur_file_paths,train_ratio=4,val_ratio=1,test_ratio=1)

    def pil_loader(self, path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

    def __len__(self):
        if self.type =='train':
            return len(self.file_paths)
        elif self.type =='val':
            return len(self.val_paths)
        elif self.type == 'test':
            return len(self.test_paths)

    def __getitem__(self, item):
        if self.type =='train':
            path = self.file_paths[item]
        elif self.type =='val':
            path = self.val_paths[item]
        elif self.type == 'test':
            path = self.test_paths[item]

        img = self.pil_loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img