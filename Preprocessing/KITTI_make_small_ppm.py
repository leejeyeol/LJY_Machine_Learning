import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch
import glob as glob
import os

inputpath = '/media/leejeyeol/74B8D3C8B8D38750/Data/KITTI_train/img'
outputpath = '/media/leejeyeol/74B8D3C8B8D38750/Data/KITTI_train/img_small_ppm'

original_images = glob.glob(os.path.join(inputpath, "*.*"))
original_images.sort()


def pil_loader(path, convert='RGB'):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert(convert)


for i, path in enumerate(original_images):
    img = pil_loader(path)
    img = img.resize((608, 96))
    img.save(os.path.join(outputpath, os.path.basename(path).split('.')[0]+'.ppm'))
    print('[%d/%d]'%(i,len(original_images)))



print(1)