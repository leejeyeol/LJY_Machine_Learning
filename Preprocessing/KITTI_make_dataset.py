import glob
import os
import random
from PIL import Image
import numpy as np
import scipy.misc as sm

random.seed(1)

def make_dir(path):
    # if there is no directory, make a directory.
    if not os.path.exists(path):
        os.makedirs(path)
        print(path)
    return

def basename_to_image_path_list(basename):
    image_2 = glob.glob(os.path.join(Image_root_path,basename,"image_02","data","*.png"))
    image_2.sort()
    image_3 = glob.glob(os.path.join(Image_root_path,basename,"image_03","data","*.png"))
    image_3.sort()
    return image_2, image_3

def basename_to_depth_path_list(basename):
    image_2= glob.glob(os.path.join(Depth_root_path,basename,"proj_depth","groundtruth","image_02","*.png"))
    image_2.sort()
    image_3= glob.glob(os.path.join(Depth_root_path,basename,"proj_depth","groundtruth","image_03","*.png"))
    image_3.sort()
    return image_2, image_3



Image_root_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/KITTI"
Depth_root_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/KITTI_Depth"

train_image_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/KITTI_train/img"
train_depth_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/KITTI_train/depth"

val_image_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/KITTI_val/val"
val_depth_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/KITTI_val/depth"



data_index = 0

image_folder_list = glob.glob(os.path.join(Image_root_path, "*"))
depth_folder_list = glob.glob(os.path.join(Depth_root_path, "*"))
image_folder_list.sort()
depth_folder_list.sort()
image_folder_list = list(map(os.path.basename,image_folder_list))
depth_folder_list = list(map(os.path.basename,depth_folder_list))

train_set = list(set.intersection(set(image_folder_list), set(depth_folder_list)))


for j, basename in enumerate(train_set):
    left_img, right_img = basename_to_image_path_list(basename)
    left_dep, right_dep = basename_to_depth_path_list(basename)

    if len(left_img) > len(left_dep):
        maxlen = left_dep
    else:
        maxlen = left_img
    print(basename)

    img_start_idx=int(os.path.basename(maxlen[0]).split('.')[0])
    for i in range(len(maxlen)):
        left_img_img = Image.open(left_img[i+img_start_idx])
        left_dep_img = Image.open(left_dep[i]).resize((608, 96))

        right_img_img = Image.open(right_img[i+img_start_idx])
        right_dep_img = Image.open(right_dep[i]).resize((608, 96))

        left_img_img.save(os.path.join(train_image_path, "%s_left_%06d.png" % (basename, i)))
        right_img_img.save(os.path.join(train_image_path, "%s_right_%06d.png" % (basename, i)))

        left_dep_img.save(os.path.join(train_depth_path, "%s_left_%06d.png" % (basename, i)))
        right_dep_img.save(os.path.join(train_depth_path, "%s_right_%06d.png" % (basename, i)))
        print("[%d/%d][%d/%d]" % (j, len(train_set), i, len(left_dep)))

print("debug")