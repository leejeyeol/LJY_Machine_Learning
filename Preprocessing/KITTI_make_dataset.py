
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
    return image_2,image_3

def basename_to_depth_path_list(basename):
    image_2= glob.glob(os.path.join(Depth_root_path,basename,"proj_depth","groundtruth","image_02","*.png"))
    image_2.sort()
    image_3= glob.glob(os.path.join(Depth_root_path,basename,"proj_depth","groundtruth","image_03","*.png"))
    image_3.sort()
    return image_2,image_3



Image_root_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/KITTI"
Depth_root_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/KITTI_Depth"
train_save_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/KITTI_train/train"
val_save_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/KITTI_train/val"
test_image_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/KITTI_test/depth"
test_depth_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/KITTI_test/img"


data_index = 0

image_folder_list = glob.glob(os.path.join(Image_root_path, "*"))
depth_folder_list = glob.glob(os.path.join(Depth_root_path, "*"))
image_folder_list.sort()
depth_folder_list.sort()
image_folder_list = list(map(os.path.basename,image_folder_list))
depth_folder_list = list(map(os.path.basename,depth_folder_list))

train_val_set = list(set.intersection(set(image_folder_list), set(depth_folder_list)))
val_set = random.sample(train_val_set, 2)
train_set = list(set.difference(set(train_val_set), set(val_set)))
test_image_set = list(set.difference(set(image_folder_list), set(train_val_set)))
test_image_set = random.sample(test_image_set, 2)
test_depth_set = list(set.difference(set(depth_folder_list), set(train_val_set)))

min_frame = np.asarray([[[0,0,0]]]).astype(float)
max_frame = np.asarray([[[255,255,255]]]).astype(float)
#1240 370 = > 60 18
for basename in train_set:
    left_img, right_img = basename_to_image_path_list(basename)
    left_dep, right_dep = basename_to_depth_path_list(basename)

    if len(left_img) > len(left_dep):
        maxlen = left_dep
    else:
        maxlen = left_img
    print(basename)
    img_start_idx=int(os.path.basename(maxlen[0]).split('.')[0])
    for i in range(len(maxlen)):
        left_img_img = sm.imresize(np.array(Image.open(left_img[i+img_start_idx]), dtype='float'),(18,60,3))
        left_dep_img = sm.imresize(np.array(Image.open(left_dep[i]), dtype='float'),(18,60,1))
        left_np_block = np.array([left_img_img[:, :, 0], left_img_img[:, :, 1], left_img_img[:, :, 2], left_dep_img])

        right_img_img = sm.imresize(np.array(Image.open(right_img[i+img_start_idx]), dtype='float'),(18,60,3))
        right_dep_img = sm.imresize(np.array(Image.open(right_dep[i]), dtype='float'),(18,60,1))
        right_np_block = np.array([right_img_img[:, :, 0], right_img_img[:, :, 1], right_img_img[:, :, 2], right_dep_img])

        np.save(os.path.join(train_save_path,"%s_left_%06d.npy"%(basename,i)),left_np_block)
        np.save(os.path.join(train_save_path,"%s_right_%06d.npy"%(basename,i)),right_np_block)
        print("%d/%d"%(i,len(left_dep)))

for basename in val_set:
    left_img, right_img = basename_to_image_path_list(basename)
    left_dep, right_dep = basename_to_depth_path_list(basename)
    if len(left_img) > len(left_dep):
        maxlen = left_dep
    else:
        maxlen = left_img
    print(basename)
    img_start_idx=int(os.path.basename(maxlen[0]).split('.')[0])
    for i in range(5, len(left_dep)):
        left_img_img = sm.imresize(np.array(Image.open(left_img[i+img_start_idx]), dtype='float'), (18, 60, 3))
        left_dep_img = sm.imresize(np.array(Image.open(left_dep[i]), dtype='float'), (18, 60, 1))
        left_np_block = np.array(
            [left_img_img[:, :, 0], left_img_img[:, :, 1], left_img_img[:, :, 2], left_dep_img])

        right_img_img = sm.imresize(np.array(Image.open(right_img[i+img_start_idx]), dtype='float'), (18, 60, 3))
        right_dep_img = sm.imresize(np.array(Image.open(right_dep[i]), dtype='float'), (18, 60, 1))
        right_np_block = np.array(
            [right_img_img[:, :, 0], right_img_img[:, :, 1], right_img_img[:, :, 2], right_dep_img])

        np.save(os.path.join(val_save_path, "%s_left_%06d.npy" % (basename, i)), left_np_block)
        np.save(os.path.join(val_save_path, "%s_right_%06d.npy" % (basename, i)), right_np_block)


for basename in test_depth_set:
    left_dep, right_dep = basename_to_depth_path_list(basename)

    for i in range(5, len(left_dep)):
        left_dep_img = sm.imresize(np.array(Image.open(left_dep[i]), dtype='float'), (18, 60, 1))
        left_np_block = np.array(
            [left_dep_img])

        right_dep_img = sm.imresize(np.array(Image.open(right_dep[i]), dtype='float'), (18, 60, 1))
        right_np_block = np.array(
            [right_dep_img])

        np.save(os.path.join(test_depth_path, "%s_left_%06d.npy" % (basename, i)), left_np_block)
        np.save(os.path.join(test_depth_path, "%s_right_%06d.npy" % (basename, i)), right_np_block)

for basename in test_image_set:
    left_img, right_img = basename_to_image_path_list(basename)

    for i in range(0, len(left_img)):
        left_img_img = sm.imresize(np.array(Image.open(left_img[i]), dtype='float'), (18, 60, 3))
        left_np_block = np.array(
            [left_img_img[:, :, 0], left_img_img[:, :, 1], left_img_img[:, :, 2]])

        right_img_img = sm.imresize(np.array(Image.open(right_img[i]), dtype='float'), (18, 60, 3))
        right_np_block = np.array(
            [right_img_img[:, :, 0], right_img_img[:, :, 1], right_img_img[:, :, 2]])

        np.save(os.path.join(val_save_path, "%s_left_%06d.npy" % (basename, i)), left_np_block)
        np.save(os.path.join(val_save_path, "%s_right_%06d.npy" % (basename, i)), right_np_block)
        print("dd")

print("debug")