import glob
import os
import random
from PIL import Image
import numpy as np
import scipy.misc as sm
import cv2

random.seed(1)


def make_dir(path):
    # if there is no directory, make a directory.
    if not os.path.exists(path):
        os.makedirs(path)
        print(path)
    return


def basename_to_image_path_list(basename):
    image_2 = glob.glob(os.path.join(Image_root_path, basename, "image_02", "data", "*.png"))
    image_2.sort()
    image_3 = glob.glob(os.path.join(Image_root_path, basename, "image_03", "data", "*.png"))
    image_3.sort()
    return image_2, image_3


def basename_to_depth_path_list(basename):
    image_2 = glob.glob(os.path.join(Depth_root_path, basename, "proj_depth", "groundtruth", "image_02", "*.png"))
    image_2.sort()
    image_3 = glob.glob(os.path.join(Depth_root_path, basename, "proj_depth", "groundtruth", "image_03", "*.png"))
    image_3.sort()
    return image_2, image_3


Image_root_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/KITTI"
Depth_root_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/KITTI_Depth"

train_image_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/KITTI_train/img"
train_depth_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/KITTI_train/depth"

val_image_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/KITTI_val/val"
val_depth_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/KITTI_val/depth"

data_index = 0

image_folder_list = sorted(glob.glob(os.path.join(Image_root_path, "*")))
depth_folder_list = sorted(glob.glob(os.path.join(Depth_root_path, "*")))

image_folder_list = list(map(os.path.basename, image_folder_list))
depth_folder_list = list(map(os.path.basename, depth_folder_list))

train_set = list(set.intersection(set(image_folder_list), set(depth_folder_list)))
val_image_set = list(set.difference(set(image_folder_list), set(train_set)))
val_depth_set = list(set.difference(set(depth_folder_list), set(train_set)))
data_max = 21931
data_min = 0

window_size = 9
minDisparity = 1
stereo = cv2.StereoSGBM_create(
    blockSize=10,
    numDisparities=32,
    preFilterCap=10,
    minDisparity=minDisparity,
    P1=4 * 3 * window_size ** 2,
    P2=32 * 3 * window_size ** 2
)


disparity_max = 1008
disparity_min = -16

for j, basename in enumerate(train_set):
    if j == 0:
        left_img, right_img = basename_to_image_path_list(basename)
        print(basename)

        for i in range(len(left_img)):
            left_img_img = Image.open(left_img[i]).resize((212, 64))
            right_img_img = Image.open(right_img[i]).resize((212, 64))
            # load img and resize it.

            left_img_img.save(os.path.join(val_image_path, "%s_left_%06d.png" % (basename, i)))
            right_img_img.save(os.path.join(val_image_path, "%s_right_%06d.png" % (basename, i)))

    else:
        left_img, right_img = basename_to_image_path_list(basename)
        left_dep, right_dep = basename_to_depth_path_list(basename)

        if len(left_img) > len(left_dep):
            maxlen = left_dep
        else:
            maxlen = left_img
        print(basename)
        img_start_idx = int(os.path.basename(maxlen[0]).split('.')[0])
        for i in range(len(maxlen)):
            left_img_img = Image.open(left_img[i + img_start_idx]).resize((212, 64))
            left_dep_img = Image.open(left_dep[i]).resize((212, 64))

            right_img_img = Image.open(right_img[i + img_start_idx]).resize((212, 64))
            right_dep_img = Image.open(right_dep[i]).resize((212, 64))
            # load img and resize it.

            right_dep_img = np.asarray(right_dep_img)
            right_dep_img.setflags(write=1)  # since np.asarray(cv image) is read only.
            left_dep_img = np.asarray(left_dep_img)
            left_dep_img.setflags(write=1)

            disparity = stereo.compute(np.asarray(left_img_img), np.asarray(right_img_img))
            disparity_visual = cv2.normalize(disparity, disparity, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

            #disparity = (disparity + abs(disparity_min)) / (disparity_max + abs(disparity_min))
            # normalizing disparity map.


            disparity_mask = disparity != 0

            right_dep_img = right_dep_img / data_max
            left_dep_img = left_dep_img / data_max

            disparity_difference_list = []

            # if element is not 0, x => 1/x(inverse depth)
            right_mask = right_dep_img != 0
            left_mask = left_dep_img != 0
            for x in range(right_mask.shape[0]):
                for y in range(right_mask.shape[1]):
                    if right_mask[x][y] is True:
                        print(right_dep_img[x][y])
                        right_dep_img[x][y] = 1 / right_dep_img[x][y]

                    if left_mask[x][y] is True:
                        print(left_dep_img[x][y])
                        left_dep_img[x][y] = 1 / left_dep_img[x][y]

            right_dep_img = sm.toimage(right_dep_img)
            left_dep_img = sm.toimage(left_dep_img)
            disparity = sm.toimage(disparity)

            left_img_img.save(os.path.join(train_image_path, 'left', "%s_left_%06d.png" % (basename, i)))
            right_img_img.save(os.path.join(train_image_path, 'right', "%s_right_%06d.png" % (basename, i)))

            # left_dep_img.save(os.path.join(train_depth_path, "%s_left_%06d.png" % (basename, i)))
            right_dep_img.save(os.path.join(train_depth_path, 'lidar', "%s_lidar_%06d.png" % (basename, i)))
            disparity.save(os.path.join(train_depth_path, 'disparity', "%s_disparity_%06d.png" % (basename, i)))

            print("[%d/%d][%d/%d]" % (j, len(train_set), i, len(left_dep)))


for j, basename in enumerate(val_depth_set):
    if j >= 10:
        break
    left_dep, right_dep = basename_to_depth_path_list(basename)
    print(basename)

    for i in range(len(left_dep)):
        left_dep_img = Image.open(left_dep[i]).resize((212, 64))
        right_dep_img = Image.open(right_dep[i]).resize((212, 64))
        # load img and resize it.

        right_dep_img = np.asarray(right_dep_img)
        right_dep_img.setflags(write=1)  # since np.asarray(cv image) is read only.
        left_dep_img = np.asarray(left_dep_img)
        left_dep_img.setflags(write=1)


        # if element is not 0, x => 1/x(inverse depth)
        right_mask = right_dep_img != 0
        left_mask = left_dep_img != 0
        for x in range(right_mask.shape[0]):
            for y in range(right_mask.shape[1]):
                if right_mask[x][y] is True:
                    print(right_dep_img[x][y])
                    right_dep_img[x][y] = 1 / right_dep_img[x][y]

                if left_mask[x][y] is True:
                    print(left_dep_img[x][y])
                    left_dep_img[x][y] = 1 / left_dep_img[x][y]

        right_dep_img = sm.toimage(right_dep_img)
        left_dep_img = sm.toimage(left_dep_img)

        right_dep_img.save(os.path.join(val_depth_path, "%s_lidar_%06d.png" % (basename, i)))
        print("[%d/%d][%d/%d]" % (j, len(val_depth_set), i, len(left_dep)))

for j, basename in enumerate(val_image_set):
    left_img, right_img = basename_to_image_path_list(basename)
    print(basename)

    for i in range(len(left_img)):
        left_img_img = Image.open(left_img[i]).resize((212, 64))
        right_img_img = Image.open(right_img[i]).resize((212, 64))
        # load img and resize it.

        left_img_img.save(os.path.join(val_image_path, "%s_left_%06d.png" % (basename, i)))
        right_img_img.save(os.path.join(val_image_path, "%s_right_%06d.png" % (basename, i)))

        print("[%d/%d][%d/%d]" % (j, len(train_set), i, len(left_img_img)))

print("eof")