# load superpixels corresponding to each images
# calculate saliency per superpixel per each level
# save object based saliency value to each superpixels

# load origianl image and import Harris corner detector
# find center based saliency value to each superpixels

# add two saliency
# save saliency value in superpixel class and save it.

# 56 hours
from glob import glob
import LJY_utils
import pickle
import numpy as np
import cv2
from scipy.spatial import distance
import os
from Automatic_Polyp_Detection import superpixel as SUPERPIXEL

import LJY_visualize_tools

original_image_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/CVC-ClinicDB/Remove_Boundary"
superpixel_root_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/CVC-ClinicDB/superpixels"
superpixel_image_list = glob(superpixel_root_path+"/*/")
superpixel_image_list.sort()
level_of_saliency = 5


for i, superpixel_image in enumerate(superpixel_image_list):
    # contrast based saliency map
    original_image = cv2.imread(os.path.join(original_image_path, "%06d.png" % (i+1)),0)
    gray = np.float32(original_image)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    point_list = np.column_stack(np.where(dst > 0.01 * dst.max()))
    center = np.int8(np.mean(point_list, axis=0))

    superpixel_list = LJY_utils.get_file_paths(superpixel_image, "/*.", ['txt', 'TXT'])
    superpixel_per_level_list = [[], [], [], [], []]
    CB_total_saliency_per_level_list = [0, 0, 0, 0, 0]
    OB_total_saliency_per_level_list = [0, 0, 0, 0, 0]
    mean_saliency_per_level_list = [0, 0, 0, 0, 0]


    for superpixel_level in range(0, level_of_saliency):
        for j, superpixel in enumerate(superpixel_list):
            with open(superpixel, "rb") as fp:
                data = pickle.load(fp)
            superpixel_per_level_list[int(os.path.basename(superpixel).split('_')[0])-1].append([superpixel, data[1], data[2], data[3], data[4], data[5]])
            # center, colorhist, hog, lm, lbp
    for superpixel_level in range(0, level_of_saliency):
        for anchor, superpixel_per_level in enumerate(superpixel_per_level_list[superpixel_level]):
            CB_saliency_value = 0
            for k in range(0, len(superpixel_per_level_list[superpixel_level])):
                OB_saliency_value = np.exp((-1 * (np.linalg.norm(superpixel_per_level_list[superpixel_level][k][1][0] - center[0]) / 2)) + (-1 * (np.linalg.norm(superpixel_per_level_list[superpixel_level][k][1][1] - center[1]) / 2)))
                OB_total_saliency_per_level_list[superpixel_level] = OB_total_saliency_per_level_list[superpixel_level] + OB_saliency_value
                if k != anchor:
                    CB_saliency_value = CB_saliency_value + \
                                     (distance.euclidean(superpixel_per_level_list[superpixel_level][k][2],\
                                     superpixel_per_level_list[superpixel_level][anchor][2])+\
                                     distance.euclidean(superpixel_per_level_list[superpixel_level][k][3],\
                                     superpixel_per_level_list[superpixel_level][anchor][3])+\
                                     distance.euclidean(superpixel_per_level_list[superpixel_level][k][4],\
                                     superpixel_per_level_list[superpixel_level][anchor][4])+\
                                     distance.euclidean(superpixel_per_level_list[superpixel_level][k][5],\
                                     superpixel_per_level_list[superpixel_level][anchor][5]))/\
                                     1 + (distance.euclidean(superpixel_per_level_list[superpixel_level][k][1], superpixel_per_level_list[superpixel_level][anchor][1]))
            superpixel_per_level_list[superpixel_level][anchor].append(CB_saliency_value)
            superpixel_per_level_list[superpixel_level][anchor].append(OB_saliency_value)

            CB_total_saliency_per_level_list[superpixel_level] = CB_total_saliency_per_level_list[superpixel_level] + CB_saliency_value
            print("[%d/%d][%d/%d] calculate saliency" % (i, len(superpixel_image_list), anchor, len(superpixel_per_level_list[superpixel_level])))
            #

    for superpixel_level in range(0, level_of_saliency):
        mean_saliency_per_level_list[superpixel_level] = 0
        CB_saliency_map = None
        OB_saliency_map = None
        WBU_saliency_map = None

        for superpixel_per_level in superpixel_per_level_list[superpixel_level]:
            superpixel = superpixel_per_level[0]

            _superpixel = SUPERPIXEL.superpixel(superpixel)

            mask = _superpixel.mask
            CB_saliency_value = superpixel_per_level[6] / CB_total_saliency_per_level_list[superpixel_level]
            OB_saliency_value = superpixel_per_level[7] / OB_total_saliency_per_level_list[superpixel_level]

            WBU_saliency_value = CB_saliency_value * OB_saliency_value

            mean_saliency_per_level_list[superpixel_level] = mean_saliency_per_level_list[superpixel_level] + WBU_saliency_value
            if CB_saliency_map is None:
                CB_saliency_map = np.zeros(mask.shape)
            CB_saliency_map = CB_saliency_map + mask * CB_saliency_value
            if OB_saliency_map is None:
                OB_saliency_map = np.zeros(mask.shape)
            OB_saliency_map = OB_saliency_map + mask * OB_saliency_value
            if WBU_saliency_map is None:
                WBU_saliency_map = np.zeros(mask.shape)
            WBU_saliency_map = WBU_saliency_map + mask * WBU_saliency_value

            _superpixel.set_WBU_saliency_value(WBU_saliency_value)
            _superpixel.save_superpixel()

        np.save(os.path.join(superpixel_image, "%d_level_contrast_based_saliency_map" % (superpixel_level+1)),
                CB_saliency_map)
        np.save(os.path.join(superpixel_image, "%d_level_objectcenter_based_saliency_map" % (superpixel_level+1)),
                OB_saliency_map)
        np.save(os.path.join(superpixel_image, "%d_level_WBU_saliency_map" % (superpixel_level+1)),
                WBU_saliency_map)

        mean_saliency_per_level_list[superpixel_level] = mean_saliency_per_level_list[superpixel_level] / len(superpixel_per_level_list[superpixel_level])
        np.save(os.path.join(superpixel_image, "%d_level_WBU_saliency_mean" % (superpixel_level+1)), mean_saliency_per_level_list[superpixel_level])


        # visualize
       # WBU_saliency_map = WBU_saliency_map-WBU_saliency_map.min()
       # WBU_saliency_map = WBU_saliency_map/WBU_saliency_map.max()
       # LJY_visualize_tools.Test_Image(WBU_saliency_map)

    print("[%d/%d] WBU saliency map" % (i, len(superpixel_image_list)))