# load superpixels corresponding to each images
# calculate saliency per superpixel per each level
# save object based saliency value to each superpixels

# load origianl image and import Harris corner detector
# find center based saliency value to each superpixels

# add two saliency
# save saliency value in superpixel class and save it.

from glob import glob
import LJY_utils
import pickle
import numpy as np
import cv2
from scipy.spatial import distance
import os

original_image_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/CVC-ClinicDB/Remove_Boundary"

superpixel_root_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/CVC-ClinicDB/superpixels"
superpixel_image_list = glob(superpixel_root_path+"/*/")


for i, superpixel_image in enumerate(superpixel_image_list):
    # contrast based saliency map

    superpixel_list = LJY_utils.get_file_paths(superpixel_image, "/*.", ['txt', 'TXT'])
    superpixel_per_level_list = [[], [], [], [], []]
    for superpixel_level in range(0, 5):
        for j, superpixel in enumerate(superpixel_list):
            with open(superpixel, "rb") as fp:
                data = pickle.load(fp)
            superpixel_per_level_list[int(os.path.basename(superpixel).split('_')[0])].append([superpixel, data[1], data[2], data[3], data[4], data[5]])
            # center, colorhist, hog, lm, lbp
    for superpixel_level in range(0, 5):
        for anchor, superpixel_per_level in enumerate(superpixel_per_level_list[superpixel_level]):
            # superpixel_per_level_list = ???? ???? ???? ??? ?? ?????
            # ???? ???? ??? ??
            saliency_value = 0
            for k in range(0, len(superpixel_per_level_list[superpixel_level])):
                if k != anchor:
                    saliency_value = saliency_value + \
                                     (distance.euclidean(superpixel_per_level_list[superpixel_level][k][2]),\
                                     (superpixel_per_level_list[superpixel_level][anchor][2])+\
                                     distance.euclidean(superpixel_per_level_list[superpixel_level][k][3]),\
                                     (superpixel_per_level_list[superpixel_level][anchor][3])+\
                                     distance.euclidean(superpixel_per_level_list[superpixel_level][k][4]),\
                                     (superpixel_per_level_list[superpixel_level][anchor][4])+\
                                     distance.euclidean(superpixel_per_level_list[superpixel_level][k][5]),\
                                     (superpixel_per_level_list[superpixel_level][anchor][5]))/\
                                     1 + (distance.euclidean(superpixel_per_level_list[superpixel_level][k][1]),(superpixel_per_level_list[superpixel_level][anchor][1]))
            superpixel_per_level_list[superpixel_level][anchor].append(saliency_value)

    for superpixel_level in range(0, 5):
        saliency_map = None
        for superpixel_per_level in superpixel_per_level_list[superpixel_level]:
            superpixel = superpixel_per_level[0]
            with open(superpixel, "rb") as fp:
                data = pickle.load(fp)
            mask = data[0]
            saliency_value = superpixel_per_level[6]
            data[6] = saliency_value
            if saliency_map is None:
                saliency_map = np.zeros(mask.shape)
            saliency_map = saliency_map + mask * saliency_value
        np.save(os.path.join(superpixel_image, "%d_level_contrast_based_saliency_map" % superpixel_level))
    print("[%d/%d] contrast based saliency map" % (i, len(superpixel_image_list)))


    # object center based saliency map

    #todo anisotropic gaussian distribution ? ??
    #todo color Harris point detetctor? ??? ????
    #todo mean? variance? variance? ?? ??? ??? ????? ??? ? ????
    original_image = cv2.imread(os.path.join(original_image_path, "/%06d.png" % i))
    gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

