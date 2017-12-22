# load superpixel
# extract feature
# save feature (for use as traing samples.)
from glob import glob
import LJY_utils
import pickle
import numpy as np
import os

superpixel_root_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/CVC-ClinicDB/superpixels"
superpixel_image_list = glob(superpixel_root_path+"/*/")
superpixel_image_list.sort()
feature_save_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/CVC-ClinicDB/features"
LJY_utils.make_dir(feature_save_path)



def min_max_refresh(data, minmax):
    if np.max(data)>minmax[1]:
        minmax[1] = np.max(data)
    if np.min(data) < minmax[0]:
        minmax[0] = np.min(data)
    return minmax
def bigger_abs(minmax):
    if abs(minmax[0]) > abs(minmax[1]):
        return minmax[0]
    else:
        return minmax[1]

min_max_hist = [0,0]
min_max_hog = [0,0]
min_max_lm = [0,0]
min_max_lbp = [0,0]
for i, superpixel_image in enumerate(superpixel_image_list):
    superpixel_list = LJY_utils.get_file_paths(superpixel_image, "/*.", ['txt', 'TXT'])
    for j, superpixel in enumerate(superpixel_list):
        with open(superpixel, "rb") as fp:
            data = pickle.load(fp)
        # color_hist_feature, HOG_feature, LM_feature, LBP_feature
        # 610 255 96 255
        min_max_hist = min_max_refresh(data[2], min_max_hist)
        min_max_hog = min_max_refresh(data[3], min_max_hog)
        min_max_lm = min_max_refresh(data[4], min_max_lm)
        min_max_lbp = min_max_refresh(data[5], min_max_lbp)
        print("%04d_%04d_features minmax" % (i, j))

normal_hist = bigger_abs(min_max_hist)
normal_hog = bigger_abs(min_max_hog)
normal_lm = bigger_abs(min_max_lm)
normal_lbp = bigger_abs(min_max_lbp)

for i, superpixel_image in enumerate(superpixel_image_list):
    if i == 68:
        break
    superpixel_list = LJY_utils.get_file_paths(superpixel_image, "/*.", ['txt', 'TXT'])
    for j, superpixel in enumerate(superpixel_list):
        with open(superpixel, "rb") as fp:
            data = pickle.load(fp)
        features = [data[2]/normal_hist, data[3]/normal_hog, data[4]/normal_lm, data[5]/normal_lbp]
        np.save(os.path.join(feature_save_path, "%04d_%04d_features" % (i, j)), features)
        print("%04d_%04d_features" % (i, j))

