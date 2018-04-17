import gc
import os
from glob import glob

import numpy as np

import LJY_utils

result_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/CVC-ClinicDB/Results"
superpixel_root_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/superpixels"
level_of_superpixels = 5
weight_for_the_combination = 0.4

LJY_utils.make_dir(result_path)

def MKB(weights, F, x):
    weights = weights/np.sum(weights)
    result = 0
    for i, fun in enumerate(F):
        # fun[0]: weight
        # fun[1]: SVM
        result = result + weights[i] * fun.predict(np.asarray(x).reshape(1,-1))
    return result

weights = np.load(os.path.join(os.path.dirname(result_path),"MKB_weights.npy"))
clfs = np.load(os.path.join(os.path.dirname(result_path),"MKB_clfs.npy"))

# ---------------------------------------------------------------------------------------------
# final


image_list = glob(superpixel_root_path + "/*/")
image_list.sort()

for i, superpixel_image in enumerate(image_list):
    print("[%d/%d]make map"%(i,len(image_list)))
    gc.collect()
    superpixel_per_level_list = [[], [], [], [], []]
    superpixel_list = LJY_utils.get_file_paths(superpixel_image, "/*.", ['txt', 'TXT'])
    for superpixel_level in range(0, level_of_superpixels):
        for superpixel in superpixel_list:
            superpixel_per_level_list[int(os.path.basename(superpixel).split('_')[0])-1].append(superpixel)

    WBU_saliency_map = None
    STD_saliency_map = None
    total_saliency_map = None

    for superpixel_level in range(0, level_of_superpixels):
        for superpixel_per_level in superpixel_per_level_list[superpixel_level]:
            superpixel = superpixel_per_level
            _superpixel = superpixel.superpixel(superpixel)

            result = MKB(weights, clfs, sum([_superpixel.color_hist_feature,_superpixel.HOG_feature,_superpixel.LBP_feature,_superpixel.LM_feature],[]))

            _superpixel.set_STD_saliency_value(result)

            total_saliency_value = weight_for_the_combination * _superpixel.saliency_value_WBU + (1 - weight_for_the_combination) * _superpixel.saliency_value_STD

            mask = _superpixel.mask
            WBU_saliency_value = _superpixel.saliency_value_WBU
            STD_saliency_value = _superpixel.saliency_value_STD

            if STD_saliency_map is None:
                STD_saliency_map = np.zeros(mask.shape)
            if WBU_saliency_map is None:
                WBU_saliency_map = np.zeros(mask.shape)
            if total_saliency_map is None:
                total_saliency_map = np.zeros(mask.shape)

            STD_saliency_map = STD_saliency_map + mask * STD_saliency_value
            WBU_saliency_map = WBU_saliency_map + mask * WBU_saliency_value
            total_saliency_map = total_saliency_map + mask * total_saliency_value

            _superpixel.save_superpixel() # todo 제거하고 테스트

    np.save(os.path.join(result_path, "%d_STD_saliency_map" % (i + 1)),
            STD_saliency_map/5)
    np.save(os.path.join(result_path, "%d_WBU_saliency_map" % (i + 1)),
            WBU_saliency_map/5)
    np.save(os.path.join(result_path, "%d_total_saliency_map" % (i + 1)),
            total_saliency_map/5)