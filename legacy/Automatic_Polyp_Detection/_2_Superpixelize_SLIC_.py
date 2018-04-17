import os

import cv2
import numpy as np
from skimage.segmentation import slic

import LJY_utils
import LM_Filter
from legacy.Automatic_Polyp_Detection import superpixel, mask_converter


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

is_train = True

# for calculate LM feature...
LM_filter_bank = LM_Filter.makeLMfilters()

root_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/CVC-ClinicDB"
image_root_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/CVC-ClinicDB/Remove_Boundary"
superpixel_save_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/CVC-ClinicDB/superpixels/"
feature_save_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/CVC-ClinicDB/features"
LJY_utils.make_dir(feature_save_path)

image_path_list = LJY_utils.get_file_paths(image_root_path, "/*.", ['png', 'PNG'])


for cnt, image_path in enumerate(image_path_list):

    print("[%d/%d] Superpixelize_SLIC" % (cnt, len(image_path_list)))
    image = cv2.imread(image_path)
    image_number = int(LJY_utils.extract_filename_from_path(image_path))

    grayimage = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    boundary_mask = grayimage > 30

    superpixels = []
    for superpixel_level in range(1, 6):
        superpixel_index = 0
        #1~5 level segments...
        segments = slic(image, n_segments=50*superpixel_level, compactness=50, convert2lab=True)
        for i in range(np.min(segments), np.max(segments)+1):
            # if image belongs to a segment and is not a boundary, the mask of the corresponding index is true.
            superpixel_mask = (segments[:] == i)
            # superpixel && boundary.
            intergrated_mask = np.logical_and(boundary_mask, superpixel_mask)
            # [weight, height] => [weight, height, channel]. for calculate with 3channel image.
            intergrated_mask = intergrated_mask[:, :, np.newaxis]

            # check there is no masked image in this level.
            # check there is no masked image in this level.
            if intergrated_mask.any():
                superpixels.append(
                    superpixel.superpixel(superpixel_save_path, mask_converter.mask_to_list(intergrated_mask),
                                          superpixel_index, image, image_number, superpixel_level, LM_filter_bank))
                min_max_hist = min_max_refresh(superpixels[superpixel_index].color_hist_feature, min_max_hist)
                min_max_hog = min_max_refresh(superpixels[superpixel_index].HOG_feature, min_max_hog)
                min_max_lm = min_max_refresh(superpixels[superpixel_index].LM_feature, min_max_lm)
                min_max_lbp = min_max_refresh(superpixels[superpixel_index].LBP_feature, min_max_lbp)
                # superpixels[superpixel_index].save_superpixel()
                # print(superpixel_index)
                print(["Create %d superpixel"])
                superpixel_index = superpixel_index + 1
                # print(i)

        normal_hist = bigger_abs(min_max_hist)
        normal_hog = bigger_abs(min_max_hog)
        normal_lm = bigger_abs(min_max_lm)
        normal_lbp = bigger_abs(min_max_lbp)

        for sp in superpixels:
            if not sp.is_SAE_feature:
                sp.color_hist_feature = sp.color_hist_feature / normal_hist
                sp.HOG_feature = sp.color_hist_feature / normal_hog
                sp.LM_feature = sp.color_hist_feature / normal_lm
                sp.LBP_feature = sp.color_hist_feature / normal_lbp
                if is_train:
                    features = [sp.color_hist_feature, sp.HOG_feature, sp.LM_feature, sp.LBP_feature]
                    np.save(os.path.join(feature_save_path, "%08d_features" % (sp.index)), features)
