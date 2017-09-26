import numpy as np
import cv2
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries


import LJY_visualize_tools
import LJY_utils


superpixel_index = 0
image_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/CVC-ClinicDB/Reflection Removed/000001.png"
image = cv2.imread(image_path)
image_number = int(LJY_utils.extract_filename_from_path(image_path))
# find LM responses at this image
# find LBP responses at this image
for superpixel_level in range(1, 6):
    #1~5 level segments...
    segments = slic(image, n_segments=50*superpixel_level, compactness=50, convert2lab=True)
    for i in range(np.min(segments), np.max(segments)):
        superpixel_mask = segments[:] == i
        superpixel = superpixel(superpixel_mask, superpixel_index, image, image_number, superpixel_level, LM_responses, LBP)
    # for j in min to max
    # j= 255 otherwise = 0 => mask
    # index ++
    # image
    # superpixel_level = i

#todo add mask of black bound



class superpixel:

    def __init__(self, mask, index, original_image, original_image_number, superpixel_level):
        self.mask = mask
        self.index = index
        self.center = self._find_center(self.mask)
        self.superpixel_level = superpixel_level
        self.origianl_image_number = original_image_number
        self.color_hist_feature = self._color_hist_calc(self.mask, original_image)
        self.HOG_feature = self._HOG_calc(self.mask, original_image)
        self.LM_feature = self._LM_feature_calc(self.mask, original_image)
        self.LBP_feature = self._LBP_feature_calc(self.mask, original_image)
    def __init__(self, path):
        # loading save superpixel
        self.path = path


    def _find_center(self, mask):
        center = "find center of mask"
        return center

    def calc_color_hist_feature(self, mask, origianl_image):
        color_hist_feature = "RGB=>LAB, find Masked histogram."
        return color_hist_feature

    def calc_HOG_feature(self, mask, origianl_image):
        HOG_feature = "RGB=>GRAY, Masked=> skimage HOG."
        return HOG_feature

    def calc_LM_feature(self, mask, origianl_image):
        LM_feature = "48 reponses =>masked  => 48 mean, 48 variance => vectorize"
        return LM_feature

    def calc_LBP_feature(self, mask, origianl_image):
        LBP_feature = "LBP => Masked => find histogram"
        return LBP_feature

    def set_color_hist_feature(self, hist_feature):
        self.color_hist_feature = hist_feature
        return None

    def set_HOG_feature(self, HOG_feature):
        self.HOG_feature = HOG_feature
        return None

    def set_LM_feature(self, LM_feature):
        self.LM_feature = LM_feature
        return None

    def set_LBP_feature(self, LBP_feature):
        self.LBP_feature = LBP_feature
        return None

    def save_superpixel(self, path):
        Savesuperepixel = "path + superpixel index => all information."
        return None
