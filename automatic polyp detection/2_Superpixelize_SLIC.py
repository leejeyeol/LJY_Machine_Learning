import numpy as np
import cv2
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
import LJY_visualize_tools

image_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/CVC-ClinicDB/Reflection Removed/1.png"

image = cv2.imread(image_path)


# find LM responses at this image
# find LBP responses at this image
for i in range(1, 6):
    #1~5 level segments...
    segments = slic(image, n_segments=50*i, compactness=50, convert2lab=True)
    # for j in min to max
    # j= 255 otherwise = 0 => mask
    # index ++
    # image

#todo add mask of black bound



class superpixel:
    def __init__(self, mask, index, original_image, LM_responses, LBP):
        self.mask = mask
        self.index = index
        self.center = self._find_center(self.mask)
        self.color_hist_feature = self._color_hist_calc(self.mask, original_image)
        self.HOG_feature = self._HOG_calc(self.mask, original_image)
        self.LM_feature = self._LM_feature_calc(self.mask, LM_responses)
        self.LBP_feature = self._LBP_feature_calc(self.mask, LBP)

        def _find_center(mask):
            center = "find center of mask"
            return center

        def _color_hist_calc(mask, origianl_image):
            color_hist_feature = "RGB=>LAB, find Masked histogram."
            return color_hist_feature

        def _HOG_calc(mask, origianl_image):
            color_hist_feature = "RGB=>GRAY, Masked=> skimage HOG."
            return color_hist_feature

        def _LM_feature_calc(mask, LM_responses):
            LM_feature = "48 reponses =>masked  => 48 mean, 48 variance => vectorize"
            return LM_feature

        def _LBP_feature_calc(mask, LBP):
            LBP_feature = "LBP => Masked => find histogram"
            return LBP_feature


