import pickle
import os
import LM_Filter
import cv2
import numpy as np
from skimage.segmentation import slic
from skimage.feature import hog
from skimage.feature import local_binary_pattern
import LJY_utils
import LJY_visualize_tools

is_test = True

# for calculate LM feature...
LM_filter_bank = LM_Filter.makeLMfilters()

class superpixel:
    def __init__(self, save_path, mask=None, index=None, original_image_number=None, superpixel_level=None,
                 LM_features=None, LAB_image=None, hog_image=None, LBP_feature=None):

        if mask is None:
            # load saved superpixels
            #  _superpixel = superpixel(superpixel_path)

            self.load_superpixel(save_path)

        else:
            # for file name
            self.index = index
            self.superpixel_level = superpixel_level
            self.origianl_image_number = original_image_number
            self.save_path = save_path

            self.mask = mask
            self.center = self._find_center(self.mask)
            self.color_hist_feature = self.calc_color_hist_feature(self.mask, LAB_image)
            self.HOG_feature = self.calc_HOG_feature(self.mask, hog_image)
            self.LM_feature = self.calc_LM_feature(self.mask, LM_features)
            self.LBP_feature = self.calc_LBP_feature(self.mask, LBP_feature)
            self.saliency_value_WBU_CB = 0
            self.saliency_value_WBU_OB = 0
            self.saliency_value_WBU = 0
            self.saliency_value_STD = 0
            self.is_SAE_feature = False
    def set_save_path(self,new_save_path):
        self.save_path = new_save_path
    def _find_center(self, mask):
        [x,y]= np.asarray(mask)[1:].sum(0) / (len(mask) - 1)
        center = (int(x), int(y))
        return center

    def calc_color_hist_feature(self, mask, LAB_image):
        hist_data = []
        for i in range(1,len(mask)):
            hist_data.append(LAB_image[mask[i][0]][mask[i][1]])
        hist_data = np.asarray(hist_data)

        histogram = LJY_utils.three_channel_superpixel_interger_histogram_LAB(hist_data)
        # color histogram per superpixel [L hist , A hist, B hist at same bin]
        return histogram

    def calc_HOG_feature(self, mask, hog_image):
        hog_data = []
        for i in range(1, len(mask)):
            hog_data.append(hog_image[mask[i][0]][mask[i][1]])
        hog_data = np.asarray(hog_data)


        HOG_feature = LJY_utils.integer_histogram(hog_data, 0, 255)[0]
        # masked feature
        # todo : HOG image is not hog feature. fix it.
        return HOG_feature

    def calc_LM_feature(self, mask, LM_features):
        LM_feature = np.zeros(96)

        for filter_number in range(0, 48):
            lm_data = []
            for i in range(1, len(mask)):
                lm_data.append(LM_features[filter_number][mask[i][0]][mask[i][1]])
            lm_data = np.asarray(lm_data)

            # [mean, variance, mean, variance... per filter]
            LM_feature[filter_number * 2] = np.mean(lm_data)
            LM_feature[filter_number * 2 + 1] = np.var(lm_data)
        # LM feature per superpixel [96] vector
        return LM_feature

    def calc_LBP_feature(self, mask, LBP_feature):

        LBP_data = []
        for i in range(1, len(mask)):
            LBP_data.append(LBP_feature[mask[i][0]][mask[i][1]])
        LBP_data = np.asarray(LBP_data)

        hist_of_LBP_feature = LJY_utils.integer_histogram(LBP_data,0,255)[0]

        return hist_of_LBP_feature

    def set_SAE_feature(self, hist_feature, HOG_feature, LM_feature, LBP_feature):
        self.color_hist_feature = hist_feature
        self.HOG_feature = HOG_feature
        self.LM_feature = LM_feature
        self.LBP_feature = LBP_feature
        self.is_SAE_feature = True
        return None

    def set_WBU_saliency_value(self, WBU):
        self.saliency_value_WBU = WBU
        return None

    def set_STD_saliency_value(self, STD):
        self.saliency_value_STD = STD
        return None

    def save_superpixel(self):
        LJY_utils.make_dir(self.save_path)
        with open(os.path.join(self.save_path, "%04d_%d_%06d.txt" % (self.origianl_image_number,self.superpixel_level, self.index)), "wb") as fb:
            pickle.dump([self.mask, self.center, self.color_hist_feature, self.HOG_feature, self.LM_feature,
                         self.LBP_feature, self.saliency_value_WBU, self.saliency_value_STD, self.is_SAE_feature], fb)
        return None

    def load_superpixel(self, path):
        with open(path, "rb") as fb:
            data = pickle.load(fb)

        self.mask = data[0]
        self.center = data[1]
        self.color_hist_feature = data[2]
        self.HOG_feature = data[3]
        self.LM_feature = data[4]
        self.LBP_feature = data[5]
        self.saliency_value_WBU = data[6]
        self.saliency_value_STD = data[7]
        self.is_SAE_feature = data[8]

        self.save_path = os.path.dirname(path)
        self.origianl_image_number = int(os.path.basename(path).split('_')[0])
        self.index = int(os.path.basename(path).split('_')[2].split('.')[0])
        self.superpixel_level = int(os.path.basename(path).split('_')[1])

        return None
