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
    def __init__(self, save_path, mask=None, index=None, original_image=None, original_image_number=None, superpixel_level=None, LM_filter_bank=None):

        if mask is None:
            # load saved superpixels
            #  _superpixel = superpixel(superpixel_path)

            self.load_superpixel(save_path)

        else:
            # for file name
            self.index = index
            self.superpixel_level = superpixel_level
            self.origianl_image_number = original_image_number
            self.save_path = os.path.join(save_path, "%06d" % (self.origianl_image_number))

            self.mask = mask
            self.center = self._find_center(self.mask)
            self.color_hist_feature = self.calc_color_hist_feature(self.mask, original_image)
            self.HOG_feature = self.calc_HOG_feature(self.mask, original_image)
            self.LM_feature = self.calc_LM_feature(self.mask, original_image, LM_filter_bank)
            self.LBP_feature = self.calc_LBP_feature(self.mask, original_image)

            self.saliency_value_WBU = None
            self.saliency_value_STD = None
            self.is_SAE_feature = False

    def _find_center(self, mask):
        sum_x = 0
        sum_y = 0
        num_of_true_pixel = 0
        for x in range(0, mask.shape[0]):
            for y in range(0, mask.shape[1]):

                if mask[x][y]:
                    sum_x = sum_x + x
                    sum_y = sum_y + y
                    num_of_true_pixel = num_of_true_pixel + 1
        center = (int(sum_x/num_of_true_pixel), int(sum_y/num_of_true_pixel))
        return center

    def calc_color_hist_feature(self, mask, origianl_image):
        LAB_image = cv2.cvtColor(origianl_image, cv2.COLOR_RGB2LAB)
        histogram = LJY_utils.three_channel_superpixel_interger_histogram_LAB(np.multiply(LAB_image, mask), mask)
        # color histogram per superpixel [L hist , A hist, B hist at same bin]
        return histogram

    def calc_HOG_feature(self, mask, origianl_image):


        hog_feature, hog_image = hog(cv2.cvtColor(origianl_image, cv2.COLOR_RGB2GRAY),block_norm='L2-Hys', visualise=True)
        masked_feature = hog_image[mask[:, :, 0]]
        HOG_feature = LJY_utils.integer_histogram(masked_feature, 0, 255)[0]
        # masked feature
        # todo : HOG image is not hog feature. fix it.
        return HOG_feature

    def calc_LM_feature(self, mask, origianl_image, LM_filter_bank):
        LM_feature = np.zeros(96)

        for filter_number in range(0, 48):
            # filtering
            masked_feature = cv2.filter2D(cv2.cvtColor(origianl_image, code=cv2.COLOR_RGB2GRAY), -1, kernel=LM_filter_bank[:, :, filter_number])
            # masked to filtered data
            masked_feature = masked_feature[mask[:, :, 0]]
            # [mean, variance, mean, variance... per filter]
            LM_feature[filter_number * 2] = np.mean(masked_feature)
            LM_feature[filter_number * 2 + 1] = np.var(masked_feature)
        # LM feature per superpixel [96] vector
        return LM_feature

    def calc_LBP_feature(self, mask, origianl_image):
        LBP_feature = local_binary_pattern(cv2.cvtColor(origianl_image, cv2.COLOR_BGR2GRAY), 8*3, 3, method='uniform')
        masked_LBP_feature = LBP_feature[mask[:, :, 0]]
        hist_of_LBP_feature = LJY_utils.integer_histogram(masked_LBP_feature,0,255)[0]

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
        with open(os.path.join(self.save_path, "%d_%06d.txt" % (self.superpixel_level, self.index)), "wb") as fb:
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
        self.origianl_image_number = int(os.path.basename(os.path.dirname(path)))
        self.index = int(os.path.basename(path).split('_')[1].split('.')[0])
        self.superpixel_level = int(os.path.basename(path).split('_')[0])

        return None



turn = False
if turn:

    root_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/CVC-ClinicDB"
    image_root_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/CVC-ClinicDB/Remove_Boundary"
    superpixel_save_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/CVC-ClinicDB/superpixels/"
    superpixel_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/CVC-ClinicDB/superpixels/000001/1_000000.txt"

    image_path_list = LJY_utils.get_file_paths(image_root_path, "/*.", ['png', 'PNG'])
    print("[make superpixels!]")

    for cnt, image_path in enumerate(image_path_list):
        print("[%d/%d]" % (cnt, len(image_path_list)))
        image = cv2.imread(image_path)
        image_number = int(LJY_utils.extract_filename_from_path(image_path))

        grayimage = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        boundary_mask = grayimage > 30


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
                if intergrated_mask.any():

                    _superpixel = superpixel(superpixel_save_path, intergrated_mask, superpixel_index, image, image_number, superpixel_level, LM_filter_bank)
                    _superpixel.save_superpixel()
                    #print(superpixel_index)
                    superpixel_index = superpixel_index + 1
                #print(i)


