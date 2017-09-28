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

    def __init__(self, mask, index, original_image, original_image_number, superpixel_level, LM_filter_bank):
        self.mask = mask
        self.index = index
        self.center = self._find_center(self.mask)
        self.superpixel_level = superpixel_level
        self.origianl_image_number = original_image_number
        self.color_hist_feature = self.calc_color_hist_feature(self.mask, original_image)
        self.HOG_feature = self.calc_HOG_feature(self.mask, original_image)
        self.LM_feature = self.calc_LM_feature(self.mask, original_image, LM_filter_bank)
        self.LBP_feature = self.calc_LBP_feature(self.mask, original_image)
    def __init__(self):
        print("for loading superpixel")

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
        histogram = LJY_utils.three_channel_superpixel_interger_histogram(np.multiply(LAB_image, mask), mask)
        # color histogram per superpixel [L hist , A hist, B hist at same bin]
        return histogram

    def calc_HOG_feature(self, mask, origianl_image):

        hog_feature, hog_image = hog(cv2.cvtColor(origianl_image, cv2.COLOR_RGB2GRAY), visualise=True)
        masked_feature = hog_image[mask[:, :, 0]]

        # masked feature
        # todo : HOG image is not hog feature. fix it.
        return masked_feature

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
        LBP_feature = local_binary_pattern(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 8*3, 3, method='uniform')
        masked_LBP_feature = LBP_feature[intergrated_mask[:, :, 0]]

        return masked_LBP_feature

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





root_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/CVC-ClinicDB"
image_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/CVC-ClinicDB/Reflection Removed/000001.png"
boundary_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/CVC-ClinicDB/boundary_mask/boundary_mask_000001.png"
superpixel_save_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/CVC-ClinicDB/superpixels/"



image = cv2.imread(image_path)
image_number = int(LJY_utils.extract_filename_from_path(image_path))
boundary = cv2.imread(boundary_path,0)
superpixel_index = 0

for superpixel_level in range(1, 6):
    #1~5 level segments...
    segments = slic(image, n_segments=50*superpixel_level, compactness=50, convert2lab=True)
    for i in range(np.min(segments), np.max(segments)):
        # if image belongs to a segment and is not a boundary, the mask of the corresponding index is true.
        superpixel_mask = (segments[:] == i)
        boundary_mask = (boundary[:] == 255)

        if is_test:
            # superpixel && boundary.
            intergrated_mask = np.logical_and(superpixel_mask, boundary_mask)
            # [weight, height] => [weight, height, channel]. for calculate with 3channel image.
            intergrated_mask = intergrated_mask[:, :, np.newaxis]

            # check there is no masked image in this level.
            if np.any(intergrated_mask):

                superpixel = superpixel(intergrated_mask, superpixel_index, image, image_number, superpixel_level, LM_filter_bank)
                superpixel.save_superpixel(superpixel_save_path)
                superpixel_index = superpixel_index + 1